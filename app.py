import os
import sqlite3
import uuid
import streamlit as st
import shutil
import json
import re
import requests
from typing import List, Dict, Annotated, TypedDict, Optional
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables.config import RunnableConfig
from langsmith import traceable

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class ReviewAnalyzer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables.")
        
        # Initialize LLM
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=self.groq_api_key
        )
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=self.hf_api_key,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Vector Store Path
        self.persist_directory = "./chroma_db"
        
        # --- LangGraph Setup for Chat Memory ---
        self.conn = sqlite3.connect("chat_history.db", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        
        # Setup the necessary tables
        self.memory.setup()
        
        # Add metadata table for historical URLs and Reports linking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT PRIMARY KEY,
                url TEXT,
                report_json TEXT
            )
        """)
        try:
            self.conn.execute("ALTER TABLE thread_metadata ADD COLUMN parent_run_id TEXT")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        # ---------------------------------------

    @traceable(run_type="chain", name="chatbot_node")
    def _chatbot_node(self, state: State, config: RunnableConfig):
        messages = state["messages"]
        last_msg = messages[-1].content
        thread_id = config["configurable"]["thread_id"]
        
        # Retrieve the relevant collection for this thread
        collection_name = f"review_{thread_id.replace('-', '_')}"
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        try:
            if not vectorstore._collection.count():
                return {"messages": [AIMessage(content="Please analyze a product URL first.")]}
        except Exception:
            return {"messages": [AIMessage(content="Please analyze a product URL first.")]}
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(last_msg)
        context = "\n\n".join(d.page_content for d in docs)
        
        system_prompt = f"Answer the question based ONLY on the following context. If the answer is not in the context, say 'I don't see that mentioned in the reviews.' Keep answers concise and helpful.\n\nContext:\n{context}"
        
        # Filter previous SystemMessages to avoid context length bloat
        filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        invoke_messages = [SystemMessage(content=system_prompt)] + filtered_messages
        
        response = self.llm.invoke(invoke_messages)
        return {"messages": [response]}
        
    def get_chat_history(self, thread_id: str) -> List[BaseMessage]:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.graph.get_state(config)
            if state and "messages" in state.values:
                return state.values["messages"]
            return []
        except Exception as e:
            return []

    def get_all_threads(self) -> List[Dict[str, str]]:
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            t_rows = cursor.fetchall()
            all_tids = [r[0] for r in t_rows]
            
            cursor.execute("SELECT thread_id, url FROM thread_metadata")
            m_rows = cursor.fetchall()
            meta_dict = {r[0]: r[1] for r in m_rows}
            
            res = []
            for tid in all_tids:
                res.append({"thread_id": tid, "url": meta_dict.get(tid)})
            return res
        except Exception as e:
            return []
            
    def get_thread_metadata(self, thread_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT url, report_json, parent_run_id FROM thread_metadata WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "url": row[0], 
                    "report": json.loads(row[1]),
                    "parent_run_id": row[2] if len(row) > 2 else None
                }
        except Exception:
            pass
        return None

    def delete_chat(self, thread_id: str):
        try:
            self.conn.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,))
            self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self.conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            self.conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
            self.conn.commit()
            
            # Clean up vector store
            collection_name = f"review_{thread_id.replace('-', '_')}"
            client = Chroma(persist_directory=self.persist_directory)._client
            client.delete_collection(name=collection_name)
        except Exception as e:
            pass

    @traceable(run_type="chain", name="chat_query_stream")
    def chat_query_stream(self, question: str, thread_id: str, parent_run_id: Optional[str] = None, **kwargs):
        """
        Chat with the processed reviews using token-by-token streaming & LangGraph persistence.
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        collection_name = f"review_{thread_id.replace('-', '_')}"
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        try:
            if not vectorstore._collection.count():
                yield "Please analyze a product URL first."
                return
        except Exception:
            yield "Please analyze a product URL first."
            return
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        system_prompt = f"Answer the question based ONLY on the following context. If the answer is not in the context, say 'I don't see that mentioned in the reviews.' Keep answers concise and helpful.\n\nContext:\n{context}"
        
        state = self.graph.get_state(config)
        messages = state.values.get("messages", []) if state and hasattr(state, "values") else []
        filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        
        invoke_messages = [SystemMessage(content=system_prompt)] + filtered_messages + [HumanMessage(content=question)]
        
        full_response = ""
        for chunk in self.llm.stream(invoke_messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
                
        # Update graph memory seamlessly
        self.graph.update_state(config, {"messages": [HumanMessage(content=question), AIMessage(content=full_response)]}, as_node="chatbot")
        
    @traceable(run_type="tool", name="scrape_url")
    def scrape_url(self, url: str) -> List[Document]:
        ual = UserAgent()
        headers = {'User-Agent': ual.random}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return [Document(page_content=clean_text, metadata={"source": url})]
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            raise e

    @traceable(run_type="chain", name="ingest_and_index")
    def ingest_and_index(self, url: str, thread_id: str) -> str:
        docs = self.scrape_url(url)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        collection_name = f"review_{thread_id.replace('-', '_')}"
        
        # Initialize chroma collection mapping specifically for this thread
        Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        return "Indexed successfully"

    @traceable(run_type="chain", name="generate_summary")
    def generate_summary(self, url: str, thread_id: str) -> Dict[str, str]:
        collection_name = f"review_{thread_id.replace('-', '_')}"
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke("What are the pros, cons, and verdict of this product reviews?")
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        system_prompt = """You are an expert product reviewer. 
        Analyze the provided context (which are reviews/details of a product) and produce a "Buying Decision Report".
        
        Output MUST be in Valid JSON format with keys: "pros", "cons", "verdict".
        - "pros": List of strings.
        - "cons": List of strings.
        - "verdict": A concise paragraph summary recommending to buy or not.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context: {context}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"context": context_text})
            clean_result = re.sub(r'```json\s*|\s*```', '', result)
            
            try:
                report = json.loads(clean_result)
            except json.JSONDecodeError:
                start = clean_result.find('{')
                end = clean_result.rfind('}') + 1
                report = json.loads(clean_result[start:end])
                
        except Exception as e:
            report = {
                "pros": ["Error parsing response"],
                "cons": [str(e)],
                "verdict": "Could not generate report."
            }
            
        return report

    @traceable(run_type="chain", name="Project Trace")
    def process_product(self, url: str, thread_id: str) -> tuple[Dict[str, str], Optional[str]]:
        from langsmith.run_helpers import get_current_run_tree
        
        # Ingest and summarize (these are traced as children automatically)
        self.ingest_and_index(url, thread_id)
        report = self.generate_summary(url, thread_id)
        
        rt = get_current_run_tree()
        parent_run_id = str(rt.id) if rt else None
        
        # Extract and Save Metadata permanently to SQLite
        self.conn.execute(
            "INSERT OR REPLACE INTO thread_metadata (thread_id, url, report_json, parent_run_id) VALUES (?, ?, ?, ?)",
            (thread_id, url, json.dumps(report), parent_run_id)
        )
        self.conn.commit()
        return report, parent_run_id

# --- Streamlit Setup ---

st.set_page_config(page_title="Semantic Product Review Analyzer", page_icon="🛍️", layout="wide")

@st.cache_resource
def get_analyzer():
    """Caches the ReviewAnalyzer so it doesn't re-initialize on every rerun"""
    return ReviewAnalyzer()

def main():
    st.title("🛍️ Semantic Product Review Analyzer")
    st.markdown("Analyze product reviews using RAG and LLMs to make better buying choices!")
    
    try:
        from typing import Optional
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Make sure the GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN are set in your .env file.")
        st.stop()
        
    # --- Check for Thread ID (Persists per Session) ---
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
        
    with st.sidebar:
        st.header("💬 Chat History")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["thread_id"] = str(uuid.uuid4())
            if "report" in st.session_state:
                del st.session_state["report"]
            if "analyzed" in st.session_state:
                del st.session_state["analyzed"]
            if "url_input" in st.session_state:
                del st.session_state["url_input"]
            if "parent_run_id" in st.session_state:
                del st.session_state["parent_run_id"]
            st.rerun()

        st.divider()
        st.write("Previous Chats:")
        threads = analyzer.get_all_threads()
        if not threads:
            st.info("No previous chats found.")
        else:
            # Show latest threads first
            for t_info in reversed(threads):
                t_id = t_info["thread_id"]
                url = t_info["url"]
                
                is_active = (t_id == st.session_state["thread_id"])
                
                # Format label nicely with url snippet or fallback
                if url:
                    # e.g., Extract domain or a slug from URL simply
                    snippet = url.split("://")[-1].split("/")[0] if "://" in url else url[:15]
                    label_suffix = f" - {snippet[:15]}"
                else:
                    label_suffix = " (Empty)"
                    
                btn_label = f"{t_id[:8]}{label_suffix}" + (" 🟢" if is_active else "")
                
                col_btn, col_del = st.columns([4, 1])
                with col_btn:
                    if st.button(btn_label, key=f"btn_{t_id}", use_container_width=True):
                        if not is_active:
                            st.session_state["thread_id"] = t_id
                            
                            # Try to load existing metadata
                            meta = analyzer.get_thread_metadata(t_id)
                            if meta:
                                st.session_state["report"] = meta["report"]
                                st.session_state["analyzed"] = True
                                st.session_state["url_input"] = meta["url"]
                                st.session_state["parent_run_id"] = meta.get("parent_run_id")
                            else:
                                if "report" in st.session_state: del st.session_state["report"]
                                if "analyzed" in st.session_state: del st.session_state["analyzed"]
                                if "url_input" in st.session_state: del st.session_state["url_input"]
                                if "parent_run_id" in st.session_state: del st.session_state["parent_run_id"]
                                    
                            st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{t_id}", help="Delete this chat"):
                        analyzer.delete_chat(t_id)
                        if is_active:
                            st.session_state["thread_id"] = str(uuid.uuid4())
                            if "report" in st.session_state: del st.session_state["report"]
                            if "analyzed" in st.session_state: del st.session_state["analyzed"]
                            if "url_input" in st.session_state: del st.session_state["url_input"]
                            if "parent_run_id" in st.session_state: del st.session_state["parent_run_id"]
                        st.rerun()
        
    thread_id = st.session_state["thread_id"]
        
    default_url = st.session_state.get("url_input", "")
    url = st.text_input("Enter Product URL", value=default_url, placeholder="https://example.com/product")
    
    col_submit, _ = st.columns([1, 4])
    with col_submit:
        analyze_clicked = st.button("Analyze Product", use_container_width=True)
    
    if analyze_clicked:
        if not url:
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Processing the product page and reviews... This may take a minute."):
                try:
                    # Index specifically for this Thread and get parent run
                    report, parent_run_id = analyzer.process_product(url, thread_id)
                    
                    st.session_state["report"] = report
                    st.session_state["parent_run_id"] = parent_run_id
                    st.session_state["analyzed"] = True
                    st.session_state["url_input"] = url
                    
                    st.success("Analysis complete!")
                    # Refresh the app immediately to update the active sidebar URL label
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    # Display Report if analyzed
    if st.session_state.get("analyzed"):
        report = st.session_state.get("report", {})
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Pros")
            for pro in report.get("pros", []):
                st.markdown(f"- {pro}")
                
        with col2:
            st.subheader("❌ Cons")
            for con in report.get("cons", []):
                st.markdown(f"- {con}")
                
        st.subheader("⚖️ Verdict")
        st.info(report.get("verdict", "No verdict generated."))
                    
    st.divider()
    
    st.subheader("💬 Ask questions about the product")
    
    # Load and render chat history directly from LangGraph SQLite memory
    history = analyzer.get_chat_history(thread_id)
    for msg in history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
            
    if prompt := st.chat_input("E.g., What do users think about the battery life?"):
        if not st.session_state.get("analyzed"):
            st.warning("Please analyze a product URL first before asking questions.")
        else:
            # 1. Output the user message locally
            st.chat_message("user").write(prompt)
            
            # 2. Output the AI response incrementally utilizing streaming generators
            with st.chat_message("assistant"):
                p_run_id = st.session_state.get("parent_run_id")
                kwargs = {}
                if p_run_id:
                    kwargs["langsmith_extra"] = {"parent_run_id": p_run_id}
                    
                response = st.write_stream(analyzer.chat_query_stream(
                    prompt, 
                    thread_id, 
                    parent_run_id=p_run_id, 
                    **kwargs
                ))

if __name__ == "__main__":
    main()
