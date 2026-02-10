import os
import shutil
import uuid
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (ensure main.py loads them, or load here)
from dotenv import load_dotenv
load_dotenv()

class ReviewAnalyzer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables")
        
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
        # We will initialize the vector store dynamically per session or universally?
        # The prompt implies a single user flow or reusable. 
        # For simplicity and robustness, we'll reload the collection or create new ones.
        # Let's use a single persistent client for now.
        
    def scrape_url(self, url: str) -> List[Document]:
        """
        Scrapes the URL pretending to be a real browser.
        Returns a list of LangChain Documents.
        """
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Create a document
            # We associate the URL as metadata
            return [Document(page_content=clean_text, metadata={"source": url})]
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            raise e

    def ingest_and_index(self, url: str) -> str:
        """
        Scrapes the URL, chunks the text, and stores in ChromaDB.
        Returns the collection name or status.
        """
        # 1. Scrape
        docs = self.scrape_url(url)
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # 3. Store in Chroma
        # We'll use a unique collection name per URL to avoid mixing contexts contextually 
        # or just clear the DB. For this app, let's reset or use a specific collection.
        # To keep it simple and robust for this "demo" app, we might clear previous data 
        # or use a fixed collection name and delete valid data first.
        
        collection_name = "product_reviews"
        
        # Clean existing collection to ensure we Analyze ONLY the new product
        # Note: In production, you'd handle multiple users via session_ids.
        # Here we assume single-user local usage as per prompt context.
        if os.path.exists(self.persist_directory):
            try:
                # Optional: Delete DB folder to reset (Brute force reset)
                # This ensures no stale data.
                shutil.rmtree(self.persist_directory)
            except Exception as e:
                print(f"Cleanup warning: {e}")
                
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        return "Indexed successfully"

    def generate_summary(self) -> Dict[str, str]:
        """
        Retrieves relevant chunks and generates a Buying Decision Report.
        """
        if not hasattr(self, 'vectorstore'):
            raise ValueError("Vectorstore not initialized. Please analyze a URL first.")
            
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Query specifically for summary content
        relevant_docs = retriever.invoke("What are the pros, cons, and verdict of this product reviews?")
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Prompt
        system_prompt = """You are an expert product reviewer. 
        Analyze the provided context (which are reviews/details of a product) and produce a "Buying Decision Report".
        
        Output MUST be in Valid JSON format with keys: "pros", "cons", "verdict".
        - "pros": List of strings.
        - "cons": List of strings.
        - "verdict": A concise paragraph summary recommending to buy or not.
        
        If the context doesn't look like a product page, state so in the verdict.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context: {context}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"context": context_text})
            # Naive JSON extraction (LLM might return backticks)
            import json
            import re
            
            # Clean markdown code blocks if present
            clean_result = re.sub(r'```json\s*|\s*```', '', result)
            return json.loads(clean_result)
        except Exception as e:
            return {
                "pros": ["Error parsing response"],
                "cons": [str(e)],
                "verdict": "Could not generate report."
            }

    def chat_query(self, question: str) -> str:
        """
        Chat with the processed reviews using RAG.
        """
        if not hasattr(self, 'vectorstore'):
            return "Please process a URL first."
            
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        template = """Answer the question based ONLY on the following context. 
        If the answer is not in the context, say "I don't see that mentioned in the reviews."
        Keep answers concise and helpful.
        
        Context:
        {context}
        
        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(question)
