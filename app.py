import sys
import os

# pysqlite3 shim — only needed on Linux (e.g. Streamlit Cloud) where the
# system SQLite is too old for ChromaDB. On Windows/macOS this is skipped.
if sys.platform.startswith("linux"):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass  # pysqlite3-binary not installed, continue with system sqlite3
import time
import random
import tempfile
import streamlit as st
import shutil
import json
import re
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Scraping helpers — ordered by reliability
# ─────────────────────────────────────────────

BROWSER_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
]


def _clean_html_to_text(html_content: bytes | str) -> str:
    """Parse HTML and extract clean readable text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "svg", "form"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        lines = (ln.strip() for ln in text.splitlines())
        chunks = (ph.strip() for ln in lines for ph in ln.split("  "))
        return "\n".join(c for c in chunks if c)
    except Exception:
        return ""


def _scrape_with_requests(url: str, retries: int = 3) -> Optional[str]:
    """Strategy 1: plain requests with rotating browser headers + exponential back-off."""
    session = requests.Session()
    for attempt in range(retries):
        try:
            headers = random.choice(BROWSER_HEADERS)
            resp = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            text = _clean_html_to_text(resp.content)
            if len(text) > 200:
                return text
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            time.sleep(wait)
    return None


def _scrape_with_cloudscraper(url: str) -> Optional[str]:
    """Strategy 2: cloudscraper — bypasses Cloudflare JS challenges."""
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
        resp = scraper.get(url, timeout=20)
        resp.raise_for_status()
        text = _clean_html_to_text(resp.content)
        return text if len(text) > 200 else None
    except ImportError:
        return None
    except Exception:
        return None


def _scrape_with_trafilatura(url: str) -> Optional[str]:
    """Strategy 3: trafilatura — article/review content extractor."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=True,
                include_tables=True,
                no_fallback=False,
            )
            return text if text and len(text) > 200 else None
    except ImportError:
        return None
    except Exception:
        return None


def _scrape_with_newspaper(url: str) -> Optional[str]:
    """Strategy 4: newspaper3k — good for article/blog/review pages."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        return text if len(text) > 200 else None
    except ImportError:
        return None
    except Exception:
        return None


def _scrape_with_selenium(url: str) -> Optional[str]:
    """Strategy 5: headless Chromium via selenium — handles JS-heavy pages."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={random.choice(BROWSER_HEADERS)['User-Agent']}")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(random.uniform(2, 4))   # let JS settle
            page_source = driver.page_source
        finally:
            driver.quit()

        text = _clean_html_to_text(page_source)
        return text if len(text) > 200 else None
    except Exception:
        return None


def scrape_url_robust(url: str, log_fn=None) -> str:
    """
    Try multiple scraping strategies in order.
    Returns extracted text or raises RuntimeError.
    """
    strategies = [
        ("requests (rotating headers)", _scrape_with_requests),
        ("cloudscraper (anti-bot bypass)", _scrape_with_cloudscraper),
        ("trafilatura (content extractor)", _scrape_with_trafilatura),
        ("newspaper3k (article parser)", _scrape_with_newspaper),
        ("selenium (headless browser)", _scrape_with_selenium),
    ]

    for name, fn in strategies:
        if log_fn:
            log_fn(f"⏳ Trying strategy: **{name}**...")
        try:
            result = fn(url)
            if result and len(result.strip()) > 200:
                if log_fn:
                    log_fn(f"✅ Success with **{name}**")
                return result.strip()
        except Exception:
            continue

    raise RuntimeError(
        "All scraping strategies failed. The site may be heavily protected, "
        "require login, or use advanced anti-bot measures. "
        "Try a direct product review aggregator URL (e.g., Amazon, GSMArena, Rtings)."
    )


# ─────────────────────────────────────────────
# Core Analyzer
# ─────────────────────────────────────────────

class ReviewAnalyzer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables.")

        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=self.groq_api_key,
        )

        self.embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=self.hf_api_key,
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        self.persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db_reviews")

    def ingest_and_index(self, url: str, log_fn=None) -> str:
        """Scrape → chunk → embed → store in ChromaDB."""
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL. Please include http:// or https://")

        # Scrape
        raw_text = scrape_url_robust(url, log_fn=log_fn)

        # Trim to a sensible max to avoid token overflow
        max_chars = 120_000
        if len(raw_text) > max_chars:
            raw_text = raw_text[:max_chars]

        doc = Document(page_content=raw_text, metadata={"source": url})

        # Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents([doc])

        if not splits:
            raise ValueError("No text could be extracted from the page.")

        # Wipe old DB
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception:
                pass

        # Embed & store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="product_reviews",
        )

        return f"Indexed {len(splits)} chunks successfully."

    def generate_summary(self) -> Dict[str, object]:
        """RAG-based buying decision report."""
        if not hasattr(self, "vectorstore"):
            raise ValueError("Vectorstore not initialized. Please analyze a URL first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke(
            "What are the pros, cons, ratings, and overall verdict of this product?"
        )
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        system_prompt = """You are an expert product reviewer.
Analyze the provided context (product reviews / details) and produce a "Buying Decision Report".

Output MUST be valid JSON with exactly these keys:
- "pros": List of strings (at least 3 if available).
- "cons": List of strings (at least 2 if available).
- "verdict": A concise paragraph recommending whether to buy or not.

If the page doesn't contain product/review content, set verdict to explain that.
Do NOT include markdown code fences in your output."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context:\n{context}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"context": context_text})
            clean = re.sub(r"```(?:json)?", "", result).strip().strip("`").strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                start, end = clean.find("{"), clean.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(clean[start:end])
                raise
        except Exception as e:
            return {
                "pros": ["Could not parse response"],
                "cons": [str(e)],
                "verdict": "Report generation failed. The page may not contain enough review content.",
            }

    def chat_query(self, question: str) -> str:
        """RAG-based chat over indexed reviews."""
        if not hasattr(self, "vectorstore"):
            return "Please process a URL first."

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        template = """Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't see that mentioned in the reviews."
Keep answers concise and helpful.

Context:
{context}

Question: {question}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)


# ─────────────────────────────────────────────
# Streamlit UI  (unchanged layout/structure)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Product Review Analyzer",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_resource
def get_analyzer():
    return ReviewAnalyzer()


def main():
    st.title("🛍️ Semantic Product Review Analyzer")
    st.markdown("Analyze product reviews using RAG and LLMs to make better buying choices!")

    try:
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Make sure GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN are set in your .env file.")
        st.stop()

    url = st.text_input("Enter Product URL", placeholder="https://example.com/product")

    col_submit, _ = st.columns([1, 4])
    with col_submit:
        analyze_clicked = st.button("Analyze Product", use_container_width=True)

    if analyze_clicked:
        if not url:
            st.warning("Please enter a valid URL.")
        else:
            status_container = st.empty()
            log_lines: List[str] = []

            def log_fn(msg: str):
                log_lines.append(msg)
                status_container.markdown("\n\n".join(log_lines))

            with st.spinner("Processing the product page and reviews… This may take a minute."):
                try:
                    analyzer.ingest_and_index(url, log_fn=log_fn)
                    report = analyzer.generate_summary()
                    st.session_state["report"] = report
                    st.session_state["analyzed"] = True
                    status_container.empty()
                    st.success("Analysis complete!")
                except Exception as e:
                    status_container.empty()
                    st.error(f"Error during analysis: {e}")

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

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., What do users think about the battery life?"):
        if not hasattr(analyzer, "vectorstore"):
            st.warning("Please analyze a product URL first before asking questions.")
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            with st.spinner("Searching reviews…"):
                response = analyzer.chat_query(prompt)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()
