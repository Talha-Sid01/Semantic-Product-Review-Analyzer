import streamlit as st
import requests
import json
import time

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Semantic Review Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Premium" Look ---
st.markdown("""
<style>
    .report-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        height: 100%;
        color: #1f2937; /* Dark Grey Text for readability */
    }
    .pros-card {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
    }
    .cons-card {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
    }
    .verdict-card {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #1f2937; /* Dark Grey Text */
    }
    /* Force headings in cards to be dark */
    .report-card h3, .verdict-card h3 {
        color: #111827 !important;
        font-weight: 700;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report" not in st.session_state:
    st.session_state.report = None

# --- Sidebar ---
with st.sidebar:
    st.title("🛍️ Analyzer Config")
    st.markdown("This tool uses **Llama3** and **Vector Embeddings** to analyze product reviews deeply.")
    
    url_input = st.text_input("Enter Product URL", placeholder="https://amazon.com/...")
    analyze_btn = st.button("Analyze Reviews 🚀", use_container_width=True)
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.report = None
        st.rerun()

# --- Main Logic: Analyzer ---
st.title("🧠 Semantic Product Review Analyzer")

if analyze_btn and url_input:
    with st.spinner("🕷️ Scraping reviews & crunching vectors... (This may take a moment)"):
        try:
            response = requests.post(f"{API_BASE_URL}/analyze", json={"url": url_input})
            if response.status_code == 200:
                st.session_state.report = response.json()
                st.session_state.messages = [] # Reset chat on new analysis
                st.success("Analysis Complete!")
            else:
                st.error(f"Analysis failed: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- Display Report ---
if st.session_state.report:
    report = st.session_state.report
    
    st.markdown("### 📊 Buying Decision Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="report-card pros-card">
            <h3>✅ Pros</h3>
            <ul>
                {"".join(f"<li>{item}</li>" for item in report.get('pros', []))}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="report-card cons-card">
            <h3>❌ Cons</h3>
            <ul>
                {"".join(f"<li>{item}</li>" for item in report.get('cons', []))}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="verdict-card">
        <h3>⚖️ Final Verdict</h3>
        <p>{report.get('verdict', 'No verdict available.')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

# --- Chat Interface ---
st.subheader("💬 Chat with the Reviews")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about battery, screen, or value..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    if not st.session_state.report:
        response_text = "Please analyze a product URL first!"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                chat_res = requests.post(
                    f"{API_BASE_URL}/chat", 
                    json={"question": prompt}
                )
                if chat_res.status_code == 200:
                    answer = chat_res.json()["answer"]
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    message_placeholder.markdown("Error connecting to backend.")
            except Exception as e:
                message_placeholder.markdown(f"Error: {e}")

if not st.session_state.report and not url_input:
    st.info("Start by pasting a URL in the sidebar!")
