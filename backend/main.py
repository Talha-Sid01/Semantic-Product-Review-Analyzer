from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from backend.rag_engine import ReviewAnalyzer
import uvicorn
import os

# Initialize Analyzer Globally
analyzer = ReviewAnalyzer()

# Pydantic Models
class AnalyzeRequest(BaseModel):
    url: str

class AnalyzeResponse(BaseModel):
    pros: list[str]
    cons: list[str]
    verdict: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Initializing Review Analyzer Backend...")
    yield
    # Shutdown logic
    print("Shutting down...")

app = FastAPI(title="Semantic Product Review Analyzer API", lifespan=lifespan)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_product(request: AnalyzeRequest):
    try:
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
            
        print(f"Processing URL: {request.url}")
        
        # 1. Scrape & Index
        analyzer.ingest_and_index(request.url)
        
        # 2. Generate Report
        report = analyzer.generate_summary()
        
        return report
    except Exception as e:
        print(f"Error in /analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_product(request: ChatRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question is required")
            
        # Call RAG engine
        response_text = analyzer.chat_query(request.question)
        return {"answer": response_text}
        
    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
