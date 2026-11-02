import os
import sys
import json
import re
from dotenv import load_dotenv

# Load environment variables from .env file
# This looks for .env in current directory and parent directories
load_dotenv()

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEndpoint
from app.tools import get_equipment_status, log_inspection
from app.utils import load_and_embed_documents

# Initialize FastAPI
app = FastAPI()

# Global variables
vectorstore = None
llm = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, llm
    
    # Debug: Print if token is loaded
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        print(f"✓ HUGGINGFACEHUB_API_TOKEN loaded (starts with: {hf_token[:8]}...)")
    else:
        print("⚠ Warning: HUGGINGFACEHUB_API_TOKEN not found in environment")
    
    # Initialize LLM (Hugging Face)
    try:
        if not hf_token:
            print("  Please set it in .env file")
            llm = None
        else:
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                max_new_tokens=512,
                temperature=0.7,
                huggingfacehub_api_token=hf_token
            )
            print("✓ Hugging Face LLM initialized (Mistral-7B-Instruct)")
    except Exception as e:
        print(f"⚠ Warning: LLM initialization failed: {e}")
        llm = None
    
    # Initialize Vector Store
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "..", "data")
        vectorstore = load_and_embed_documents(data_path)
        print("✓ Vector store initialized")
    except Exception as e:
        print(f"⚠ Warning: Vector store initialization failed: {e}")
        vectorstore = None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    citations: List[str]
    tool_calls: List[Dict[str, Any]]

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store not initialized.")
    
    if not llm:
        return QuestionResponse(
            answer="LLM Service (Hugging Face) is not configured. Please set HUGGINGFACEHUB_API_TOKEN.",
            citations=[],
            tool_calls=[]
        )

    # 1. Retrieve Context for RAG
    try:
        docs = vectorstore.similarity_search(request.question, k=3)
        context_text = "\n\n".join([d.page_content for d in docs])
        citations = list(set([d.metadata.get("source", "Unknown") for d in docs]))
    except Exception as e:
        context_text = ""
        citations = []
        print(f"Retrieval error: {e}")

    # 2. Check if question needs tool calling (simple keyword detection)
    tool_used = None
    tool_result = None
    
    question_lower = request.question.lower()
    
    # Check for equipment status queries
    if any(keyword in question_lower for keyword in ["status", "safe", "alarm", "running", "operational"]):
        # Try to extract equipment ID
        eq_match = re.search(r'[A-Z]+-\d+', request.question)
        if eq_match:
            eq_id = eq_match.group()
            tool_result = get_equipment_status(eq_id)
            tool_used = {"name": "get_equipment_status", "input": eq_id, "output": tool_result}
    
    # Check for inspection logging
    elif any(keyword in question_lower for keyword in ["log", "record", "note", "inspection"]):
        # Try to extract equipment ID and note
        eq_match = re.search(r'[A-Z]+-\d+', request.question)
        note_match = re.search(r"'([^']+)'", request.question)
        if eq_match and note_match:
            eq_id = eq_match.group()
            note = note_match.group(1)
            tool_result = log_inspection(eq_id, note)
            tool_used = {"name": "log_inspection", "input": {"equipment_id": eq_id, "note": note}, "output": tool_result}

    # 3. Build Prompt
    tool_context = ""
    if tool_result:
        tool_context = f"\n\nTOOL RESULT: {json.dumps(tool_result, indent=2)}"
    
    prompt = f"""You are a helpful maintenance assistant. Answer questions using the context below.

CONTEXT FROM MANUALS:
{context_text}

SOURCES: {', '.join(citations) if citations else 'None'}
{tool_context}

RULES:
1. If the answer is in the CONTEXT above, use it and cite the SOURCE.
2. If the CONTEXT doesn't have the answer, say "I don't know based on the available manuals."
3. If tool results are provided above, use them to answer the question.
4. Always cite your sources when answering from manuals.

Question: {request.question}

Answer:"""

    # 4. Get LLM Response
    try:
        response = llm.invoke(prompt)
        
        return QuestionResponse(
            answer=response,
            citations=citations,
            tool_calls=[tool_used] if tool_used else []
        )
        
    except Exception as e:
        print(f"LLM execution error: {e}")
        # Fallback: Simple retrieval-based answer
        if context_text:
            return QuestionResponse(
                answer=f"Based on the manuals: {context_text[:500]}...",
                citations=citations,
                tool_calls=[tool_used] if tool_used else []
            )
        else:
            raise HTTPException(status_code=500, detail=f"LLM execution failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_connected": llm is not None,
        "vectorstore_loaded": vectorstore is not None
    }