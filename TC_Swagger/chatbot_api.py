from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

app = FastAPI(
    title="Tamil Bible Chatbot",
    description="RAG chatbot using Gemini + FAISS + SentenceTransformers",
    version="1.0"
)

# Add CORS middleware to support Swagger UI & frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for /chat endpoint
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Global variables
embedding_model = None
model = None
index = None
texts = []
sources = []

@app.get("/health", tags=["Utils"])
def health_check():
    return {"status": "API is running "}

def load_verses(file_path="cleaned_tamil_bible_verses.jsonl"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def create_chunks(verses, chunk_size=5):
    chunks = []
    for i in range(0, len(verses), chunk_size):
        chunk_text = " ".join(v["text"] for v in verses[i:i + chunk_size])
        source = ", ".join(f'{v["book"]} {v["chapter"]}:{v["verse"]}' for v in verses[i:i + chunk_size])
        chunks.append({"text": chunk_text, "source": source})
    return chunks

def load_embedding_and_index():
    global embedding_model, index, texts, sources
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    verses = load_verses()
    chunks = create_chunks(verses)
    texts = [chunk['text'] for chunk in chunks]
    sources = [chunk['source'] for chunk in chunks]

    print("Encoding text and building FAISS index...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print(" FAISS index built.")

def load_gemini():
    global model
    print("Loading Gemini model...")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "gemini_api_key_here")) 
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    print(" Gemini model loaded.")

@app.post("/chat", summary="Ask a Tamil Bible question", tags=["Chatbot"])
def chat(request: QueryRequest):
    try:
        global embedding_model, index, model, texts, sources
        if embedding_model is None or index is None:
            load_embedding_and_index()
        if model is None:
            load_gemini()

        query_embedding = embedding_model.encode([request.query])
        D, I = index.search(np.array(query_embedding), request.top_k)

        retrieved_texts = [texts[i] for i in I[0]]
        retrieved_sources = [sources[i] for i in I[0]]

        context = "\n\n".join(f"📖 {text}" for text in retrieved_texts)
        prompt = f"""
நீ ஒரு தமிழ் வேதாகம உதவியாளர். பின்வரும் வேதாகம உரைகளைப் பயன்படுத்தி, பயனரின் கேள்விக்கு நட்பு மற்றும் ஆதரவாக பதிலளி.

📚 உரைகள்:
{context}

🙋‍♂️ பயனர்: {request.query}
🤖 பதில்:"""

        response = model.generate_content(prompt)

        return {
            "response": (response.text or "").strip().replace("\n", "\n\n"),
            "sources": retrieved_sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Internal Error: {str(e)}")
