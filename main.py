from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse          # ← ADD
from fastapi.staticfiles import StaticFiles         # ← ADD

from models.schemas import ChatRequest, ChatResponse
from services.embedding_service import get_embedding
from services.pinecone_service import search_pinecone
from services.llm_service import get_llm_response

app = FastAPI(
    title="RoadLex AI — Traffic Laws Chatbot",
    description="RAG chatbot powered by Punjab Motor Vehicle Ordinance 1965",
    version="1.0.0"
)

# Allow frontend (traffic.html) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SERVE HTML FILES THROUGH FASTAPI ─────────────────────────── ADD ──
@app.get("/traffic", include_in_schema=False)
def serve_frontend():
    return FileResponse("traffic.html")

@app.get("/login", include_in_schema=False)
def serve_login():
    return FileResponse("login.html")

# ── PRINT CLICKABLE LINKS ON STARTUP ─────────────────────────── ADD ──
@app.on_event("startup")
async def print_startup_banner():
    print("\n" + "━" * 52)
    print("  🚀  RoadLex AI is running!")
    print("  🌐  App      →  http://127.0.0.1:8000/traffic")
    print("  📄  API Docs →  http://127.0.0.1:8000/docs")
    print("━" * 52 + "\n")
# ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RoadLex AI backend is running ✅"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main RAG endpoint.
    Flow: question → embedding → pinecone search → groq LLM → answer
    """
    try:
        # Step 1: Convert question to embedding
        embedding = get_embedding(request.question)

        # Step 2: Search Pinecone for relevant chunks
        context = search_pinecone(embedding, top_k=5)

        # Step 3: Send context + question to Groq LLM
        answer = get_llm_response(request.question, context)

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))