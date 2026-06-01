# RoadLex AI — Punjab Traffic Laws Chatbot

A RAG-powered chatbot that answers questions about the Punjab Motor Vehicle Ordinance 1965.

## Stack

- **Backend:** FastAPI + Pinecone + Groq + fastembed (BAAI/bge-small-en-v1.5)
- **Frontend:** Single HTML file (traffic.html)
- **Deployed:** Render (backend + frontend served together)

## How It Works

1. User asks a question
2. Question is converted to a 384-dim vector using fastembed
3. Pinecone returns the most relevant chunks from the ordinance
4. Groq LLM generates a grounded answer from those chunks

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/traffic`

## Environment Variables

```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
GROQ_API_KEY=
```
