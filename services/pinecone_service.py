from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize Pinecone client once at startup
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def search_pinecone(embedding: list, top_k: int = 5) -> str:
    """
    Query Pinecone with the embedding vector.
    Returns the top matching chunks joined as a single context string.
    """
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Extract text from each matched chunk
    context_parts = []
    for match in results["matches"]:
        text = match["metadata"].get("text", "")
        page = match["metadata"].get("page_num", "?")
        if text:
            context_parts.append(f"[Page {page}]: {text}")

    return "\n\n".join(context_parts)