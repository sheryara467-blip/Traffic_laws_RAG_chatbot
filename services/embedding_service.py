# from sentence_transformers import SentenceTransformer

# Load model once at startup (not on every request)
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def get_embedding(text: str) -> list:
#     """Convert a text string into a 384-dim embedding vector."""
#     embedding = model.encode(text, convert_to_numpy=True)
#     return embedding.tolist()



from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_embedding(text: str) -> list:
    embeddings = list(model.embed([text]))  # ← .embed() not .encode()
    return embeddings[0].tolist()