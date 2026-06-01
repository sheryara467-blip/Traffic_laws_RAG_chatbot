import requests
from config import GROQ_API_KEY

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated model name as per Groq's latest offerings

def get_llm_response(question: str, context: str) -> str:
    """
    Send the user question + retrieved context to Groq API.
    Returns the AI-generated answer as a string.
    """

    system_prompt = """You are RoadLex AI, an expert assistant on the Punjab Motor Vehicle Ordinance 1965 and Pakistani traffic laws.

Answer the user's question using ONLY the context provided below.
- Be clear, accurate, and helpful
- Mention relevant section numbers when available
- If the context does not contain enough information, say: "I could not find specific information about this in the ordinance."
- Keep your answer concise and well-structured"""

    user_message = f"""Context from Punjab Motor Vehicle Ordinance 1965:
{context}

Question: {question}"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens":  512
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]