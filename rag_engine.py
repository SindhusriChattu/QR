import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Loaded once and reused across calls (small, fast model)
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def build_index(opinions: list[str]):
    """Embed all opinion texts and build a FAISS similarity index."""
    if not opinions:
        return None, None
    model = get_embed_model()
    embeddings = model.encode(opinions, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)
    return index, embeddings


def retrieve_relevant(question: str, opinions: list[str], index, top_k: int = 5):
    """Return the top_k opinions most relevant to the question."""
    if index is None or not opinions:
        return []
    model = get_embed_model()
    q_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    k = min(top_k, len(opinions))
    scores, idxs = index.search(q_embedding, k)
    return [opinions[i] for i in idxs[0] if i != -1]


def generate_grounded_answer(question: str, retrieved_opinions: list[str]) -> str:
    """Call Groq LLM to answer the question, grounded only in retrieved opinions."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY not set. Add it as an environment variable to enable AI answers."

    if not retrieved_opinions:
        return "No responses yet to answer from."

    context = "\n".join(f"- {op}" for op in retrieved_opinions)
    prompt = f"""You are summarizing live audience opinions for a presenter.
Answer the question using ONLY the responses below. Do not invent anything
not supported by them. Keep the answer to 3-4 sentences.

Audience responses:
{context}

Question: {question}
Answer:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
