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


def generate_question(topic: str, mode: str) -> dict:
    """Use Groq LLM to auto-generate a poll or opinion question from a topic.
    Returns {"question": str, "options": list[str]} — options empty for opinion mode.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set. Add it as an environment variable to use AI generation."}

    if mode == "poll":
        prompt = f"""Generate one engaging multiple-choice poll question about the topic: "{topic}".
Respond ONLY in this exact format, nothing else:
Question: <the question>
Options: <option1>, <option2>, <option3>, <option4>"""
    else:
        prompt = f"""Generate one open-ended opinion question about the topic: "{topic}"
that invites a short free-text answer from an audience.
Respond ONLY in this exact format, nothing else:
Question: <the question>"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
    )
    text = response.choices[0].message.content.strip()

    question, options = "", []
    for line in text.splitlines():
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("options:"):
            options = [o.strip() for o in line.split(":", 1)[1].split(",") if o.strip()]

    if not question:
        return {"error": "Could not parse AI response. Please try again or type manually."}

    return {"question": question, "options": options}
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
