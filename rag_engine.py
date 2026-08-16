import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


def build_index(opinions: list[str]):
    """Vectorize all opinion texts using TF-IDF for similarity-based retrieval."""
    if not opinions:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(opinions)
    return vectorizer, tfidf_matrix


def retrieve_relevant(question: str, opinions: list[str], index, top_k: int = 5):
    """Return the top_k opinions most relevant to the question (TF-IDF cosine similarity)."""
    vectorizer, tfidf_matrix = index if index else (None, None)
    if vectorizer is None or not opinions:
        return []
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    k = min(top_k, len(opinions))
    top_idx = np.argsort(scores)[::-1][:k]
    return [opinions[i] for i in top_idx if scores[i] > 0] or [opinions[i] for i in top_idx]


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
