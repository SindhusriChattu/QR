# QR Live Poll & Opinion Wall

A Streamlit app where a host creates a live poll or an open-ended opinion
question, generates a QR code, and audience members scan it on their phones
to respond in real time. The host dashboard auto-refreshes with live
percentages (poll) or a response wall with the most-repeated words in bold
(opinion).

## How it works

1. Host picks **Poll** or **Opinion Wall**, enters the question (+ options
   for poll), clicks Start.
2. App generates a unique `session_id` and a QR code encoding a URL like
   `http://<base_url>/?role=voter&session=<id>`.
3. Audience scans the QR → lands directly on the voting/answer page (same
   app, routed via `st.query_params`).
4. Responses are written to a local SQLite database (WAL mode enabled so
   many phones can submit at once without locking issues).
5. Host dashboard polls the DB every 3 seconds (`streamlit-autorefresh`)
   and shows live results.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit gives you (usually `http://localhost:8501`) — that's
the host view. Leave the sidebar "Base URL" as `http://localhost:8501` only
if testers are on the same machine. If testers are on other phones on the
same WiFi, replace it with your machine's local IP, e.g.
`http://192.168.1.5:8501`, and run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Deploying (Streamlit Cloud)

1. Push this folder to a GitHub repo.
2. Deploy on Streamlit Cloud as usual.
3. Once deployed, copy the public app URL and paste it into the sidebar
   "Base URL" field so the QR code points to the real public link.

## Tech stack (accurate, for resume use)

| Layer | Tool | Category |
|---|---|---|
| UI / app | Streamlit | Web app framework |
| QR generation | `qrcode` | Utility library |
| Storage | SQLite (WAL mode) | Database |
| Live refresh | `streamlit-autorefresh` | Polling mechanism |
| Word highlighting | `collections.Counter` + regex | Rule-based NLP (not ML/DL) |
| **RAG Q&A (Opinion Wall only)** | `sentence-transformers` (embeddings) + `faiss-cpu` (vector search) + Groq LLM | **Genuine GenAI / RAG** |

**Important — be precise in interviews:** RAG is used only for the Opinion
Wall's "Ask AI about responses" feature, where there's an actual text
corpus to retrieve from. The Poll (MCQ) feature is plain counting/charting
— don't describe that part as AI. This distinction itself is a good
interview answer: it shows you understand *when* RAG is appropriate, not
just that you can wire one up.

### How the RAG feature works
1. Every collected opinion is embedded using `all-MiniLM-L6-v2`
   (sentence-transformers) and indexed in FAISS (cosine similarity via
   normalized inner product) — same pattern as your Legal Clause Risk
   Analyzer RAG version.
2. Host types a question (e.g. "What do people think about Instagram?").
3. The question is embedded, top-5 most similar opinions are retrieved.
4. Only those retrieved opinions are passed to Groq's LLM
   (`llama-3.1-8b-instant`) with an instruction to answer strictly from
   them — this is what makes it RAG rather than a plain LLM call.
5. Answer + the exact responses used are shown, so the host (and anyone
   reviewing your project) can verify it's grounded, not hallucinated.

### Setup for the RAG feature
You need a free Groq API key from https://console.groq.com

```bash
export GROQ_API_KEY="your_key_here"      # Mac/Linux
setx GROQ_API_KEY "your_key_here"        # Windows (restart terminal after)
```

If deploying on Streamlit Cloud, add `GROQ_API_KEY` under
**App settings → Secrets** instead.
