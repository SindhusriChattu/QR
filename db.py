import sqlite3
import re
from collections import Counter
from contextlib import contextmanager

DB_PATH = "qr_poll_app.db"

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "but", "to", "of", "in", "on",
    "for", "with", "it", "this", "that", "i", "we", "you", "are", "was",
    "be", "been", "very", "so", "my", "our", "your", "its", "as", "at",
    "have", "has", "had", "not", "no", "do", "does", "did", "am", "im",
    "me", "us", "they", "he", "she", "his", "her", "their"
}


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent reads/writes
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                mode TEXT,           -- 'poll' or 'opinion'
                question TEXT,
                options TEXT,        -- comma-separated, only for poll
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                option_selected TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS opinions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                answer_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_session(session_id, mode, question, options=None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, mode, question, options) VALUES (?, ?, ?, ?)",
            (session_id, mode, question, ",".join(options) if options else None)
        )


def get_session(session_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT session_id, mode, question, options FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "mode": row[1],
            "question": row[2],
            "options": row[3].split(",") if row[3] else []
        }


def add_vote(session_id, option):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO votes (session_id, option_selected) VALUES (?, ?)",
            (session_id, option)
        )


def get_poll_results(session_id, options):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT option_selected, COUNT(*) FROM votes WHERE session_id = ? GROUP BY option_selected",
            (session_id,)
        ).fetchall()
    counts = {opt: 0 for opt in options}
    for opt, c in rows:
        counts[opt] = c
    total = sum(counts.values())
    percentages = {
        opt: round((c / total) * 100, 1) if total > 0 else 0.0
        for opt, c in counts.items()
    }
    return counts, percentages, total


def add_opinion(session_id, text):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO opinions (session_id, answer_text) VALUES (?, ?)",
            (session_id, text)
        )


def get_opinions(session_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT answer_text FROM opinions WHERE session_id = ? ORDER BY id DESC",
            (session_id,)
        ).fetchall()
    return [r[0] for r in rows]


def get_top_words(texts, top_n=8):
    """Basic NLP: tokenize + stopword removal + frequency count."""
    all_words = []
    for t in texts:
        words = re.findall(r"[a-zA-Z']+", t.lower())
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        all_words.extend(words)
    counts = Counter(all_words)
    return set(w for w, _ in counts.most_common(top_n))


def bold_top_words(text, top_words):
    """Wrap frequently-used words in markdown bold for display."""
    def replacer(match):
        word = match.group(0)
        if word.lower() in top_words:
            return f"**{word}**"
        return word
    return re.sub(r"[a-zA-Z']+", replacer, text)
