import uuid
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from db import (
    init_db, create_session, get_session, add_vote, get_poll_results,
    add_opinion, get_opinions, get_top_words, bold_top_words
)
from qr_utils import generate_qr_bytes
from rag_engine import build_index, retrieve_relevant, generate_grounded_answer, generate_question

st.set_page_config(page_title="QR Live Poll & Opinion Wall", layout="centered")
init_db()

params = st.query_params
role = params.get("role", "host")
session_id = params.get("session", None)


# ---------------------------------------------------------
# VOTER VIEW  (opened when someone scans the QR code)
# ---------------------------------------------------------
def voter_view(session_id):
    session = get_session(session_id)
    if not session:
        st.error("This session doesn't exist or has expired.")
        return

    st.title("📢 " + session["question"])

    if session["mode"] == "poll":
        choice = st.radio("Pick one option:", session["options"])
        if st.button("Submit Vote", use_container_width=True):
            add_vote(session_id, choice)
            st.success("✅ Your vote has been recorded. Thank you!")
            st.balloons()

    elif session["mode"] == "opinion":
        answer = st.text_area("Type your answer:")
        if st.button("Submit Answer", use_container_width=True):
            if answer.strip():
                add_opinion(session_id, answer.strip())
                st.success("✅ Your answer has been recorded. Thank you!")
                st.balloons()
            else:
                st.warning("Please type something before submitting.")


# ---------------------------------------------------------
# HOST VIEW  (create a session + live dashboard)
# ---------------------------------------------------------
def host_view():
    st.title("🎤 QR Live Poll & Opinion Wall — Host")

    base_url = st.sidebar.text_input(
        "App Base URL (for QR code)",
        value="http://localhost:8501",
        help="When deployed (e.g. Streamlit Cloud), paste that public URL here."
    )

    if "active_session" not in st.session_state:
        st.session_state.active_session = None

    if st.session_state.active_session is None:
        st.markdown("### 🎯 Set up your session")
        mode = st.radio("Choose activity type:", ["Poll (MCQ)", "Opinion Wall (open text)"])
        st.write("")

        # ---- AI-assisted question generation ----
        with st.expander("✨ Generate question with AI (optional)"):
            topic = st.text_input("Topic", placeholder="e.g. social media habits")
            if st.button("Generate"):
                mode_key = "poll" if mode == "Poll (MCQ)" else "opinion"
                with st.spinner("Generating with AI..."):
                    result = generate_question(topic.strip() or "social media", mode_key)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["gen_question"] = result["question"]
                    st.session_state["gen_options"] = ", ".join(result.get("options", []))
                    st.rerun()

        if mode == "Poll (MCQ)":
            question = st.text_input(
                "Poll Question",
                value=st.session_state.get("gen_question", ""),
                placeholder="e.g. Which social media platform do you use the most?"
            )
            options_raw = st.text_input(
                "Answer Options (separate with commas)",
                value=st.session_state.get("gen_options", ""),
                placeholder="e.g. WhatsApp, Instagram, Facebook, Snapchat"
            )
            st.caption("Add 2–6 options for best results.")
            st.write("")
            if st.button("🚀 Start Poll Session", use_container_width=True):
                if not question.strip() or not options_raw.strip():
                    st.warning("Please enter both a question and at least two options.")
                else:
                    options = [o.strip() for o in options_raw.split(",") if o.strip()]
                    sid = str(uuid.uuid4())[:8]
                    create_session(sid, "poll", question.strip(), options)
                    st.session_state.active_session = sid
                    st.session_state.pop("gen_question", None)
                    st.session_state.pop("gen_options", None)
                    st.rerun()

        else:
            question = st.text_input(
                "Opinion Question",
                value=st.session_state.get("gen_question", ""),
                placeholder="e.g. What is your opinion about social media?"
            )
            st.caption("Audience members will type a free-text response.")
            st.write("")
            if st.button("🚀 Start Opinion Session", use_container_width=True):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    sid = str(uuid.uuid4())[:8]
                    create_session(sid, "opinion", question.strip())
                    st.session_state.active_session = sid
                    st.session_state.pop("gen_question", None)
                    st.rerun()

    else:
        sid = st.session_state.active_session
        session = get_session(sid)
        vote_url = f"{base_url}/?role=voter&session={sid}"

        st.subheader(session["question"])
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(generate_qr_bytes(vote_url), caption="Scan to respond", width=220)
            st.code(vote_url, language=None)
            if st.button("🔴 End Session"):
                st.session_state.active_session = None
                st.rerun()

        with col2:
            st_autorefresh(interval=3000, key="live_refresh")  # auto-refresh every 3s

            if session["mode"] == "poll":
                counts, percentages, total = get_poll_results(sid, session["options"])
                st.metric("Total votes", total)
                for opt in session["options"]:
                    st.write(f"**{opt}** — {percentages[opt]}% ({counts[opt]} votes)")
                    st.progress(percentages[opt] / 100)

            elif session["mode"] == "opinion":
                opinions = get_opinions(sid)
                st.metric("Total responses", len(opinions))
                if opinions:
                    top_words = get_top_words(opinions, top_n=8)
                    st.caption("🔥 Most repeated words are shown in **bold**")
                    for ans in opinions:
                        st.markdown("— " + bold_top_words(ans, top_words))

                    st.divider()
                    st.subheader("🤖 Ask AI about the responses (RAG)")
                    ai_question = st.text_input(
                        "Ask a question about what the audience said",
                        placeholder="e.g. What do people think about Instagram?"
                    )
                    if st.button("Get AI Answer") and ai_question.strip():
                        with st.spinner("Retrieving relevant responses and generating answer..."):
                            index, _ = build_index(opinions)
                            relevant = retrieve_relevant(ai_question, opinions, index, top_k=5)
                            answer = generate_grounded_answer(ai_question, relevant)
                        st.success(answer)
                        with st.expander("Responses used to generate this answer"):
                            for r in relevant:
                                st.write("• " + r)
                else:
                    st.info("Waiting for responses...")


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
if role == "voter" and session_id:
    voter_view(session_id)
else:
    host_view()
