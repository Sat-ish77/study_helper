"""
pages/2_🧪_Quiz_Lab.py — Study Helper v2
Keeps ALL original logic:
  - generate_quiz_from_context, grade_answer
  - render_quiz_question, render_quiz_results
  - session stats tracking
Adds:
  - Supabase pgvector instead of ChromaDB
  - 5 question types (adds short + medium answer)
  - Difficulty selector
  - Language selector (6 languages)
  - Supabase quiz history saved after every submission
  - Auth gate + production UI
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="Study Helper · Quiz Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector, get_llm
from agents.study_agent import LANGUAGES
from ingest import get_user_documents
from main import retrieve_docs, build_tagged_context
from quiz_db import save_score, get_user_history, get_user_stats

inject_css()
user_id = require_auth()

# ── Quiz generation (kept from original, extended for 5 types + difficulty) ──
def generate_quiz_from_context(
    llm,
    context: str,
    num_questions: int,
    quiz_types: List[str],
    difficulty: str,
    language: str,
) -> List[Dict]:
    type_instructions = {
        "mcq":        'Multiple choice — 4 options. correct = "A"/"B"/"C"/"D".',
        "true_false": 'True/False. correct = "True" or "False".',
        "fill_blank": 'Fill-in-the-blank. Use _____ in question. correct = answer word/phrase.',
        "short":      'Short answer. correct = 1-2 sentence ideal answer.',
        "medium":     'Medium answer. correct = 3-5 sentence ideal answer.',
    }

    selected_instructions = "\n".join(
        f"- {type_instructions[t]}" for t in quiz_types if t in type_instructions
    )

    prompt = f"""Based on the study material below, generate exactly {num_questions} quiz questions.
Difficulty: {difficulty}
Language: {language}
Use a mix of these types:\n{selected_instructions}

STUDY MATERIAL:
{context[:4000]}

Return ONLY a valid JSON array. Each object must have:
  "question": string
  "type": one of {quiz_types}
  "options": list (MCQ: 4 options; true_false: ["True","False"]; others: [])
  "correct": string
  "explanation": string
  "difficulty": "{difficulty}"

No markdown fences, no extra text. Pure JSON array only."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw  = resp.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        questions = json.loads(raw.strip())
        return questions if isinstance(questions, list) else []
    except Exception as e:
        st.error(f"Quiz generation error: {e}")
        return []


# ── AI grading (kept from original) ──────────────────────────────────────────
def grade_answer(llm, question: str, user_answer: str, correct_answer: str) -> Dict:
    if not user_answer.strip():
        return {"is_correct": False, "score": 0, "feedback": "No answer provided."}

    prompt = f"""Grade this answer:

Question: {question}
User's Answer: {user_answer}
Correct Answer: {correct_answer}

Consider exact matches, synonyms, and partial credit.
Respond with JSON only:
{{"is_correct": true/false, "score": 0-100, "feedback": "brief explanation"}}"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw  = resp.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
        return {
            "is_correct": is_correct,
            "score": 100 if is_correct else 0,
            "feedback": "Correct!" if is_correct else f"Correct answer: {correct_answer}",
        }


# ── Render single question (kept from original) ───────────────────────────────
def render_quiz_question(question: Dict, idx: int):
    q_type = question.get("type", "mcq")

    st.markdown(
        f'<div class="quiz-question">Q{idx+1}: {question["question"]}</div>',
        unsafe_allow_html=True
    )

    answer_key = f"answer_{idx}"

    if q_type in ("fill_blank", "short", "medium"):
        height = 80 if q_type == "medium" else 40
        user_answer = st.text_area(
            "Your answer:",
            key=answer_key,
            height=height,
            placeholder="Type your answer here..."
        )
    else:
        options = question.get("options", [])
        user_answer = st.radio(
            "Select your answer:",
            options,
            key=answer_key,
            index=None
        ) if options else ""

    st.session_state[f"user_answer_{idx}"] = user_answer
    st.markdown("---")


# ── Render results (kept from original, adds Supabase save) ──────────────────
def render_quiz_results(questions: List[Dict], llm, topic: str, source_file: str,
                         difficulty: str, types_used: List[str], language: str):
    correct_count = 0
    total         = len(questions)

    st.markdown(
        '<h2 class="sh-section"><span class="sh-section-accent">✦</span>Results</h2>',
        unsafe_allow_html=True
    )

    for idx, q in enumerate(questions):
        user_answer   = st.session_state.get(f"user_answer_{idx}", "") or ""
        correct_answer = q.get("correct", "")
        q_type        = q.get("type", "mcq")

        if q_type in ("fill_blank", "short", "medium"):
            grade_result = grade_answer(llm, q["question"], user_answer, correct_answer)
            is_correct   = grade_result.get("is_correct", False)
            feedback     = grade_result.get("feedback", "")
        else:
            user_letter  = user_answer[0].upper() if user_answer else ""
            is_correct   = (
                user_letter == correct_answer.upper()
                or user_answer == correct_answer
            )
            feedback = ""

        if is_correct:
            correct_count += 1

        status_icon  = "✅" if is_correct else "❌"
        status_color = "#4ade80" if is_correct else "#f87171"

        feedback_html = f'<br><span style="color:#9ca3af;">Feedback: {feedback}</span>' if feedback else ""
        st.markdown(
            f'<div style="padding:15px; background:rgba(17,19,24,0.8);'
            f' border-radius:10px; border-left:4px solid {status_color}; margin:10px 0;">'
            f'<strong>{status_icon} Q{idx+1}:</strong> {q["question"]}<br>'
            f'<span style="color:#6b7280;">Your answer: {user_answer or "Not answered"}</span><br>'
            f'<span style="color:#4ade80;">Correct answer: {correct_answer}</span>'
            f'{feedback_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        if q.get("explanation"):
            st.markdown(
                f'<div style="background:rgba(13,26,46,0.8); border-left:4px solid #1d4ed8;'
                f' padding:12px 15px; border-radius:0 8px 8px 0; margin:-5px 0 10px;">'
                f'💡 {q["explanation"]}</div>',
                unsafe_allow_html=True
            )

    # Score display
    percentage = (correct_count / total * 100) if total > 0 else 0
    score_color = "#4ade80" if percentage >= 70 else "#fbbf24" if percentage >= 50 else "#f87171"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1c22 0%,#111318 100%);'
        f' border:1px solid {score_color}; border-radius:12px; padding:20px 30px;'
        f' text-align:center; margin:20px 0;">'
        f'<div style="font-family:\'DM Serif Display\',serif; font-size:2.5rem;'
        f' color:{score_color};">{correct_count}/{total}</div>'
        f'<div style="font-size:1rem; color:#9ca3af;">{percentage:.0f}% · {difficulty.title()}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Update local session stats (kept from original)
    st.session_state.total_quizzes             = st.session_state.get("total_quizzes", 0) + 1
    st.session_state.total_correct             = st.session_state.get("total_correct", 0) + correct_count
    st.session_state.total_questions_answered  = st.session_state.get("total_questions_answered", 0) + total

    # Save to Supabase (new)
    save_score(
        user_id=user_id,
        topic=topic or "General",
        source_file=source_file,
        score=correct_count,
        total=total,
        difficulty=difficulty,
        types_used=types_used,
        language=language,
    )

    if percentage >= 80:
        st.balloons()
        st.success("🎉 Excellent! You really know this material!")
    elif percentage >= 60:
        st.info("👍 Good job! Keep studying to improve further.")
    else:
        st.warning("📚 Keep practicing! Review the explanations above.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Quiz", use_container_width=True, type="primary"):
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.rerun()
    with col2:
        if st.button("📚 Back to Study Helper", use_container_width=True):
            st.switch_page("pages/1_📚_Study_Helper.py")


# ── Session state ─────────────────────────────────────────────────────────────
if "quiz_questions" not in st.session_state: st.session_state.quiz_questions = None
if "quiz_submitted" not in st.session_state: st.session_state.quiz_submitted = False
if "quiz_topic"     not in st.session_state: st.session_state.quiz_topic     = ""
if "quiz_source"    not in st.session_state: st.session_state.quiz_source    = "All documents"
if "quiz_difficulty"not in st.session_state: st.session_state.quiz_difficulty= "medium"
if "quiz_types"     not in st.session_state: st.session_state.quiz_types     = ["mcq", "true_false"]
if "quiz_language"  not in st.session_state: st.session_state.quiz_language  = "English"

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_header(active_page="Quiz Lab")

with st.sidebar:
    st.markdown("---")
    model_label = render_model_selector() 

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.5rem;">Session stats</div>',
        unsafe_allow_html=True
    )

    # Supabase stats
    db_stats = get_user_stats(user_id)
    sc1, sc2 = st.columns(2)
    sc1.metric("Accuracy",  f"{db_stats['accuracy_pct']}%")
    sc2.metric("Quizzes",   db_stats["total_quizzes"])

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.5rem;">Recent quizzes</div>',
        unsafe_allow_html=True
    )
    history = get_user_history(user_id, limit=8)
    if history:
        for h in history:
            pct   = round(h["score"] / h["total"] * 100) if h["total"] else 0
            col   = "#4ade80" if pct >= 70 else "#fbbf24" if pct >= 50 else "#f87171"
            src   = (h.get("source_file") or "All")[:18]
            st.markdown(
                f'<div style="font-size:0.78rem; color:#9ca3af; padding:4px 0;'
                f' border-bottom:1px solid #1a1c22;">'
                f'{src} &nbsp;<span style="color:{col}; font-weight:500;">'
                f'{h["score"]}/{h["total"]}</span>'
                f'<span style="color:#4b5563;"> · {h.get("difficulty","")}</span></div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No quizzes yet.")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="font-family:\'DM Serif Display\',serif; color:#f0ede8;">🧪 Quiz Lab</h2>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#6b7280; margin-bottom:1.5rem;">'
    'Generate quizzes from your notes and test your knowledge.</p>',
    unsafe_allow_html=True
)

# ── Quiz setup form ───────────────────────────────────────────────────────────
if st.session_state.quiz_questions is None:
    docs           = get_user_documents(user_id)
    source_options = ["All documents"] + docs

    with st.form("quiz_config"):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Topic (optional)",
                placeholder="e.g. Cell division, TCP/IP — leave blank for all notes"
            )
            source = st.selectbox("Source document", source_options)
            lang   = st.selectbox("Language", list(LANGUAGES.keys()))

        with col2:
            difficulty = st.selectbox(
                "Difficulty",
                ["easy", "medium", "hard"],
                index=1,
                help="Easy = recall/definitions · Medium = application · Hard = analysis"
            )
            count = st.selectbox("Questions", [5, 10, 20, 50], index=1)

        st.markdown(
            '<div style="font-size:0.78rem; color:#6b7280; margin:0.75rem 0 0.4rem;">'
            'Question types</div>',
            unsafe_allow_html=True
        )
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        use_mcq    = tc1.checkbox("MCQ",          value=True)
        use_tf     = tc2.checkbox("True/False",   value=True)
        use_fill   = tc3.checkbox("Fill blank",   value=False)
        use_short  = tc4.checkbox("Short answer", value=False)
        use_medium = tc5.checkbox("Medium answer",value=False)

        submitted = st.form_submit_button(
            "🎯 Generate Quiz", type="primary", use_container_width=True
        )

    if submitted:
        types = []
        if use_mcq:    types.append("mcq")
        if use_tf:     types.append("true_false")
        if use_fill:   types.append("fill_blank")
        if use_short:  types.append("short")
        if use_medium: types.append("medium")

        if not types:
            st.warning("Please select at least one question type.")
        elif not docs and source != "All documents":
            st.warning("No documents uploaded yet. Visit Upload Docs first.")
        else:
            with st.spinner("🎯 Generating your quiz..."):
                llm = get_llm(model_label)

                # Get context
                query   = topic if topic.strip() else "study material key concepts"
                rr      = retrieve_docs(user_id, query)

                if rr.docs:
                    # Filter by source file if specific file selected
                    if source != "All documents":
                        filtered = [d for d in rr.docs if d.metadata.get("filename") == source]
                        rr.docs  = filtered if filtered else rr.docs
                    context, _ = build_tagged_context(rr.docs)
                else:
                    context = ""

                if not context:
                    st.error("No content found. Upload documents first on the Upload Docs page.")
                else:
                    questions = generate_quiz_from_context(
                        llm=llm,
                        context=context,
                        num_questions=count,
                        quiz_types=types,
                        difficulty=difficulty,
                        language=lang,
                    )
                    if questions:
                        st.session_state.quiz_questions  = questions
                        st.session_state.quiz_submitted  = False
                        st.session_state.quiz_topic      = topic or "General"
                        st.session_state.quiz_source     = source
                        st.session_state.quiz_difficulty = difficulty
                        st.session_state.quiz_types      = types
                        st.session_state.quiz_language   = lang
                        st.rerun()
                    else:
                        st.error("Could not generate quiz. Try a different topic.")

# ── Active quiz ───────────────────────────────────────────────────────────────
elif not st.session_state.quiz_submitted:
    questions  = st.session_state.quiz_questions
    difficulty = st.session_state.quiz_difficulty
    lang       = st.session_state.quiz_language

    st.markdown(
        f'<div style="background:#111318; border:0.5px solid #1e2028; border-radius:8px;'
        f' padding:12px 16px; margin-bottom:1rem; font-size:0.875rem; color:#9ca3af;">'
        f'📝 {len(questions)} questions &nbsp;·&nbsp; {difficulty.title()}'
        f' &nbsp;·&nbsp; {lang}</div>',
        unsafe_allow_html=True
    )

    for idx, q in enumerate(questions):
        render_quiz_question(q, idx)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Submit Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_submitted = True
            st.rerun()

    if st.button("❌ Cancel", use_container_width=True):
        st.session_state.quiz_questions = None
        st.session_state.quiz_submitted = False
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
else:
    llm = get_llm(model_label)
    render_quiz_results(
        questions   = st.session_state.quiz_questions,
        llm         = llm,
        topic       = st.session_state.quiz_topic,
        source_file = st.session_state.quiz_source,
        difficulty  = st.session_state.quiz_difficulty,
        types_used  = st.session_state.quiz_types,
        language    = st.session_state.quiz_language,
    )