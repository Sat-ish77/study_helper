"""
Study Helper - Quiz Lab Page
Generate and take quizzes from your study notes
"""

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from main import DB_DIR, DEFAULT_MODEL, TEMPERATURE

load_dotenv()

# Page config
st.set_page_config(
    page_title="Quiz Lab - Study Helper",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------
# Custom CSS
# -----------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global dark theme */
        .stApp {
            background-color: #1a1a1a;
            color: #e8e8e8;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a1a 0%, #242424 100%);
            border-right: 1px solid #333;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f5a623 !important;
            font-weight: 600;
        }
        
        /* Quiz card */
        .quiz-card {
            background: linear-gradient(135deg, rgba(42, 42, 42, 0.8) 0%, rgba(32, 32, 32, 0.9) 100%);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(245, 166, 35, 0.2);
            margin: 15px 0;
        }
        
        .quiz-question {
            font-size: 1.2rem;
            color: #f5a623;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        /* Option buttons */
        .option-correct {
            background: rgba(46, 204, 113, 0.2) !important;
            border: 2px solid #2ecc71 !important;
        }
        
        .option-wrong {
            background: rgba(231, 76, 60, 0.2) !important;
            border: 2px solid #e74c3c !important;
        }
        
        /* Score display */
        .score-display {
            background: linear-gradient(135deg, #f5a623 0%, #d88e1a 100%);
            color: #1a1a1a;
            padding: 20px 40px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            margin: 20px 0;
        }
        
        /* Progress bar */
        .quiz-progress {
            background: rgba(42, 42, 42, 0.6);
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
        }
        
        /* Explanation box */
        .explanation-box {
            background: rgba(52, 152, 219, 0.1);
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-top: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# Cache resources
# -----------------------
@st.cache_resource
def init_vectordb():
    """Initialize Chroma vectordb with OpenAI embeddings"""
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def init_llm():
    """Initialize ChatOpenAI LLM"""
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3)  # Slightly higher temp for variety


@st.cache_data
def get_indexed_files(_vectordb) -> List[str]:
    """Extract unique filenames from vectordb metadata"""
    try:
        collection = _vectordb._collection
        all_metadata = collection.get()["metadatas"]
        filenames = set()
        for meta in all_metadata:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
        return sorted(list(filenames))
    except Exception:
        return []


# -----------------------
# Quiz Generation
# -----------------------
def generate_quiz_from_topic(llm, vectordb, topic: str, num_questions: int, quiz_type: str) -> List[Dict]:
    """Generate quiz questions from a specific topic"""
    
    # Retrieve relevant context
    try:
        docs = vectordb.similarity_search(topic, k=8)
        context = "\n\n".join([d.page_content for d in docs])
    except Exception:
        context = ""
    
    if not context:
        return []
    
    type_instructions = {
        "mcq": """Generate multiple choice questions with 4 options each. 
Format each question as JSON:
{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A", "explanation": "..."}""",
        
        "true_false": """Generate True/False questions.
Format each question as JSON:
{"question": "... (True or False)", "options": ["True", "False"], "correct": "True" or "False", "explanation": "..."}""",
        
        "fill_blank": """Generate fill-in-the-blank questions. Use _____ for the blank.
Format each question as JSON:
{"question": "The _____ is responsible for...", "options": [], "correct": "answer word", "explanation": "..."}"""
    }
    
    prompt = f"""Based on the following study material, generate {num_questions} quiz questions about "{topic}".

STUDY MATERIAL:
{context[:4000]}

INSTRUCTIONS:
{type_instructions.get(quiz_type, type_instructions["mcq"])}

Generate exactly {num_questions} questions. Return ONLY a valid JSON array of questions, no other text.
Example format:
[
  {{"question": "...", "options": [...], "correct": "...", "explanation": "..."}}
]
"""
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        response_text = resp.content.strip()
        
        # Try to extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        questions = json.loads(response_text)
        return questions if isinstance(questions, list) else []
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []


def generate_quiz_from_file(llm, vectordb, filename: str, num_questions: int, quiz_type: str) -> List[Dict]:
    """Generate quiz questions from a specific file"""
    
    # Get all chunks from the specific file
    try:
        collection = vectordb._collection
        results = collection.get(where={"filename": filename})
        if results and results.get("documents"):
            context = "\n\n".join(results["documents"][:10])
        else:
            return []
    except Exception:
        return []
    
    return generate_quiz_from_context(llm, context, num_questions, quiz_type)


def generate_quiz_from_context(llm, context: str, num_questions: int, quiz_type: str) -> List[Dict]:
    """Generate quiz from raw context"""
    
    type_instructions = {
        "mcq": """Generate multiple choice questions with 4 options each. 
Format each question as JSON:
{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A", "explanation": "..."}""",
        
        "true_false": """Generate True/False questions.
Format each question as JSON:
{"question": "... (True or False)", "options": ["True", "False"], "correct": "True" or "False", "explanation": "..."}""",
        
        "fill_blank": """Generate fill-in-the-blank questions. Use _____ for the blank.
Format each question as JSON:
{"question": "The _____ is responsible for...", "options": [], "correct": "answer word", "explanation": "..."}"""
    }
    
    prompt = f"""Based on the following study material, generate {num_questions} quiz questions.

STUDY MATERIAL:
{context[:4000]}

INSTRUCTIONS:
{type_instructions.get(quiz_type, type_instructions["mcq"])}

Generate exactly {num_questions} questions. Return ONLY a valid JSON array of questions, no other text.
"""
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        response_text = resp.content.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        questions = json.loads(response_text)
        return questions if isinstance(questions, list) else []
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []


def grade_answer(llm, question: str, user_answer: str, correct_answer: str, context: str = "") -> Dict:
    """Grade a user's answer (for fill-in-blank or short answer)"""
    
    prompt = f"""Grade this answer:

Question: {question}
User's Answer: {user_answer}
Correct Answer: {correct_answer}

Is the user's answer correct or acceptable? Consider:
- Exact matches
- Synonyms or equivalent terms
- Partial credit for partially correct answers

Respond with JSON:
{{"is_correct": true/false, "score": 0-100, "feedback": "..."}}
"""
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        response_text = resp.content.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        return json.loads(response_text)
    except Exception:
        # Fallback to simple matching
        is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
        return {
            "is_correct": is_correct,
            "score": 100 if is_correct else 0,
            "feedback": "Correct!" if is_correct else f"The correct answer was: {correct_answer}"
        }


# -----------------------
# UI Components
# -----------------------
def render_sidebar(vectordb):
    """Render sidebar"""
    with st.sidebar:
        st.markdown("# 🧪 Quiz Lab")
        st.markdown("*Test your knowledge*")
        st.markdown("---")
        
        st.markdown("### 📊 Your Stats")
        
        total_quizzes = st.session_state.get("total_quizzes", 0)
        total_correct = st.session_state.get("total_correct", 0)
        total_questions = st.session_state.get("total_questions_answered", 0)
        
        if total_questions > 0:
            accuracy = (total_correct / total_questions) * 100
            st.metric("Quizzes Taken", total_quizzes)
            st.metric("Questions Answered", total_questions)
            st.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.info("Take a quiz to see your stats!")
        
        st.markdown("---")
        
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.total_quizzes = 0
            st.session_state.total_correct = 0
            st.session_state.total_questions_answered = 0
            st.rerun()
        
        st.markdown("---")
        
        st.markdown(
            '<div style="text-align: center; color: #888; font-size: 0.85em;">'
            'Powered by LangChain + OpenAI'
            '</div>',
            unsafe_allow_html=True
        )


def render_quiz_setup(vectordb, indexed_files):
    """Render quiz configuration UI"""
    
    st.markdown("### 🎯 Generate a Quiz")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_type = st.radio(
            "Generate from:",
            ["📝 Specific Topic", "📄 Specific File", "📚 All Notes"],
            key="source_type"
        )
        
        if source_type == "📝 Specific Topic":
            topic = st.text_input("Enter topic:", placeholder="e.g., Apoptosis, Cell Cycle")
            st.session_state.quiz_source = ("topic", topic)
        elif source_type == "📄 Specific File":
            if indexed_files:
                selected_file = st.selectbox("Select file:", indexed_files)
                st.session_state.quiz_source = ("file", selected_file)
            else:
                st.warning("No files indexed. Run `python ingest.py` first.")
                st.session_state.quiz_source = None
        else:
            st.session_state.quiz_source = ("all", None)
    
    with col2:
        quiz_type = st.radio(
            "Question type:",
            ["Multiple Choice", "True/False", "Fill in the Blank"],
            key="quiz_type"
        )
        
        type_map = {
            "Multiple Choice": "mcq",
            "True/False": "true_false",
            "Fill in the Blank": "fill_blank"
        }
        st.session_state.quiz_type_key = type_map[quiz_type]
        
        num_questions = st.slider("Number of questions:", 3, 15, 5)
        st.session_state.num_questions = num_questions
    
    # Generate button
    if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
        return True
    return False


def render_quiz_question(question: Dict, idx: int, quiz_type: str):
    """Render a single quiz question"""
    
    st.markdown(f'<div class="quiz-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="quiz-question">Q{idx + 1}: {question["question"]}</div>', unsafe_allow_html=True)
    
    answer_key = f"answer_{idx}"
    
    if quiz_type == "fill_blank":
        # Fill in the blank - text input
        user_answer = st.text_input(
            "Your answer:",
            key=answer_key,
            placeholder="Type your answer..."
        )
        st.session_state[f"user_answer_{idx}"] = user_answer
    else:
        # MCQ or True/False - radio buttons
        options = question.get("options", [])
        if options:
            user_answer = st.radio(
                "Select your answer:",
                options,
                key=answer_key,
                index=None
            )
            st.session_state[f"user_answer_{idx}"] = user_answer
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_quiz_results(questions: List[Dict], quiz_type: str, llm):
    """Render quiz results after submission"""
    
    correct_count = 0
    total = len(questions)
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    for idx, q in enumerate(questions):
        user_answer = st.session_state.get(f"user_answer_{idx}", "")
        correct_answer = q.get("correct", "")
        
        # Determine if correct
        if quiz_type == "fill_blank":
            # Use AI grading for fill-in-blank
            grade_result = grade_answer(llm, q["question"], user_answer or "", correct_answer)
            is_correct = grade_result.get("is_correct", False)
        else:
            # Simple matching for MCQ/True-False
            if user_answer:
                # Extract just the letter for MCQ
                user_letter = user_answer[0] if user_answer else ""
                is_correct = user_letter.upper() == correct_answer.upper() or user_answer == correct_answer
            else:
                is_correct = False
        
        if is_correct:
            correct_count += 1
        
        # Display question result
        status_icon = "✅" if is_correct else "❌"
        status_color = "#2ecc71" if is_correct else "#e74c3c"
        
        with st.container():
            st.markdown(
                f'<div style="padding: 15px; background: rgba(42,42,42,0.6); '
                f'border-radius: 10px; border-left: 4px solid {status_color}; margin: 10px 0;">'
                f'<strong>{status_icon} Q{idx + 1}:</strong> {q["question"]}<br>'
                f'<span style="color: #888;">Your answer: {user_answer or "Not answered"}</span><br>'
                f'<span style="color: #2ecc71;">Correct answer: {correct_answer}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show explanation if available
            if q.get("explanation"):
                st.markdown(
                    f'<div class="explanation-box">💡 {q["explanation"]}</div>',
                    unsafe_allow_html=True
                )
    
    # Final score
    percentage = (correct_count / total) * 100 if total > 0 else 0
    
    st.markdown(
        f'<div class="score-display">'
        f'🎯 Score: {correct_count}/{total} ({percentage:.0f}%)'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Update stats
    st.session_state.total_quizzes = st.session_state.get("total_quizzes", 0) + 1
    st.session_state.total_correct = st.session_state.get("total_correct", 0) + correct_count
    st.session_state.total_questions_answered = st.session_state.get("total_questions_answered", 0) + total
    
    # Emoji feedback
    if percentage >= 80:
        st.balloons()
        st.success("🎉 Excellent work! You really know this material!")
    elif percentage >= 60:
        st.info("👍 Good job! Keep studying to improve further.")
    else:
        st.warning("📚 Keep practicing! Review the explanations above.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Another Quiz", use_container_width=True):
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.rerun()
    
    with col2:
        if st.button("📚 Back to Study Helper", use_container_width=True):
            st.switch_page("pages/1_📚_Study_Helper.py")


# -----------------------
# Main App
# -----------------------
def main():
    inject_custom_css()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Missing `OPENAI_API_KEY` in your .env file.")
        st.stop()
    
    # Initialize resources
    try:
        vectordb = init_vectordb()
        llm = init_llm()
    except Exception as e:
        st.error(f"⚠️ Error initializing: {str(e)}")
        st.stop()
    
    # Initialize session state
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = None
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    
    # Render sidebar
    render_sidebar(vectordb)
    
    # Header
    st.markdown(
        '<h1 style="color: #f5a623; margin-bottom: 10px;">🧪 Quiz Lab</h1>',
        unsafe_allow_html=True
    )
    st.markdown("*Generate quizzes from your study notes and test your knowledge*")
    st.markdown("---")
    
    # Get indexed files
    indexed_files = get_indexed_files(vectordb)
    
    # If no quiz is active, show setup
    if st.session_state.quiz_questions is None:
        should_generate = render_quiz_setup(vectordb, indexed_files)
        
        if should_generate:
            source = st.session_state.get("quiz_source")
            quiz_type = st.session_state.get("quiz_type_key", "mcq")
            num_q = st.session_state.get("num_questions", 5)
            
            if source is None:
                st.warning("Please select a source for the quiz.")
            else:
                with st.spinner("🎯 Generating your quiz..."):
                    source_type, source_value = source
                    
                    if source_type == "topic" and source_value:
                        questions = generate_quiz_from_topic(llm, vectordb, source_value, num_q, quiz_type)
                    elif source_type == "file" and source_value:
                        questions = generate_quiz_from_file(llm, vectordb, source_value, num_q, quiz_type)
                    elif source_type == "all":
                        # Get random chunks from all notes
                        try:
                            all_docs = vectordb.similarity_search("study material concepts", k=10)
                            context = "\n\n".join([d.page_content for d in all_docs])
                            questions = generate_quiz_from_context(llm, context, num_q, quiz_type)
                        except Exception:
                            questions = []
                    else:
                        questions = []
                    
                    if questions:
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    else:
                        st.error("Could not generate quiz. Try a different topic or check your notes.")
    
    # If quiz is active but not submitted
    elif not st.session_state.quiz_submitted:
        questions = st.session_state.quiz_questions
        quiz_type = st.session_state.get("quiz_type_key", "mcq")
        
        # Progress indicator
        st.markdown(
            f'<div class="quiz-progress">'
            f'📝 {len(questions)} Questions | Type: {quiz_type.upper().replace("_", " ")}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Render all questions
        for idx, q in enumerate(questions):
            render_quiz_question(q, idx, quiz_type)
        
        st.markdown("---")
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Submit Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()
        
        # Cancel button
        if st.button("❌ Cancel Quiz", use_container_width=True):
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.rerun()
    
    # If quiz is submitted, show results
    else:
        questions = st.session_state.quiz_questions
        quiz_type = st.session_state.get("quiz_type_key", "mcq")
        render_quiz_results(questions, quiz_type, llm)


if __name__ == "__main__":
    main()

