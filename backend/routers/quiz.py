"""
backend/routers/quiz.py
Quiz endpoints.

POST /api/v1/quiz/generate   → generate quiz from user's docs
POST /api/v1/quiz/save       → save quiz score to sh_quiz_scores
GET  /api/v1/quiz/history    → get past scores
GET  /api/v1/quiz/weak-topics → topics where avg score < 60%

Quiz generation uses RAG — retrieves relevant chunks from 
the user's documents and asks the LLM to generate questions
from that content. This ensures questions are about the
student's actual study material, not generic knowledge.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from dependencies import get_current_user, get_db
from model_manager import get_llm
from services.rag_service import retrieve_docs, build_file_context

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateQuizRequest(BaseModel):
    topic: str
    num_questions: int = 5
    difficulty: str = "medium"       # easy / medium / hard
    question_types: list = ["mcq"]   # mcq, true_false, short_answer, fill_blank
    language: str = "English"
    model: str = "Llama 3.3 70B"
    course_name: Optional[str] = None


class SaveScoreRequest(BaseModel):
    topic: str
    score: int           # raw count of correct answers
    total: int           # total questions
    # accuracy removed — no such column in sh_quiz_scores
    # compute from score/total when needed
    course_name: Optional[str] = None


# ── Generate Quiz ─────────────────────────────────────────────────────────────

@router.post("/generate")
async def generate_quiz(
    body: GenerateQuizRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Generate a quiz from the user's uploaded study materials.
    
    1. Retrieves relevant chunks for the topic
    2. Builds context from those chunks
    3. Asks LLM to generate quiz questions
    4. Returns structured JSON with questions + answers
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Retrieve relevant docs for the topic
    retrieval = retrieve_docs(user_id, body.topic)
    docs = retrieval["docs"]

    if not docs:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this topic. Please upload study materials first."
        )

    file_ctx, _, _ = build_file_context(docs[:5])  # top 5 chunks
    llm = get_llm(body.model)

    types_str = ", ".join(body.question_types)
    lang_note = f"Generate questions in {body.language}." if body.language != "English" else ""

    prompt = f"""Generate {body.num_questions} quiz questions about: {body.topic}

Study material:
{file_ctx}

Requirements:
- Question types: {types_str}
- Difficulty: {body.difficulty}
- {lang_note}
- NEVER generate questions about chapter numbers, page references, or document structure
- ONLY test understanding of actual concepts
- Each explanation must be unique and specific to that question
- Keep ALL technical terms in English even if translating

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "Question text here",
      "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
      "correct_answer": "A",
      "explanation": "Unique explanation specific to this question"
    }}
  ]
}}

For true_false type, options should be ["True", "False"].
For short_answer type, options should be [].
For fill_blank type, question should have ___ for blank."""

    try:
        resp = llm.invoke([
            SystemMessage(content="You are a quiz generator. Return only valid JSON, no markdown."),
            HumanMessage(content=prompt)
        ])

        import json
        content = resp.content.strip()
        # Strip markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        quiz_data = json.loads(content)
        return quiz_data

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation returned invalid JSON: {str(e)}"
        )
    except Exception as e:
        error = str(e)
        if "403" in error or "401" in error:
            raise HTTPException(
                status_code=503,
                detail="Model API key invalid. Try switching to Groq or OpenAI."
            )
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {error}")


# ── Save Score ────────────────────────────────────────────────────────────────

@router.post("/save")
async def save_score(
    body: SaveScoreRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Save a quiz score to sh_quiz_scores.
    score = raw correct count (e.g. 4)
    total = total questions (e.g. 5)
    No accuracy column in DB — computed from score/total when needed.
    """
    sb = get_db()
    row = {
        "user_id": user_id,
        "topic": body.topic,
        "score": body.score,
        "total": body.total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if body.course_name:
        row["course_name"] = body.course_name

    result = sb.table("sh_quiz_scores").insert(row).execute()
    return {
        "saved": True,
        "id": result.data[0]["id"] if result.data else None,
        "percentage": round(body.score / body.total * 100) if body.total > 0 else 0
    }


# ── Quiz History ──────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(
    limit: int = 20,
    user_id: str = Depends(get_current_user)
):
    """Get recent quiz scores for this user."""
    sb = get_db()
    result = sb.table("sh_quiz_scores")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()

    scores = result.data or []

    # Compute percentage for each score
    for s in scores:
        if s.get("total") and s.get("total") > 0:
            s["percentage"] = round(s["score"] / s["total"] * 100)
        else:
            s["percentage"] = 0

    return {"scores": scores, "count": len(scores)}


# ── Weak Topics ───────────────────────────────────────────────────────────────

@router.get("/weak-topics")
async def get_weak_topics(
    threshold: float = 0.6,
    user_id: str = Depends(get_current_user)
):
    """
    Find topics where average score/total < threshold (default 60%).
    No accuracy column — computed from score and total.
    """
    sb = get_db()
    result = sb.table("sh_quiz_scores")\
        .select("topic, score, total, course_name")\
        .eq("user_id", user_id)\
        .execute()

    scores = result.data or []

    # Group by topic and compute average from score/total
    topic_scores = {}
    for s in scores:
        topic = s["topic"]
        if topic not in topic_scores:
            topic_scores[topic] = {
                "ratios": [],
                "course_name": s.get("course_name")
            }
        if s.get("total") and s["total"] > 0:
            topic_scores[topic]["ratios"].append(s["score"] / s["total"])

    weak = []
    for topic, data in topic_scores.items():
        if not data["ratios"]:
            continue
        avg = sum(data["ratios"]) / len(data["ratios"])
        if avg < threshold:
            weak.append({
                "topic": topic,
                "avg_accuracy": round(avg, 3),
                "avg_percentage": round(avg * 100),
                "attempts": len(data["ratios"]),
                "course_name": data["course_name"],
            })

    weak.sort(key=lambda x: x["avg_accuracy"])
    return {"weak_topics": weak, "threshold": threshold}