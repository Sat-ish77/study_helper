"""
backend/routers/dashboard.py
Dashboard stats endpoint.

GET /api/v1/dashboard/stats → counts for all dashboard widgets
GET /api/v1/dashboard/study-plan → daily study plan (cached per day)
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone, date
from dependencies import get_current_user, get_db
from model_manager import get_llm

router = APIRouter()


@router.get("/stats")
async def get_stats(user_id: str = Depends(get_current_user)):
    """
    Get all dashboard stats in one call.
    React displays these as metric cards.
    """
    sb = get_db()
    today = date.today().isoformat()

    # Documents uploaded — use * not id for count in supabase-py v2
    docs = sb.table("sh_document_chunks")\
        .select("*", count="exact")\
        .eq("user_id", user_id)\
        .execute()

    # Total flashcards
    cards = sb.table("sh_flashcards")\
        .select("*", count="exact")\
        .eq("user_id", user_id)\
        .execute()

    # Flashcards due today
    due = sb.table("sh_flashcards")\
        .select("*", count="exact")\
        .eq("user_id", user_id)\
        .lte("next_review", today)\
        .execute()

    # Quiz attempts — no accuracy column, compute from score/total
    quizzes = sb.table("sh_quiz_scores")\
        .select("score, total")\
        .eq("user_id", user_id)\
        .execute()

    quiz_data = quizzes.data or []
    avg_accuracy = 0
    if quiz_data:
        valid = [q for q in quiz_data if q.get("total") and q["total"] > 0]
        if valid:
            avg_accuracy = round(
                sum(q["score"] / q["total"] for q in valid) / len(valid) * 100
            )

    # Recent quiz scores — no accuracy column
    recent = sb.table("sh_quiz_scores")\
        .select("topic, score, total, created_at, course_name")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .limit(6)\
        .execute()

    recent_scores = recent.data or []
    for s in recent_scores:
        if s.get("total") and s["total"] > 0:
            s["percentage"] = round(s["score"] / s["total"] * 100)

    return {
        "documents": docs.count or 0,
        "flashcards_total": cards.count or 0,
        "flashcards_due": due.count or 0,
        "quiz_attempts": len(quiz_data),
        "quiz_avg_percentage": avg_accuracy,
        "recent_scores": recent_scores,
    }


@router.post("/study-plan")
async def get_study_plan(
    model: str = "Llama 3.3 70B",
    user_id: str = Depends(get_current_user)
):
    """
    Generate a personalized daily study plan.
    Uses: Canvas deadlines + weak quiz topics + due flashcards.
    Cached in sh_user_settings.daily_plan for today.
    One LLM call per day per user.
    """
    from langchain_core.messages import HumanMessage
    import json

    sb = get_db()
    today = date.today().isoformat()

    # Check cache
    settings = sb.table("sh_user_settings")\
        .select("daily_plan, plan_date")\
        .eq("user_id", user_id)\
        .execute()

    if settings.data:
        plan_date = settings.data[0].get("plan_date")
        if plan_date == today and settings.data[0].get("daily_plan"):
            return {"plan": settings.data[0]["daily_plan"], "cached": True}

    # Gather data for plan
    # Due flashcards count
    due = sb.table("sh_flashcards")\
        .select("*", count="exact")\
        .eq("user_id", user_id)\
        .lte("next_review", today)\
        .execute()

    # Weak topics — no accuracy column, compute from score/total
    quizzes = sb.table("sh_quiz_scores")\
        .select("topic, score, total")\
        .eq("user_id", user_id)\
        .execute()

    weak_topics = []
    if quizzes.data:
        topic_map = {}
        for q in quizzes.data:
            t = q["topic"]
            if t not in topic_map:
                topic_map[t] = []
            if q.get("total") and q["total"] > 0:
                topic_map[t].append(q["score"] / q["total"])
        weak_topics = [
            t for t, ratios in topic_map.items()
            if ratios and sum(ratios) / len(ratios) < 0.6
        ][:3]

    # Canvas events (next 7 days)
    canvas_events = []
    try:
        ical_settings = sb.table("sh_user_settings")\
            .select("ical_url")\
            .eq("user_id", user_id)\
            .execute()
        if ical_settings.data and ical_settings.data[0].get("ical_url"):
            # Check canvas cache
            cache = sb.table("sh_canvas_cache")\
                .select("data")\
                .like("cache_key", f"%{user_id}")\
                .execute()
            if cache.data:
                all_events = cache.data[0].get("data", [])
                canvas_events = [
                    e["title"] for e in all_events
                    if e.get("start") and e["start"][:10] >= today
                ][:5]
    except Exception:
        pass

    # Generate plan
    llm = get_llm(model)
    prompt = f"""Create a short, actionable daily study plan for today ({today}).

Student data:
- Flashcards due for review: {due.count or 0}
- Weak topics (< 60% quiz score): {weak_topics or 'none identified yet'}
- Upcoming Canvas deadlines: {canvas_events or 'none found'}

Create a realistic 2-3 hour study plan. Be specific and encouraging.
Format as JSON:
{{
  "greeting": "Good morning! Here's your study plan for today.",
  "sessions": [
    {{
      "time": "30 min",
      "activity": "Review flashcards",
      "reason": "You have N cards due today"
    }}
  ],
  "tip": "One motivational study tip"
}}"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip().replace("```json", "").replace("```", "").strip()
        plan = json.loads(content)
    except Exception:
        plan = {
            "greeting": f"Good luck studying today!",
            "sessions": [{"time": "30 min", "activity": "Review your notes", "reason": "Stay consistent"}],
            "tip": "Short study sessions beat long cramming sessions."
        }

    # Cache plan for today
    sb.table("sh_user_settings").upsert({
        "user_id": user_id,
        "daily_plan": plan,
        "plan_date": today,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }, on_conflict="user_id").execute()

    return {"plan": plan, "cached": False}