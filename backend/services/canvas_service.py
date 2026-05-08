"""
backend/services/canvas_service.py
Canvas calendar business logic ported from canvas_api.py.

Tables:
  sh_user_settings  — ical_url, user_id, updated_at
  sh_canvas_cache   — cache_key, data (jsonb), created_at, expires_at
  sh_dismissed_events — id, user_id, event_id, dismissed_at
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from dependencies import get_db


# ── Event Categorization ─────────────────────────────────────────────────────

CATEGORY_PATTERNS = {
    "exam": [
        r"\bexam\b", r"\bmidterm\b", r"\bfinal\b", r"\btest\b",
        r"\bquiz\b", r"\bassessment\b",
    ],
    "assignment": [
        r"\bassignment\b", r"\bhomework\b", r"\bhw\b", r"\bproject\b",
        r"\blab\b", r"\breport\b", r"\bpaper\b", r"\bessay\b",
    ],
    "lecture": [
        r"\blecture\b", r"\bclass\b", r"\bsession\b", r"\bseminar\b",
    ],
    "discussion": [
        r"\bdiscussion\b", r"\bforum\b", r"\bpost\b", r"\breply\b",
    ],
    "deadline": [
        r"\bdue\b", r"\bdeadline\b", r"\bsubmit\b", r"\bsubmission\b",
    ],
}


def categorize_event(event: Dict) -> str:
    """
    Categorize a calendar event based on its title and description.
    Ported from canvas_api.categorize_event().
    Returns one of: exam, assignment, lecture, discussion, deadline, other.
    """
    text = (
        (event.get("title", "") + " " + event.get("description", ""))
        .lower()
    )

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return category

    return "other"


# ── Dismiss / Undismiss Events ───────────────────────────────────────────────

def dismiss_event(user_id: str, event_id: str) -> bool:
    """
    Dismiss a calendar event for this user.
    Ported from canvas_api.dismiss_event().
    Table: sh_dismissed_events
    """
    try:
        sb = get_db()
        sb.table("sh_dismissed_events").upsert({
            "user_id": user_id,
            "event_id": event_id,
            "dismissed_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id,event_id").execute()
        return True
    except Exception as e:
        print(f"[canvas_service] Dismiss error: {e}")
        return False


def get_dismissed(user_id: str) -> List[str]:
    """
    Get list of dismissed event IDs for this user.
    Ported from canvas_api.get_dismissed().
    """
    try:
        sb = get_db()
        result = sb.table("sh_dismissed_events") \
            .select("event_id") \
            .eq("user_id", user_id) \
            .execute()
        return [r["event_id"] for r in (result.data or [])]
    except Exception as e:
        print(f"[canvas_service] Get dismissed error: {e}")
        return []


# ── Cache clearing ───────────────────────────────────────────────────────────

def clear_cache_for_user(user_id: str) -> bool:
    """
    Clear canvas cache entries for a user.
    Called when the user removes their iCal URL.
    """
    try:
        sb = get_db()
        sb.table("sh_canvas_cache") \
            .delete() \
            .like("cache_key", f"%{user_id}%") \
            .execute()
        return True
    except Exception as e:
        print(f"[canvas_service] Cache clear error: {e}")
        return False