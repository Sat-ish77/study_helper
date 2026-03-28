"""
canvas_api.py — Study Helper v2
Canvas LMS iCal integration with caching.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests

from supabase_client import get_supabase


def _serialize_events(events: List[Dict]) -> list:
    """Convert datetime objects to ISO strings for JSON storage."""
    serializable = []
    for e in events:
        ev = dict(e)
        if 'start' in ev and isinstance(ev['start'], datetime):
            ev['start'] = ev['start'].isoformat()
        if 'end' in ev and isinstance(ev['end'], datetime):
            ev['end'] = ev['end'].isoformat()
        serializable.append(ev)
    return serializable


def _deserialize_events(data) -> List[Dict]:
    """Convert ISO strings back to datetime objects."""
    events = json.loads(data) if isinstance(data, str) else data
    for e in events:
        if 'start' in e and isinstance(e['start'], str):
            try:
                e['start'] = datetime.fromisoformat(e['start'])
            except Exception:
                pass
        if 'end' in e and isinstance(e['end'], str):
            try:
                e['end'] = datetime.fromisoformat(e['end'])
            except Exception:
                pass
    return events


def parse_ical_content(ical_text: str) -> List[Dict]:
    """Parse iCal content and extract events."""
    events = []
    current_event = {}

    for line in ical_text.split('\n'):
        line = line.strip()

        if line.startswith('BEGIN:VEVENT'):
            current_event = {}
        elif line.startswith('END:VEVENT'):
            if current_event.get('title'):
                events.append(current_event)
            current_event = {}
        elif line.startswith('SUMMARY:'):
            current_event['title'] = line[8:].strip()
        elif line.startswith('DTSTART'):
            try:
                date_str = line.split(':', 1)[-1].strip()
                if 'T' in date_str:
                    date_part, time_part = date_str.split('T')
                    time_part = time_part.rstrip('Z')
                    dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    dt = dt.replace(tzinfo=timezone.utc)
                current_event['start'] = dt
            except Exception as e:
                print(f"[canvas_api] Error parsing start date: {e}")
        elif line.startswith('DTEND'):
            try:
                date_str = line.split(':', 1)[-1].strip()
                if 'T' in date_str:
                    date_part, time_part = date_str.split('T')
                    time_part = time_part.rstrip('Z')
                    dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    dt = dt.replace(tzinfo=timezone.utc)
                current_event['end'] = dt
            except Exception as e:
                print(f"[canvas_api] Error parsing end date: {e}")
        elif line.startswith('DESCRIPTION:'):
            current_event['description'] = line[12:].strip()
        elif line.startswith('UID:'):
            current_event['event_uid'] = line[4:].strip()

    return events


def fetch_canvas_events(canvas_url: str, user_id: str) -> List[Dict]:
    """Fetch events from Canvas iCal feed with 30-min Supabase cache."""
    supabase = get_supabase()
    cache_key = f"canvas_{abs(hash(canvas_url))}_{user_id}"

    # Check cache
    try:
        cached = supabase.table("sh_canvas_cache") \
            .select("data") \
            .eq("cache_key", cache_key) \
            .gte("expires_at", datetime.now(timezone.utc).isoformat()) \
            .execute()
        if cached.data:
            print(f"[canvas_api] Cache hit")
            return _deserialize_events(cached.data[0]["data"])
    except Exception as e:
        print(f"[canvas_api] Cache check error: {e}")

    # Fetch fresh
    try:
        headers = {"User-Agent": "StudyHelper/2.0 (educational app; contact@studyhelper.app)"}
        response = requests.get(canvas_url, headers=headers, timeout=10)
        response.raise_for_status()
        events = parse_ical_content(response.text)
        print(f"[canvas_api] Fetched {len(events)} events")

        # Save to cache — data column is jsonb, pass list directly
        try:
            supabase.table("sh_canvas_cache").upsert({
                "cache_key": cache_key,
                "data": _serialize_events(events),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
            }).execute()
        except Exception as e:
            print(f"[canvas_api] Cache save error: {e}")

        return events

    except Exception as e:
        print(f"[canvas_api] Fetch error: {e}")
        return []


def get_upcoming_events(events: List[Dict], days_ahead: int = 14) -> List[Dict]:
    """Filter to upcoming events within N days."""
    # Add type guard to filter out non-dict events
    events = [e for e in events if isinstance(e, dict)]
    
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    upcoming = []
    for event in events:
        start = event.get('start')
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if now <= start <= cutoff:
                upcoming.append(event)
    return sorted(upcoming, key=lambda x: x['start'])


def categorize_event(event: Dict) -> str:
    """Categorize event by title/description keywords."""
    text = f"{event.get('title','')} {event.get('description','')}".lower()
    if any(w in text for w in ['exam', 'midterm', 'final']):
        return 'exam'
    if 'quiz' in text:
        return 'quiz'
    if any(w in text for w in ['assignment', 'homework', 'project', 'submit']):
        return 'assignment'
    if any(w in text for w in ['lecture', 'class', 'meeting']):
        return 'lecture'
    if any(w in text for w in ['deadline', 'due', 'turn in']):
        return 'deadline'
    return 'other'


def save_ical_url(user_id: str, url: str):
    """Persist iCal URL to Supabase."""
    try:
        sb = get_supabase()
        sb.table("sh_user_settings").upsert({
            "user_id": user_id,
            "ical_url": url,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }, on_conflict="user_id").execute()

        # Verify the row was saved
        verify = sb.table("sh_user_settings") \
            .select("*").eq("user_id", user_id).execute()
        print(f"[canvas_api] Save verify: {verify.data}")
    except Exception as e:
        print(f"[canvas_api] Save URL error: {e}")


def load_ical_url(user_id: str) -> Optional[str]:
    """Load persisted iCal URL from Supabase."""
    try:
        result = get_supabase().table("sh_user_settings") \
            .select("ical_url").eq("user_id", user_id).execute()
        if result.data:
            return result.data[0].get("ical_url")
    except Exception as e:
        print(f"[canvas_api] Load URL error: {e}")
    return None


def clear_ical_url(user_id: str):
    """Clear the stored iCal URL and cached events for a user."""
    try:
        sb = get_supabase()
        sb.table("sh_user_settings") \
            .update({"ical_url": None}) \
            .eq("user_id", user_id) \
            .execute()
        # sh_canvas_cache uses cache_key (not user_id column).
        # cache_key format: "canvas_{hash}_{user_id}"
        sb.table("sh_canvas_cache") \
            .delete() \
            .like("cache_key", f"%_{user_id}") \
            .execute()
    except Exception as e:
        print(f"[canvas_api] Clear URL error: {e}")


def dismiss_event(user_id: str, event_id: str):
    """Mark event as dismissed."""
    try:
        get_supabase().table("sh_dismissed_events").upsert({
            "user_id": user_id,
            "event_id": event_id,
            "dismissed_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        print(f"[canvas_api] Dismiss error: {e}")


def get_dismissed(user_id: str) -> set:
    """Get set of dismissed event IDs for this user."""
    try:
        result = get_supabase().table("sh_dismissed_events") \
            .select("event_id").eq("user_id", user_id).execute()
        return {r["event_id"] for r in result.data}
    except Exception as e:
        print(f"[canvas_api] Get dismissed error: {e}")
        return set()


def render_daily_digest(user_id: str):
    """Render Canvas digest widget in sidebar."""
    import streamlit as st

    # Load from Supabase if not in session state
    if 'canvas_url' not in st.session_state:
        saved = load_ical_url(user_id)
        if saved:
            st.session_state.canvas_url = saved

    canvas_url = st.session_state.get('canvas_url', '')

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">📅 Canvas</div>',
        unsafe_allow_html=True
    )

    if not canvas_url:
        with st.expander("Connect Calendar", expanded=False):
            st.caption("Canvas → Calendar → Calendar Feed → Export")
            url_input = st.text_input(
                "iCal URL", placeholder="https://canvas.unt.edu/feeds/calendars/...",
                key="canvas_url_input", label_visibility="collapsed"
            )
            if st.button("Connect", type="primary", width='stretch', key="save_canvas"):
                if url_input.strip():
                    st.session_state.canvas_url = url_input.strip()
                    save_ical_url(user_id, url_input.strip())
                    st.rerun()
                else:
                    st.error("Enter a valid URL")
        return

    events = fetch_canvas_events(canvas_url, user_id)
    upcoming = get_upcoming_events(events, days_ahead=14)
    dismissed = get_dismissed(user_id)
    upcoming = [e for e in upcoming if e.get('event_uid') not in dismissed]

    if not upcoming:
        st.caption("No upcoming events")
        if st.button("Clear", key="clear_canvas", width='stretch'):
            del st.session_state['canvas_url']
            st.rerun()
        return

    icons = {'assignment': '📝', 'exam': '📋', 'quiz': '🧪',
             'lecture': '👥', 'deadline': '⏰', 'other': '📌'}
    now = datetime.now(timezone.utc)

    for event in upcoming[:4]:
        category = categorize_event(event)
        start = event.get('start')
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            days_until = (start.date() - now.date()).days
            if days_until <= 1:
                date_str = "Today" if days_until == 0 else "Tomorrow"
                color = "#e85454"
            elif days_until <= 3:
                date_str = f"In {days_until} days"
                color = "#f59e0b"
            else:
                date_str = f"In {days_until} days"
                color = "#4caf88"
        else:
            date_str, color = "No date", "#6b7280"

        st.markdown(
            f'<div style="background:rgba(232,164,74,0.08);padding:8px 10px;'
            f'border-radius:6px;margin:4px 0;border-left:2px solid {color};font-size:0.82rem;">'
            f'{icons.get(category,"📌")} <b>{event.get("title","Untitled")}</b><br>'
            f'<span style="color:{color};">{date_str}</span></div>',
            unsafe_allow_html=True
        )

    if len(upcoming) > 4:
        st.caption(f"+{len(upcoming) - 4} more")

    if st.button("Clear Calendar", key="clear_canvas", width='stretch'):
        del st.session_state['canvas_url']
        clear_ical_url(user_id)  # Also clear from Supabase
        st.rerun()