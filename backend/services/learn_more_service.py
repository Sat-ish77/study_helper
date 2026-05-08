"""
backend/services/learn_more_service.py
Learn More business logic ported from learn_more.py.

Table: sh_learn_more_cache
Columns: topic, query, videos (jsonb), images (jsonb), cached_at, expires_at
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dependencies import get_db


# ── Ambiguity detection ──────────────────────────────────────────────────────

AMBIGUOUS_TERMS = {
    "claude": [
        "Claude (AI assistant)",
        "Claude (French mathematician)",
        "Claude (musical artist)",
        "Claude (company)"
    ],
    "python": [
        "Python (programming language)",
        "Python (snake)",
        "Monty Python (comedy group)"
    ],
    "java": [
        "Java (programming language)",
        "Java (island in Indonesia)",
        "Java (coffee)"
    ],
    "ruby": [
        "Ruby (programming language)",
        "Ruby (gemstone)",
    ],
    "apple": [
        "Apple (company)",
        "Apple (fruit)",
        "Apple (records)"
    ],
    "aws": [
        "AWS (Amazon Web Services)",
        "AWS (Advanced Wireless Services)"
    ],
    "api": [
        "API (Application Programming Interface)",
        "API (other meanings)"
    ]
}


def detect_ambiguity(topic: str, answer: str = "") -> Optional[str]:
    """
    Detect if a topic might be ambiguous based on known terms or answer patterns.
    Returns the ambiguous term or None.
    Ported from learn_more.detect_ambiguity().
    """
    if answer:
        ambiguity_phrases = [
            "can refer to", "various contexts", "multiple meanings",
            "disambiguation", "depending on context", "could mean several"
        ]
        if any(phrase in answer.lower() for phrase in ambiguity_phrases):
            return topic.lower()

    topic_lower = topic.lower()
    for term in AMBIGUOUS_TERMS:
        if term == topic_lower or topic_lower.startswith(term + " "):
            return term

    words = topic.split()
    if len(words) == 1 and len(topic) <= 6:
        return topic.lower()

    return None


def get_ambiguity_options(term: str) -> List[str]:
    """Get disambiguation options for an ambiguous term."""
    return AMBIGUOUS_TERMS.get(term.lower(), [
        f"{term.title()} (general topic)",
        f"{term.title()} (company)",
        f"{term.title()} (other)"
    ])


# ── YouTube search ───────────────────────────────────────────────────────────

def search_youtube(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search YouTube Data API for educational videos.
    Ported from learn_more.search_youtube().
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return []

    try:
        import requests

        params = {
            "part": "snippet",
            "q": query,
            "key": api_key,
            "maxResults": 3,
            "type": "video"
        }

        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        video_ids = [item["id"]["videoId"] for item in data.get("items", [])]
        if not video_ids:
            return []

        details_params = {
            "part": "contentDetails,snippet",
            "id": ",".join(video_ids),
            "key": api_key
        }
        details_response = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params=details_params, timeout=10
        )
        details_response.raise_for_status()
        details_data = details_response.json()

        results = []
        for item in details_data.get("items", [])[:max_results]:
            snippet = item["snippet"]
            duration = item["contentDetails"]["duration"]

            duration_clean = duration.replace("PT", "").replace("H", ":").replace("M", ":").replace("S", "")
            if duration_clean.count(":") == 1:
                duration_clean = f"0:{duration_clean}"

            results.append({
                "title": snippet["title"],
                "video_id": item["id"],
                "duration": duration_clean,
                "channel_name": snippet["channelTitle"],
                "thumbnail": snippet["thumbnails"]["medium"]["url"],
                "url": f"https://youtube.com/watch?v={item['id']}"
            })

        return results

    except Exception as e:
        print(f"[learn_more_service] YouTube error: {e}")
        return []


# ── Wikimedia search ─────────────────────────────────────────────────────────

def search_wikimedia(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search Wikimedia Commons for relevant images.
    Ported from learn_more.search_wikimedia().
    """
    try:
        import requests

        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": max_results,
            "prop": "imageinfo",
            "iiprop": "url|size|thumburl",
            "iiurlwidth": 200
        }

        headers = {
            "User-Agent": "StudyHelper/2.0 (educational app; contact@studyhelper.app)"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        pages = data.get('query', {}).get('pages', {})

        for page_id, page in pages.items():
            if 'imageinfo' in page:
                info = page['imageinfo'][0]
                results.append({
                    'title': page['title'].replace('File:', ''),
                    'url': info.get('url', ''),
                    'thumbnail': info.get('thumburl', ''),
                    'description': page.get('extract', '')[:200]
                })

        return results[:max_results]

    except Exception as e:
        print(f"[learn_more_service] Wikimedia error: {e}")
        return []


# ── Main entry point (with caching) ─────────────────────────────────────────

def get_learn_more(topic: str) -> Dict:
    """
    Get cached Learn More content or fetch fresh data.
    Ported from learn_more.get_learn_more().
    Table: sh_learn_more_cache
    """
    sb = get_db()

    # Check cache
    try:
        cached = sb.table("sh_learn_more_cache") \
            .select("*") \
            .eq("topic", topic) \
            .gte("expires_at", datetime.now().isoformat()) \
            .execute()

        if cached.data:
            return {
                "videos": cached.data[0]["videos"],
                "images": cached.data[0]["images"]
            }
    except Exception as e:
        print(f"[learn_more_service] Cache check error: {e}")

    # Fetch fresh
    youtube_results = search_youtube(topic)
    image_results = search_wikimedia(topic)

    # Cache for 30 days
    try:
        sb.table("sh_learn_more_cache").upsert({
            "topic": topic,
            "query": topic,
            "videos": youtube_results,
            "images": image_results,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }).execute()
    except Exception as e:
        print(f"[learn_more_service] Cache save error: {e}")

    return {
        "videos": youtube_results,
        "images": image_results
    }
