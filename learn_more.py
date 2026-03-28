"""
learn_more.py — Study Helper v2
YouTube + Wikimedia integration with Supabase caching.
Fetches educational videos and images for any topic.
"""
from __future__ import annotations

import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote
import streamlit as st

from supabase_client import get_supabase


# Known ambiguous terms that need clarification
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
        "Ruby (programming language)"
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
    """
    # Check if answer mentions ambiguity
    if answer:
        ambiguity_phrases = [
            "can refer to",
            "various contexts",
            "multiple meanings",
            "disambiguation",
            "depending on context",
            "could mean several"
        ]
        if any(phrase in answer.lower() for phrase in ambiguity_phrases):
            return topic.lower()
    
    # Check known ambiguous terms (case-insensitive)
    topic_lower = topic.lower()
    for term in AMBIGUOUS_TERMS:
        if term == topic_lower or topic_lower.startswith(term + " "):
            return term
    
    # Check if it's a single short word (likely ambiguous)
    words = topic.split()
    if len(words) == 1 and len(topic) <= 6:
        return topic.lower()
    
    return None


def render_ambiguity_widget(ambiguous_term: str, user_id: str) -> Optional[str]:
    """
    Render a clarification widget for ambiguous terms.
    Returns the clarified search term or None if not selected.
    """
    st.markdown("### 🤔 Need Clarification")
    st.markdown(f"**{ambiguous_term.title()}** can mean several things. What are you interested in?")
    
    options = AMBIGUOUS_TERMS.get(ambiguous_term.lower(), [])
    if not options:
        # Generic options for unknown ambiguous terms
        options = [
            f"{ambiguous_term.title()} (general topic)",
            f"{ambiguous_term.title()} (company)",
            f"{ambiguous_term.title()} (other)"
        ]
    
    selected = st.radio("Choose the meaning:", options)
    
    if st.button("🔍 Search with clarification", type="primary"):
        # Extract the core term from the selection
        clarified = selected.split("(")[0].strip()
        if ambiguous_term.lower() == "claude":
            if "AI assistant" in selected:
                return "Claude AI assistant"
            elif "mathematician" in selected:
                return "Claude Shannon mathematician"
            elif "musical" in selected:
                return "Claude musical artist"
            elif "company" in selected:
                return f"{ambiguous_term} company"
        elif ambiguous_term.lower() == "python":
            if "programming" in selected:
                return "Python programming language"
            elif "snake" in selected:
                return "Python snake species"
            elif "comedy" in selected:
                return "Monty Python comedy"
        elif ambiguous_term.lower() == "java":
            if "programming" in selected:
                return "Java programming language"
            elif "island" in selected:
                return "Java island Indonesia"
            elif "coffee" in selected:
                return "Java coffee"
        elif ambiguous_term.lower() == "ruby":
            if "programming" in selected:
                return "Ruby programming language"
            elif "gemstone" in selected:
                return "Ruby gemstone"
        elif ambiguous_term.lower() == "apple":
            if "company" in selected:
                return "Apple company"
            elif "fruit" in selected:
                return "Apple fruit"
            elif "records" in selected:
                return "Apple records"
        elif ambiguous_term.lower() == "aws":
            if "Amazon" in selected or "Web Services" in selected:
                return "Amazon Web Services AWS"
            elif "Wireless" in selected:
                return "Advanced Wireless Services AWS"
        elif ambiguous_term.lower() == "api":
            if "Programming" in selected or "Application" in selected:
                return "Application Programming Interface API"
        elif "fruit" in selected.lower() or "snake" in selected.lower() or "bird" in selected.lower():
            return selected
        elif "just the general topic" in selected.lower():
            return ambiguous_term
        else:
            # Use the full selection as search term
            return selected
    
    return None


def search_youtube(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search YouTube Data API for educational videos.
    Returns list of video dicts with title, video_id, duration, channel_name, thumbnail.
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
        
        response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        video_ids = [item["id"]["videoId"] for item in data["items"]]
        
        # Get video details (duration, channel)
        details_params = {
            "part": "contentDetails,snippet",
            "id": ",".join(video_ids),
            "key": api_key
        }
        details_response = requests.get("https://www.googleapis.com/youtube/v3/videos", params=details_params)
        details_response.raise_for_status()
        
        details_data = details_response.json()
        
        results = []
        for item in details_data["items"][:max_results]:
            snippet = item["snippet"]
            duration = item["contentDetails"]["duration"]
            
            # Parse ISO 8601 duration (PT4M13S -> 4:13)
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
        print(f"[YouTube fetch error]: {e}")
        return []


def search_wikimedia(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search Wikimedia Commons for relevant images.
    Returns list of image dicts with title, url, thumbnail, description.
    """
    try:
        import requests
        
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",  # File namespace
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
        print(f"[Wikimedia] Error: {e}")
        return []


def get_learn_more(topic: str, user_id: str) -> Dict:
    """
    Get cached Learn More content or fetch fresh data.
    Returns dict with youtube_videos and images lists.
    """
    supabase = get_supabase()
    
    # Check cache first
    try:
        cached = supabase.table("sh_learn_more_cache") \
            .select("*") \
            .eq("topic", topic) \
            .gte("expires_at", datetime.now().isoformat()) \
            .execute()
        
        if cached.data:
            print(f"[learn_more] Cache hit for: {topic}")
            return {
                "videos": cached.data[0]["videos"],
                "images": cached.data[0]["images"]
            }
    except Exception as e:
        print(f"[learn_more] Cache check error: {e}")
    
    # Fetch fresh data
    print(f"[learn_more] Cache miss, fetching: {topic}")
    youtube_results = search_youtube(topic)
    image_results = search_wikimedia(topic)
    
    # Cache for 30 days
    try:
        supabase.table("sh_learn_more_cache").upsert({
            "topic": topic,
            "query": topic,
            "videos": youtube_results,
            "images": image_results,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }).execute()
    except Exception as e:
        print(f"[learn_more] Cache save error: {e}")
    
    return {
        "videos": youtube_results,
        "images": image_results
    }


def render_learn_more_section(topic: str, user_id: str, answer: str = ""):
    """
    Render the Learn More section in Streamlit.
    Shows YouTube videos and Wikimedia images.
    Includes ambiguity detection and clarification.
    """
    # Check for ambiguity first
    ambiguous_term = detect_ambiguity(topic, answer)
    
    if ambiguous_term:
        # Show clarification widget
        clarification_key = f"clarified_{ambiguous_term}_{user_id}"
        
        if clarification_key not in st.session_state:
            clarified_topic = render_ambiguity_widget(ambiguous_term, user_id)
            
            if clarified_topic:
                st.session_state[clarification_key] = clarified_topic
                st.rerun()
            else:
                return
        
        # Use clarified topic
        search_topic = st.session_state[clarification_key]
    else:
        search_topic = topic
    
    with st.spinner("Finding learning resources..."):
        content = get_learn_more(search_topic, user_id)
    
    if not content["videos"] and not content["images"]:
        return
    
    with st.expander("📚 Learn More", expanded=False):
        cols = st.columns(2)
        
        # Render videos
        if content.get("videos"):
            st.markdown("#### 📹 Related Videos")
            for video in content["videos"][:3]:
                st.markdown(
                    f'<div style="background:rgba(33,150,243,0.08); padding:10px;'
                    f' border-radius:8px; margin:4px 0;">'
                    f'<a href="https://www.youtube.com/watch?v={video["video_id"]}" '
                    f'target="_blank" style="text-decoration:none; color:inherit;">'
                    f'<img src="{video["thumbnail"]}" alt="{video["title"]}" '
                    f'style="width:100px; height:75px; float:left; margin-right:10px; border-radius:4px;">'
                    f'<div style="font-size:0.9rem; font-weight:500;">{video["title"]}</div>'
                    f'<div style="font-size:0.8rem; color:#666;">Click to watch on YouTube</div>'
                    f'</a></div>',
                    unsafe_allow_html=True
                )
        
        # Render images
        if content.get("images"):
            st.markdown("#### 🖼️ Related Images")
            cols = st.columns(min(3, len(content["images"])))
            for i, image in enumerate(content["images"][:3]):
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'<a href="{image["url"]}" target="_blank">'
                        f'<img src="{image["thumbnail"]}" alt="{image["title"]}" '
                        f'style="width:100%; border-radius:8px;">'
                        f'</a><br>'
                        f'<small>{image["title"][:30]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        st.caption("Content cached for 30 days • Sources: YouTube, Wikimedia Commons")
        
        # Show what we searched for if it was clarified
        if ambiguous_term and clarification_key in st.session_state:
            st.caption(f"🔍 Searched for: {search_topic}")
