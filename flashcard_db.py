"""
flashcard_db.py — Study Helper v2
Flashcard database with SM-2 spaced repetition algorithm.
Handles card creation, review scheduling, and progress tracking.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import random

from supabase_client import get_supabase


# SM-2 Algorithm parameters
INITIAL_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3
EASE_FACTOR_BONUS = 0.1

# Quality ratings (0-5)
QUALITY_RATINGS = {
    0: "Blackout - Complete failure to recall",
    1: "Bad - Incorrect response, but remembered with effort",
    2: "Poor - Incorrect response, but seemed easy to recall",
    3: "Good - Correct response with hesitation",
    4: "Easy - Correct response with little effort",
    5: "Perfect - Immediate, confident response"
}


def calculate_sm2_next_review(
    ease_factor: float,
    repetition_interval: int,
    repetitions: int,
    quality: int
) -> Tuple[float, int, int]:
    """
    Calculate next review parameters using SM-2 algorithm.
    Returns: (new_ease_factor, new_interval, new_repetitions)
    """
    # Update ease factor
    new_ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    
    # Ensure ease factor doesn't go below minimum
    if new_ease_factor < MIN_EASE_FACTOR:
        new_ease_factor = MIN_EASE_FACTOR
    
    # Calculate new interval and repetitions
    if quality < 3:
        # Reset for poor performance
        new_interval = 1
        new_repetitions = 0
    else:
        # Increase interval for good performance
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = int(repetition_interval * new_ease_factor)
        
        new_repetitions = repetitions + 1
    
    return new_ease_factor, new_interval, new_repetitions


def create_flashcard(
    user_id: str,
    question: str,
    answer: str,
    source: str = "manual",
    source_file: str = ""
) -> Optional[str]:
    """
    Create a new flashcard.
    Returns card ID if successful, None otherwise.
    """
    try:
        supabase = get_supabase()
        
        result = supabase.table("sh_flashcards").insert({
            "user_id": user_id,
            "question": question.strip(),
            "answer": answer.strip(),
            "source_file": source_file.strip(),
            "ease_factor": INITIAL_EASE_FACTOR,
            "interval_days": 1,
            "repetitions": 0,
            "next_review": datetime.now(timezone.utc).date().isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "front": question.strip(),  # Keep for compatibility
            "back": answer.strip()      # Keep for compatibility
        }).execute()
        
        if result.data:
            return result.data[0]["id"]
        return None
        
    except Exception as e:
        print(f"[flashcard_db] Create error: {e}")
        return None


def get_due_flashcards(user_id: str, limit: int = 20) -> List[Dict]:
    """
    Get flashcards due for review.
    """
    try:
        supabase = get_supabase()
        
        today = datetime.now(timezone.utc).date().isoformat()
        
        result = supabase.table("sh_flashcards") \
            .select("*") \
            .eq("user_id", user_id) \
            .lte("next_review", today) \
            .order("next_review", desc=False) \
            .limit(limit) \
            .execute()
        
        return result.data or []
        
    except Exception as e:
        print(f"[flashcard_db] Get due cards error: {e}")
        return []


def update_flashcard_review(
    card_id: str,
    quality: int,
    review_time_ms: int = 0
) -> bool:
    """
    Update flashcard after review with SM-2 algorithm.
    Returns True if successful.
    """
    try:
        supabase = get_supabase()
        
        # Get current card data
        result = supabase.table("sh_flashcards") \
            .select("ease_factor, interval_days, repetitions") \
            .eq("id", card_id) \
            .execute()
        
        if not result.data:
            return False
        
        card = result.data[0]
        
        # Calculate next review using SM-2
        new_ease, new_interval, new_reps = calculate_sm2_next_review(
            float(card["ease_factor"]),
            int(card["interval_days"]),
            int(card["repetitions"]),
            quality
        )
        
        # Calculate next review date
        next_review = datetime.now(timezone.utc).date() + timedelta(days=new_interval)
        
        # Update card
        supabase.table("sh_flashcards") \
            .update({
                "ease_factor": new_ease,
                "interval_days": new_interval,
                "repetitions": new_reps,
                "next_review": next_review.isoformat(),
                "last_review": datetime.now(timezone.utc).date().isoformat()
            }) \
            .eq("id", card_id) \
            .execute()
        
        return True
        
    except Exception as e:
        print(f"[flashcard_db] Update review error: {e}")
        return False


def get_all_flashcards(user_id: str) -> List[Dict]:
    """
    Get all flashcards for a user.
    """
    try:
        supabase = get_supabase()
        
        result = supabase.table("sh_flashcards") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        
        return result.data or []
        
    except Exception as e:
        print(f"[flashcard_db] Get all cards error: {e}")
        return []


def delete_flashcard(card_id: str, user_id: str) -> bool:
    """
    Delete a flashcard.
    """
    try:
        supabase = get_supabase()
        
        supabase.table("sh_flashcards") \
            .delete() \
            .eq("id", card_id) \
            .eq("user_id", user_id) \
            .execute()
        
        return True
        
    except Exception as e:
        print(f"[flashcard_db] Delete error: {e}")
        return False


def get_flashcard_stats(user_id: str) -> Dict:
    """
    Get flashcard statistics for dashboard.
    """
    try:
        supabase = get_supabase()
        
        # Total cards
        total_result = supabase.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .execute()
        total_cards = total_result.count or 0
        
        # Due cards
        now = datetime.now(timezone.utc).isoformat()
        due_result = supabase.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .lte("next_review", now) \
            .execute()
        due_cards = due_result.count or 0
        
        # Learned cards (repetitions >= 3)
        learned_result = supabase.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .gte("repetitions", 3) \
            .execute()
        learned_cards = learned_result.count or 0
        
        # New cards (repetitions == 0)
        new_result = supabase.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .eq("repetitions", 0) \
            .execute()
        new_cards = new_result.count or 0
        
        return {
            "total": total_cards,
            "due": due_cards,
            "learned": learned_cards,
            "new": new_cards,
            "retention": (learned_cards / total_cards * 100) if total_cards > 0 else 0
        }
        
    except Exception as e:
        print(f"[flashcard_db] Stats error: {e}")
        return {
            "total": 0,
            "due": 0,
            "learned": 0,
            "new": 0,
            "retention": 0
        }


def generate_flashcards_from_content(
    user_id: str,
    content: str,
    num_cards: int = 5,
    llm=None
) -> List[Dict]:
    """
    Generate flashcards from study content using LLM.
    Returns list of {question, answer, source_file} dicts.
    """
    if not llm:
        return []
    
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_prompt = """You are creating flashcards for studying.
Extract the most important concepts, definitions, and relationships from the given content.

Create exactly {num_cards} flashcard pairs in this JSON format:
[
  {{"question": "Question or term", "answer": "Answer or definition", "source_file": "Brief context"}},
  ...
]

Guidelines:
- Question should be a clear question or key term
- Answer should be concise but complete answer
- Source_file should hint at where this fits in the material
- Focus on fundamental concepts, not trivial details
- Each card should test one specific idea
- Make them challenging but fair

Return ONLY valid JSON array, no markdown fences."""

        prompt = f"""CONTENT:
{content[:2000]}

Create {num_cards} flashcards from this content."""

        messages = [
            SystemMessage(content=system_prompt.format(num_cards=num_cards)),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        raw = response.content.strip()
        
        # Clean up response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        
        cards = json.loads(raw)
        
        # Validate and clean cards
        valid_cards = []
        for card in cards[:num_cards]:
            if isinstance(card, dict) and "question" in card and "answer" in card:
                valid_cards.append({
                    "question": str(card["question"]).strip(),
                    "answer": str(card["answer"]).strip(),
                    "source_file": str(card.get("source_file", ""))[:100].strip()
                })
        
        return valid_cards
        
    except Exception as e:
        if "403" in str(e) or "permission" in str(e).lower():
            print(f"[flashcard_db] Model error: Invalid API key. Try switching to Groq or Gemini.")
        else:
            print(f"[flashcard_db] Generate error: {e}")
        return []


def save_flashcard_from_qa(
    user_id: str,
    question: str,
    answer: str,
    source_file: str = ""
) -> Optional[str]:
    """
    Save a flashcard from a Q&A pair.
    Returns card ID if successful.
    """
    # Clean and format but don't truncate
    question = question.strip()
    answer = answer.strip()
    
    # Remove preamble lines from flashcard answers
    preamble_lines = [
        "📄 I couldn't find this in your documents.",
        "🌐 Here's what I found from the web:",
        "📄 Couldn't find enough in your documents.",
        "📄 I can't find this in your uploaded files.",
        "🌐 Here's what I found from the web:"
    ]
    
    # Split answer into lines and filter out preamble
    answer_lines = answer.split('\n')
    clean_lines = []
    skip_next = False
    
    for line in answer_lines:
        # Check if this line is a preamble
        is_preamble = any(preamble in line for preamble in preamble_lines)
        
        if is_preamble:
            skip_next = True  # Skip the next line (usually empty)
        elif skip_next:
            skip_next = False  # Skip this empty line and reset
        else:
            clean_lines.append(line)
    
    # Rejoin clean lines
    clean_answer = '\n'.join(clean_lines).strip()
    
    return create_flashcard(
        user_id=user_id,
        question=question,
        answer=clean_answer,
        source="qa",
        source_file=source_file
    )
