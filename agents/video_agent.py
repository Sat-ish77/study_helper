"""
agents/video_agent.py
Topic → structured slide JSON for video generator.
"""
from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are an educational video script writer.
Given context about a topic, produce 3-6 slides.

Output valid JSON array only. No markdown fences. No extra text.
Each slide object:
  title         : str  (max 6 words)
  bullets       : list[str]  (3-4 points, max 12 words each)
  narrator_text : str  (1-3 sentences spoken while slide shows)

Language: {language}
Reading level: {level}
"""


class VideoAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_slides(
        self,
        topic: str,
        context: str,
        language: str = "English",
        level: str = "Standard",
    ) -> tuple[list[dict], str]:
        """
        Returns (slides, source_label)
        source_label: "rag" | "web" | "llm"
        """
        source = "rag" if context.strip() else "llm"

        system = SYSTEM_PROMPT.format(language=language, level=level)
        prompt = f"""CONTEXT:
{context if context.strip() else "(No specific context — use general knowledge)"}

TOPIC: {topic}

Generate slides for a {level.lower()} level explainer video in {language}."""

        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            slides = json.loads(raw.strip())
            return (slides if isinstance(slides, list) else []), source
        except Exception:
            return [], source