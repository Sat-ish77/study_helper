"""
agents/image_agent.py
Decides DALL-E 3 vs matplotlib diagram + returns structured data.
"""
from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are an educational visualization assistant.
Given a topic or question, decide the best visualization type.

Output valid JSON only. No markdown fences. No extra text.

If it needs a real image (scene, object, illustration, concept art):
  {"type": "dalle", "prompt": "detailed DALL-E prompt here"}

If it needs a chart or data diagram:
  {"type": "chart", "chart_type": "bar|pie|line",
   "title": "chart title",
   "labels": ["A", "B", "C"],
   "values": [10, 20, 30],
   "xlabel": "x axis label",
   "ylabel": "y axis label"}

If it needs a flowchart or process diagram:
  {"type": "flowchart",
   "title": "flowchart title",
   "steps": ["Step 1", "Step 2", "Step 3"]}

Choose "dalle" for: animals, people, places, objects, concept illustrations.
Choose "chart" for: comparisons, statistics, distributions, trends.
Choose "flowchart" for: processes, algorithms, workflows, sequences.
"""


class ImageAgent:
    def __init__(self, llm):
        self.llm = llm

    def decide(self, concept: str) -> dict:
        """
        Returns structured dict telling image_generator what to render.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Concept: {concept}"),
        ]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            result = json.loads(raw.strip())
            return result if isinstance(result, dict) else {"type": "dalle", "prompt": concept}
        except Exception:
            return {"type": "dalle", "prompt": concept}