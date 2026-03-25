"""
agents/study_agent.py
RAG Q&A agent with language support + quick actions
"""
from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

LANGUAGES = {
    "English": "en",
    "Nepali": "ne",
    "Spanish": "es",
    "Hindi": "hi",
    "French": "fr",
    "Japanese": "ja",
}

SYSTEM_PROMPT = """You are a precise academic study assistant.

Rules:
- Use ONLY the provided context [S#] and web sources [W#] to answer.
- Cite every fact inline with [S#] or [W#].
- If context is insufficient, say exactly what is missing — do NOT guess.
- Never invent definitions, dates, formulas, or facts.
- Adjust depth: short = key point only, medium = explanation + example, long = full structured breakdown.
- If a language is specified, respond entirely in that language.
"""


class StudyAgent:
    def __init__(self, llm):
        self.llm = llm

    def invoke(
        self,
        question: str,
        file_context: str,
        web_context: str = "",
        length_mode: str = "medium",
        language: str = "English",
        action: str = "default",
    ) -> str:
        length_instruction = {
            "short":  "Answer briefly — key point only, 2-4 sentences.",
            "medium": "Answer clearly with a short explanation and one example if helpful.",
            "long":   "Write a structured exam answer: definition → explanation → examples → conclusion.",
        }.get(length_mode, "Answer clearly.")

        action_instruction = {
            "simplify":   "Explain this as simply as possible, as if to a 10-year-old. No jargon.",
            "technical":  "Give the most technical, precise explanation possible with terminology.",
            "translate":  f"Translate the answer into {language}. Keep citations.",
            "deep_dive":  "Go deep — cover edge cases, nuances, and related concepts.",
            "default":    "",
        }.get(action, "")

        lang_instruction = ""
        if language != "English" and action != "translate":
            lang_instruction = f"\nRespond in {language}."

        web_block = f"\nWEB CONTEXT:\n{web_context}\n" if web_context else ""

        prompt = f"""FILE CONTEXT:
{file_context}
{web_block}
QUESTION: {question}

{length_instruction}
{action_instruction}
{lang_instruction}

CITATION RULE: Every important claim must end with [S#] or [W#].
If context is insufficient, state exactly what is missing."""

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content.strip()