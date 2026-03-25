"""
agents/quiz_agent.py
Quiz generation + AI semantic grading. 5 question types.
"""
from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage

GENERATION_PROMPT = """You are an academic quiz creator. Generate questions ONLY from the provided context.

Output valid JSON array only. No markdown fences. No extra text.
Each object must have:
  question    : str
  type        : "mcq" | "truefalse" | "fillblank" | "short" | "medium"
  options     : list[str] (MCQ only, 4 options labeled A/B/C/D)
  correct     : str  (for mcq: "A"/"B"/"C"/"D"; for truefalse: "True"/"False";
                      for fillblank/short/medium: ideal answer string)
  explanation : str  (why this is correct, cite source if possible)
  difficulty  : "easy" | "medium" | "hard"

For fillblank: use _____ as the blank in the question string.
For short: correct = 1-2 sentence ideal answer.
For medium: correct = 3-5 sentence ideal answer.
Language: {language}
"""

GRADING_PROMPT = """You are a fair academic grader.

Question: {question}
Ideal answer: {correct}
Student answer: {student}

Grade on a scale 0-2:
  2 = Correct or semantically equivalent
  1 = Partially correct, missing key details
  0 = Incorrect or missing

Output JSON only: {{"score": 0|1|2, "feedback": "brief explanation"}}
No markdown, no extra text."""


class QuizAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(
        self,
        context: str,
        count: int = 10,
        difficulty: str = "medium",
        types: list[str] | None = None,
        language: str = "English",
    ) -> list[dict]:
        types = types or ["mcq", "truefalse", "fillblank", "short"]
        types_str = ", ".join(types)

        prompt = f"""CONTEXT:
{context}

Generate exactly {count} questions.
Difficulty: {difficulty}
Question types to use (mix them): {types_str}
Language: {language}

Return JSON array only."""

        system = GENERATION_PROMPT.format(language=language)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            questions = json.loads(raw)
            return questions if isinstance(questions, list) else []
        except json.JSONDecodeError:
            return []

    def grade_open(self, question: str, correct: str, student: str) -> dict:
        """Grade short/medium/fillblank answers semantically."""
        if not student.strip():
            return {"score": 0, "feedback": "No answer provided."}

        prompt = GRADING_PROMPT.format(
            question=question,
            correct=correct,
            student=student,
        )
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw.strip())
        except Exception:
            return {"score": 0, "feedback": "Grading error — please try again."}