import type { AskRequest, AskResponse, ModelOption, Document, Conversation } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
  token?: string
): Promise<T> {
  const headers: Record<string, string> = {
    ...(token && { Authorization: `Bearer ${token}` }),
  };

  const isFormData = options.body instanceof FormData;
  if (!isFormData) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: { ...headers, ...(options.headers as Record<string, string>) },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${body}`);
  }

  return res.json();
}

// ── Health ───────────────────────────────────────────

export async function checkHealth() {
  return request("/health");
}


// ── RAG ──────────────────────────────────────────────

export async function askQuestion(
  body: AskRequest,
  token: string
): Promise<AskResponse> {
  return request("/api/v1/rag/ask", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function askStream(
  body: AskRequest,
  token: string
): Promise<Response> {
  const res = await fetch(`${API_URL}/api/v1/rag/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`Stream ${res.status}: ${text}`);
  }
  return res;
}

export async function getDocuments(token: string) {
  return request("/api/v1/rag/documents", {}, token);
}

export async function deleteDocument(sourceFile: string, token: string) {
  return request(`/api/v1/rag/documents/${encodeURIComponent(sourceFile)}`, {
    method: "DELETE",
  }, token);
}

export async function ingestFile(file: File, token: string): Promise<{ filename: string; chunks_created: number; status: string }> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/v1/rag/ingest", { method: "POST", body: form }, token);
}

// ── Speech ───────────────────────────────────────────

export async function transcribeAudio(audioBlob: Blob, token: string) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  return request("/api/v1/speech/transcribe", { method: "POST", body: form }, token);
}

export async function synthesizeSpeech(
  body: { text: string; language: string; engine?: string },
  token: string
) {
  return request("/api/v1/speech/synthesize", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

// ── Quiz ─────────────────────────────────────────────

export async function generateQuiz(body: any, token: string) {
  return request("/api/v1/quiz/generate", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function saveQuizScore(
  body: { topic: string; score: number; total: number; course_name?: string },
  token: string
) {
  return request("/api/v1/quiz/save", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function getQuizHistory(token: string) {
  return request("/api/v1/quiz/history", {}, token);
}

export async function getWeakTopics(token: string) {
  return request("/api/v1/quiz/weak-topics", {}, token);
}

// ── Flashcards ───────────────────────────────────────

export async function getFlashcards(dueOnly: boolean, token: string) {
  const params = dueOnly ? "?due_only=true" : "";
  return request(`/api/v1/flashcards${params}`, {}, token);
}

export async function createFlashcard(
  body: { question: string; answer: string; source_file?: string },
  token: string
) {
  return request("/api/v1/flashcards", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function reviewFlashcard(
  id: string,
  quality: number,
  token: string
) {
  return request(`/api/v1/flashcards/${id}`, {
    method: "PUT",
    body: JSON.stringify({ quality }),
  }, token);
}

export async function deleteFlashcard(id: string, token: string) {
  return request(`/api/v1/flashcards/${id}`, { method: "DELETE" }, token);
}

export async function generateFlashcardsFromDoc(
  body: { source_file: string; num_cards: number; model: string },
  token: string
) {
  return request("/api/v1/flashcards/generate", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function bulkSaveFlashcards(cards: any[], token: string) {
  return request("/api/v1/flashcards/bulk", {
    method: "POST",
    body: JSON.stringify({ cards }),
  }, token);
}

// ── Canvas ───────────────────────────────────────────

export async function saveCanvasUrl(url: string, token: string) {
  return request("/api/v1/canvas/url", {
    method: "POST",
    body: JSON.stringify({ url }),
  }, token);
}

export async function getCanvasUrl(token: string) {
  return request("/api/v1/canvas/url", {}, token);
}

export async function clearCanvasUrl(token: string) {
  return request("/api/v1/canvas/url", { method: "DELETE" }, token);
}

export async function getCanvasEvents(token: string) {
  return request("/api/v1/canvas/events", {}, token);
}

// ── Conversations ────────────────────────────────────

export async function getConversations(token: string): Promise<{ conversations: Conversation[] }> {
  return request("/api/v1/conversations", {}, token);
}

export async function createConversation(
  body: { title: string; messages: any[] },
  token: string
) {
  return request("/api/v1/conversations", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function getConversation(id: string, token: string) {
  return request(`/api/v1/conversations/${id}`, {}, token);
}

export async function updateConversation(
  id: string,
  messages: any[],
  token: string
) {
  return request(`/api/v1/conversations/${id}`, {
    method: "PUT",
    body: JSON.stringify({ messages }),
  }, token);
}

export async function deleteConversation(id: string, token: string) {
  return request(`/api/v1/conversations/${id}`, { method: "DELETE" }, token);
}

// ── Dashboard ────────────────────────────────────────

export async function getDashboardStats(token: string) {
  return request("/api/v1/dashboard/stats", {}, token);
}

export async function getStudyPlan(token: string) {
  return request("/api/v1/dashboard/study-plan", {}, token);
}

// ── Flashcard Stats ─────────────────────────────────

export async function getFlashcardStats(token: string) {
  return request("/api/v1/flashcards/stats", {}, token);
}

export async function saveFlashcardFromQA(
  body: { question: string; answer: string; source_file?: string },
  token: string
) {
  return request("/api/v1/flashcards/from-qa", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

// ── Quiz Grade (AI semantic grading) ─────────────────

export async function gradeQuizAnswer(
  body: { question: string; user_answer: string; correct_answer: string; question_type: string; model?: string },
  token: string
) {
  return request("/api/v1/quiz/grade", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

// ── RAG Translate ────────────────────────────────────

export async function translateAnswer(
  body: { text: string; target_language: string; question?: string },
  token: string
) {
  return request("/api/v1/rag/translate", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

// ── Quiz Stats ──────────────────────────────────────

export async function getQuizStats(token: string) {
  return request("/api/v1/quiz/stats", {}, token);
}

export async function getUserStats(token: string) {
  return request("/api/v1/quiz/user-stats", {}, token);
}

// ── Canvas Dismiss ──────────────────────────────────

export async function dismissCanvasEvent(eventId: string, token: string) {
  return request("/api/v1/canvas/dismiss", {
    method: "POST",
    body: JSON.stringify({ event_id: eventId }),
  }, token);
}

export async function getDismissedEvents(token: string) {
  return request("/api/v1/canvas/dismissed", {}, token);
}

// ── Learn More ──────────────────────────────────────

export async function getLearnMore(topic: string, token: string) {
  return request(`/api/v1/learn-more?topic=${encodeURIComponent(topic)}`, {}, token);
}

export async function checkTopicAmbiguity(topic: string, token: string, answer?: string) {
  const params = new URLSearchParams({ topic });
  if (answer) params.append("answer", answer);
  return request(`/api/v1/learn-more/ambiguity?${params}`, {}, token);
}

// ── Images ──────────────────────────────────────────

export async function generateImage(concept: string, token: string, model?: string) {
  return request("/api/v1/images/generate", {
    method: "POST",
    body: JSON.stringify({ concept, model: model || "Llama 3.3 70B" }),
  }, token);
}

export async function generateDalleImage(prompt: string, token: string) {
  return request("/api/v1/images/dalle", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  }, token);
}

export async function generateChart(
  body: { chart_type: string; title: string; labels: string[]; values: number[]; xlabel?: string; ylabel?: string },
  token: string
) {
  return request("/api/v1/images/chart", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

export async function generateFlowchart(
  body: { title: string; steps: string[] },
  token: string
) {
  return request("/api/v1/images/flowchart", {
    method: "POST",
    body: JSON.stringify(body),
  }, token);
}

// ── Constants ────────────────────────────────────────

export const AVAILABLE_MODELS: ModelOption[] = [
  { label: "Llama 3.3 70B", value: "Llama 3.3 70B", badge: "Free", provider: "Groq" },
  { label: "Llama 3.1 8B", value: "Llama 3.1 8B", badge: "Free", provider: "Groq" },
  { label: "Gemini 2.0 Flash", value: "Gemini 2.0 Flash", badge: "Free", provider: "Google" },
  { label: "Gemini 1.5 Flash", value: "Gemini 1.5 Flash", badge: "Free", provider: "Google" },
  { label: "GPT-4o", value: "GPT-4o", badge: "Premium", provider: "OpenAI" },
  { label: "GPT-4o mini", value: "GPT-4o mini", badge: "Premium", provider: "OpenAI" },
];

export const DEFAULT_MODEL = "Llama 3.3 70B";

export const LANGUAGES = [
  "English",
  "Nepali",
  "Hindi",
  "Spanish",
  "French",
  "German",
  "Chinese",
  "Japanese",
  "Korean",
  "Arabic",
  "Portuguese",
];

