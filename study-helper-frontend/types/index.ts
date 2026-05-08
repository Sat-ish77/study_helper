export interface RawSource {
  source_file: string;
  page_num: number | null;
  similarity: number;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  file_sources?: string[];
  web_sources?: string[];
  raw_sources?: RawSource[];
  used_web?: boolean;
}

export interface AskRequest {
  question: string;
  model: string;
  mode: "short" | "medium" | "long";
  web_fallback: boolean;
  language: string;
  history: Array<{ role: string; content: string }>;
}

export interface AskResponse {
  answer: string;
  file_sources: string[];
  web_sources: string[];
  raw_sources: RawSource[];
  used_web: boolean;
}

export interface Document {
  source_file: string;
  filetype: string;
}

export interface QuizQuestion {
  type: "mcq" | "true_false" | "short_answer" | "medium_answer" | "fill_blank";
  question: string;
  options: string[];
  correct_answer: string;
  explanation: string;
}

export interface Flashcard {
  id: string;
  user_id: string;
  question: string;
  answer: string;
  source_file: string;
  ease_factor: number;
  interval_days: number;
  repetitions: number;
  next_review: string;
  last_review: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationFull extends Conversation {
  messages: Message[];
}

export interface CanvasEvent {
  uid: string;
  title: string;
  start: string;
  end: string;
  description: string;
  location: string;
}

export interface DashboardStats {
  documents: number;
  flashcards_total: number;
  flashcards_due: number;
  quiz_attempts: number;
  quiz_avg_percentage: number;
  recent_scores: RecentScore[];
}

export interface RecentScore {
  topic: string;
  score: number;
  total: number;
  percentage: number;
  created_at: string;
  course_name: string | null;
}

export interface StudyPlan {
  greeting: string;
  sessions: StudySession[];
  tip: string;
}

export interface StudySession {
  time: string;
  activity: string;
  reason: string;
}

export interface ModelOption {
  label: string;
  value: string;
  badge: string;
  provider: string;
}
