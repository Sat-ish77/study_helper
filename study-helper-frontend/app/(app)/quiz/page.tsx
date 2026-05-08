"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle, XCircle, Loader2, ArrowRight, RotateCcw,
  Plus, Trophy, Target, Flame, Globe, Send,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import {
  generateQuiz, saveQuizScore, getDocuments, bulkSaveFlashcards,
  gradeQuizAnswer, getQuizStats, LANGUAGES,
} from "@/lib/api";
import type { QuizQuestion, Document } from "@/types";

const DIFFICULTIES = ["easy", "medium", "hard"] as const;
const QUESTION_COUNTS = [5, 10, 20, 50] as const;
const QUESTION_TYPES = [
  { value: "mcq", label: "Multiple Choice" },
  { value: "true_false", label: "True/False" },
  { value: "short_answer", label: "Short Answer" },
  { value: "medium_answer", label: "Medium Answer" },
  { value: "fill_blank", label: "Fill in Blank" },
];

const NEEDS_TEXT_INPUT = ["short_answer", "medium_answer", "fill_blank"];

interface GradeResult {
  is_correct: boolean;
  score: number;
  feedback: string;
}

export default function QuizPage() {
  const { token, selectedModel } = useAppStore();
  const [step, setStep] = useState<"setup" | "quiz" | "results">("setup");
  const [topic, setTopic] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [difficulty, setDifficulty] = useState("medium");
  const [questionTypes, setQuestionTypes] = useState<string[]>(["mcq"]);
  const [quizLanguage, setQuizLanguage] = useState("English");
  const [loading, setLoading] = useState(false);
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [showExplanation, setShowExplanation] = useState(false);
  const [score, setScore] = useState(0);
  const [textAnswer, setTextAnswer] = useState("");
  const [grading, setGrading] = useState(false);
  const [gradeResults, setGradeResults] = useState<Record<number, GradeResult>>({});
  const [sessionStats, setSessionStats] = useState<{
    quiz_avg_percentage?: number;
    total_quizzes?: number;
    recent_scores?: { topic: string; percentage: number }[];
  } | null>(null);

  useEffect(() => {
    if (!token) return;
    getDocuments(token).then((data: any) => setDocuments(data.documents || []));
    getQuizStats(token).then((data: any) => setSessionStats(data)).catch(() => {});
  }, [token]);

  const handleGenerate = async () => {
    if (!token) return;
    setLoading(true);

    try {
      const data = await generateQuiz(
        {
          topic,
          num_questions: numQuestions,
          difficulty,
          question_types: questionTypes,
          language: quizLanguage,
          model: selectedModel,
          source_file: selectedDoc || undefined,
        },
        token
      ) as any;
      setQuestions(data.questions || []);
      setStep("quiz");
    } catch (err) {
      console.error("Failed to generate quiz:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswer = (answer: string) => {
    setAnswers((prev) => ({ ...prev, [currentIndex]: answer }));
    setShowExplanation(true);

    const currentQuestion = questions[currentIndex];
    const isCorrect = answer === currentQuestion.correct_answer;
    if (isCorrect) setScore((s) => s + 1);
  };

  const handleTextAnswer = async () => {
    if (!textAnswer.trim() || !token) return;
    const currentQuestion = questions[currentIndex];

    setAnswers((prev) => ({ ...prev, [currentIndex]: textAnswer.trim() }));
    setGrading(true);

    try {
      const result = await gradeQuizAnswer(
        {
          question: currentQuestion.question,
          user_answer: textAnswer.trim(),
          correct_answer: currentQuestion.correct_answer,
          question_type: currentQuestion.type,
          model: selectedModel,
        },
        token
      ) as GradeResult;

      setGradeResults((prev) => ({ ...prev, [currentIndex]: result }));
      if (result.is_correct) setScore((s) => s + 1);
    } catch {
      const fallback = textAnswer.trim().toLowerCase() === currentQuestion.correct_answer.toLowerCase();
      setGradeResults((prev) => ({
        ...prev,
        [currentIndex]: {
          is_correct: fallback,
          score: fallback ? 100 : 0,
          feedback: fallback ? "Correct!" : `Expected: ${currentQuestion.correct_answer}`,
        },
      }));
      if (fallback) setScore((s) => s + 1);
    } finally {
      setGrading(false);
      setShowExplanation(true);
    }
  };

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex((i) => i + 1);
      setShowExplanation(false);
      setTextAnswer("");
    } else {
      setStep("results");
      if (token) {
        saveQuizScore(
          {
            topic: topic || "General Quiz",
            score,
            total: questions.length,
          },
          token
        );
      }
    }
  };

  const handleSaveFlashcards = async () => {
    if (!token) return;
    const wrongQuestions = questions
      .map((q, i) => ({ ...q, index: i }))
      .filter((q) => {
        const grade = gradeResults[q.index];
        if (grade) return !grade.is_correct;
        return answers[q.index] !== q.correct_answer;
      });

    if (wrongQuestions.length === 0) return;

    const cards = wrongQuestions.map((q) => ({
      question: q.question,
      answer: q.correct_answer,
    }));

    await bulkSaveFlashcards(cards, token);
    alert(`Saved ${cards.length} flashcards!`);
  };

  const reset = () => {
    setStep("setup");
    setQuestions([]);
    setCurrentIndex(0);
    setAnswers({});
    setScore(0);
    setShowExplanation(false);
    setTextAnswer("");
    setGradeResults({});
  };

  if (step === "setup") {
    return (
      <div className="flex gap-6 max-w-4xl mx-auto">
        <div className="flex-1 max-w-2xl">
          <h1 className="text-2xl font-serif text-text-primary mb-6">Quiz Lab</h1>

          <div className="card space-y-6">
            {/* Topic */}
            <div>
              <label className="block text-sm text-text-secondary mb-2">Topic (optional)</label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Photosynthesis, Data Structures..."
                className="input"
              />
            </div>

            {/* Document */}
            {documents.length > 0 && (
              <div>
                <label className="block text-sm text-text-secondary mb-2">From Document (optional)</label>
                <select
                  value={selectedDoc}
                  onChange={(e) => setSelectedDoc(e.target.value)}
                  className="input"
                >
                  <option value="">Any document</option>
                  {documents.map((doc) => (
                    <option key={doc.source_file} value={doc.source_file}>
                      {doc.source_file}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Language */}
            <div>
              <label className="block text-sm text-text-secondary mb-2">
                <Globe size={14} className="inline mr-1" />
                Quiz Language
              </label>
              <select
                value={quizLanguage}
                onChange={(e) => setQuizLanguage(e.target.value)}
                className="input"
              >
                {LANGUAGES.map((lang) => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
            </div>

            {/* Number of Questions */}
            <div>
              <label className="block text-sm text-text-secondary mb-2">Questions</label>
              <div className="flex gap-2">
                {QUESTION_COUNTS.map((count) => (
                  <button
                    key={count}
                    onClick={() => setNumQuestions(count)}
                    className={`
                      px-4 py-2 rounded-lg text-sm transition-all
                      ${numQuestions === count
                        ? "bg-accent text-background-primary"
                        : "bg-background-secondary text-text-secondary hover:text-text-primary"
                      }
                    `}
                  >
                    {count}
                  </button>
                ))}
              </div>
            </div>

            {/* Difficulty */}
            <div>
              <label className="block text-sm text-text-secondary mb-2">Difficulty</label>
              <div className="flex gap-2">
                {DIFFICULTIES.map((diff) => (
                  <button
                    key={diff}
                    onClick={() => setDifficulty(diff)}
                    className={`
                      px-4 py-2 rounded-lg text-sm capitalize transition-all
                      ${difficulty === diff
                        ? "bg-accent text-background-primary"
                        : "bg-background-secondary text-text-secondary hover:text-text-primary"
                      }
                    `}
                  >
                    {diff}
                  </button>
                ))}
              </div>
            </div>

            {/* Question Types */}
            <div>
              <label className="block text-sm text-text-secondary mb-2">Question Types</label>
              <div className="flex flex-wrap gap-2">
                {QUESTION_TYPES.map((type) => (
                  <button
                    key={type.value}
                    onClick={() => {
                      setQuestionTypes((prev) =>
                        prev.includes(type.value)
                          ? prev.filter((t) => t !== type.value)
                          : [...prev, type.value]
                      );
                    }}
                    className={`
                      px-3 py-1.5 rounded-lg text-xs transition-all
                      ${questionTypes.includes(type.value)
                        ? "bg-accent/20 text-accent border border-accent/30"
                        : "bg-background-secondary text-text-secondary border border-border-subtle hover:text-text-primary"
                      }
                    `}
                  >
                    {type.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <motion.button
              onClick={handleGenerate}
              disabled={loading || questionTypes.length === 0}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full btn-amber disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin mr-2" />
                  Generating Quiz...
                </>
              ) : (
                "Generate Quiz"
              )}
            </motion.button>
          </div>
        </div>

        {/* Session Stats Sidebar */}
        <div className="w-56 hidden lg:block">
          <div className="card space-y-4">
            <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wider">Your Stats</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Trophy size={16} className="text-accent" />
                <div>
                  <p className="text-lg font-semibold text-text-primary">
                    {sessionStats?.quiz_avg_percentage ?? 0}%
                  </p>
                  <p className="text-[10px] text-text-tertiary">Average Score</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Target size={16} className="text-accent" />
                <div>
                  <p className="text-lg font-semibold text-text-primary">
                    {sessionStats?.total_quizzes ?? 0}
                  </p>
                  <p className="text-[10px] text-text-tertiary">Quizzes Taken</p>
                </div>
              </div>
            </div>
            {sessionStats?.recent_scores && sessionStats.recent_scores.length > 0 && (
              <div className="border-t border-border-subtle pt-3">
                <p className="text-[10px] text-text-tertiary uppercase tracking-wider mb-2">Recent</p>
                <div className="space-y-1.5">
                  {sessionStats.recent_scores.slice(0, 5).map((s, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-text-secondary truncate max-w-[120px]">{s.topic}</span>
                      <span className={`font-medium ${s.percentage >= 70 ? "text-status-success" : s.percentage >= 50 ? "text-status-warning" : "text-status-error"}`}>
                        {s.percentage}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (step === "quiz" && questions.length > 0) {
    const currentQuestion = questions[currentIndex];
    const hasAnswered = answers[currentIndex] !== undefined;
    const isTextType = NEEDS_TEXT_INPUT.includes(currentQuestion.type);
    const gradeResult = gradeResults[currentIndex];
    const isCorrect = hasAnswered && (gradeResult ? gradeResult.is_correct : answers[currentIndex] === currentQuestion.correct_answer);

    return (
      <div className="max-w-2xl mx-auto">
        {/* Progress */}
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-text-secondary mb-2">
            <span>Question {currentIndex + 1} of {questions.length}</span>
            <span className="flex items-center gap-1">
              <Flame size={14} className="text-accent" />
              Score: {score}/{currentIndex + (hasAnswered ? 1 : 0)}
            </span>
          </div>
          <div className="h-2 bg-background-secondary rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-accent"
              initial={{ width: 0 }}
              animate={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Question Card */}
        <div className="card mb-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="px-2 py-0.5 rounded bg-accent/10 text-accent text-[10px] uppercase tracking-wider">
              {currentQuestion.type.replace("_", " ")}
            </span>
          </div>
          <h2 className="text-xl font-serif text-text-primary mb-6">{currentQuestion.question}</h2>

          {/* MCQ / True-False options */}
          {!isTextType && currentQuestion.options && currentQuestion.options.length > 0 && (
            <div className="space-y-3">
              {currentQuestion.options.map((option, i) => {
                const isSelected = answers[currentIndex] === option;
                const showCorrect = showExplanation && option === currentQuestion.correct_answer;
                const showWrong = showExplanation && isSelected && !isCorrect;

                return (
                  <motion.button
                    key={i}
                    onClick={() => !hasAnswered && handleAnswer(option)}
                    disabled={hasAnswered}
                    whileHover={!hasAnswered ? { scale: 1.01 } : {}}
                    whileTap={!hasAnswered ? { scale: 0.99 } : {}}
                    className={`
                      w-full p-4 rounded-xl border text-left transition-all
                      ${showCorrect
                        ? "bg-status-success/10 border-status-success text-status-success"
                        : showWrong
                          ? "bg-status-error/10 border-status-error text-status-error"
                          : isSelected
                            ? "bg-accent/10 border-accent text-accent"
                            : "bg-background-secondary border-border-subtle text-text-primary hover:border-border-default"
                      }
                      ${hasAnswered ? "cursor-default" : "cursor-pointer"}
                    `}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium">{String.fromCharCode(65 + i)}.</span>
                      <span>{option}</span>
                      {showCorrect && <CheckCircle size={18} className="ml-auto" />}
                      {showWrong && <XCircle size={18} className="ml-auto" />}
                    </div>
                  </motion.button>
                );
              })}
            </div>
          )}

          {/* Text input for short/medium/fill_blank */}
          {isTextType && !hasAnswered && (
            <div className="space-y-3">
              <textarea
                value={textAnswer}
                onChange={(e) => setTextAnswer(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && currentQuestion.type !== "medium_answer") {
                    e.preventDefault();
                    handleTextAnswer();
                  }
                }}
                placeholder={
                  currentQuestion.type === "medium_answer"
                    ? "Write a detailed answer (2-4 sentences)..."
                    : currentQuestion.type === "fill_blank"
                      ? "Fill in the blank..."
                      : "Type your answer..."
                }
                rows={currentQuestion.type === "medium_answer" ? 4 : 2}
                className="w-full input resize-none"
              />
              <motion.button
                onClick={handleTextAnswer}
                disabled={!textAnswer.trim() || grading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="btn-amber disabled:opacity-50"
              >
                {grading ? (
                  <>
                    <Loader2 size={16} className="animate-spin mr-2" />
                    AI Grading...
                  </>
                ) : (
                  <>
                    <Send size={16} className="mr-2" />
                    Submit Answer
                  </>
                )}
              </motion.button>
            </div>
          )}

          {/* AI Grade Feedback */}
          {isTextType && gradeResult && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-4 p-4 rounded-xl border ${
                gradeResult.is_correct
                  ? "bg-status-success/10 border-status-success/30"
                  : "bg-status-error/10 border-status-error/30"
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                {gradeResult.is_correct ? (
                  <CheckCircle size={18} className="text-status-success" />
                ) : (
                  <XCircle size={18} className="text-status-error" />
                )}
                <span className={`text-sm font-medium ${gradeResult.is_correct ? "text-status-success" : "text-status-error"}`}>
                  Score: {gradeResult.score}/100
                </span>
              </div>
              <p className="text-sm text-text-secondary">{gradeResult.feedback}</p>
              {!gradeResult.is_correct && (
                <p className="text-sm text-text-secondary mt-2">
                  <strong className="text-accent">Expected:</strong> {currentQuestion.correct_answer}
                </p>
              )}
            </motion.div>
          )}

          {/* Explanation */}
          {showExplanation && currentQuestion.explanation && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 rounded-xl bg-accent/5 border border-accent/10"
            >
              <p className="text-sm text-text-secondary">
                <strong className="text-accent">Explanation:</strong> {currentQuestion.explanation}
              </p>
            </motion.div>
          )}
        </div>

        {/* Next Button */}
        {hasAnswered && !grading && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleNext}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full btn-amber"
          >
            {currentIndex < questions.length - 1 ? (
              <>
                Next Question <ArrowRight size={18} className="ml-2" />
              </>
            ) : (
              "View Results"
            )}
          </motion.button>
        )}
      </div>
    );
  }

  if (step === "results") {
    const percentage = Math.round((score / questions.length) * 100);

    return (
      <div className="max-w-2xl mx-auto text-center">
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 180, damping: 14 }}
          className="mb-8"
        >
          <div className="relative w-32 h-32 mx-auto mb-4">
            <svg className="w-full h-full -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="60"
                fill="none"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="8"
              />
              <motion.circle
                cx="64"
                cy="64"
                r="60"
                fill="none"
                stroke="#e8a44a"
                strokeWidth="8"
                strokeLinecap="round"
                initial={{ strokeDasharray: "0 377" }}
                animate={{ strokeDasharray: `${(percentage / 100) * 377} 377` }}
                transition={{ duration: 1, delay: 0.3 }}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-3xl font-bold text-accent">{percentage}%</span>
            </div>
          </div>
          <p className="text-xl text-text-primary">
            You scored {score} out of {questions.length}
          </p>
        </motion.div>

        <div className="flex gap-3 justify-center">
          <motion.button
            onClick={handleSaveFlashcards}
            disabled={score === questions.length}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="btn-amber disabled:opacity-50"
          >
            <Plus size={18} className="mr-2" />
            Save Wrong Answers as Flashcards
          </motion.button>
          <motion.button
            onClick={reset}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-4 py-2 rounded-lg border border-border-subtle text-text-secondary hover:text-text-primary transition-colors"
          >
            <RotateCcw size={18} className="mr-2 inline" />
            Try Again
          </motion.button>
        </div>
      </div>
    );
  }

  return null;
}
