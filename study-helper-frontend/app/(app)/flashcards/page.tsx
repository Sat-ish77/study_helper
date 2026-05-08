"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Search, Trash2, Plus, Sparkles, Check } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { getFlashcards, reviewFlashcard, deleteFlashcard, generateFlashcardsFromDoc, bulkSaveFlashcards, getDocuments } from "@/lib/api";
import type { Flashcard, Document } from "@/types";
import { TiltCard } from "@/components/ui/TiltCard";

const QUALITY_RATINGS = [
  { value: 0, label: "Forgot", emoji: "😵", color: "text-status-error" },
  { value: 3, label: "Hard", emoji: "😬", color: "text-status-warning" },
  { value: 4, label: "Good", emoji: "😊", color: "text-status-success" },
  { value: 5, label: "Easy", emoji: "🎯", color: "text-accent" },
];

export default function FlashcardsPage() {
  const { token, selectedModel } = useAppStore();
  const [activeTab, setActiveTab] = useState<"review" | "browse" | "generate">("review");
  const [cards, setCards] = useState<Flashcard[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [numCards, setNumCards] = useState(10);
  const [generating, setGenerating] = useState(false);
  const [generatedCards, setGeneratedCards] = useState<{ question: string; answer: string }[]>([]);

  useEffect(() => {
    fetchCards();
    if (token) {
      getDocuments(token).then((data: any) => setDocuments(data.documents || []));
    }
  }, [token]);

  const fetchCards = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const data = await getFlashcards(true, token) as any;
      setCards(data.cards || []);
    } catch (err) {
      console.error("Failed to fetch flashcards:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRate = async (quality: number) => {
    if (!token) return;

    const card = cards[currentIndex];
    try {
      await reviewFlashcard(card.id, quality, token);
      setIsFlipped(false);
      setTimeout(() => {
        fetchCards();
        setCurrentIndex(0);
      }, 300);
    } catch (err) {
      console.error("Failed to review card:", err);
    }
  };

  const handleDelete = async (id: string) => {
    if (!token) return;
    try {
      await deleteFlashcard(id, token);
      fetchCards();
    } catch (err) {
      console.error("Failed to delete card:", err);
    }
  };

  const handleGenerate = async () => {
    if (!token) return;
    setGenerating(true);
    try {
      const data = await generateFlashcardsFromDoc(
        { source_file: selectedDoc, num_cards: numCards, model: selectedModel },
        token
      ) as any;
      setGeneratedCards(data.cards || []);
    } catch (err) {
      console.error("Failed to generate cards:", err);
    } finally {
      setGenerating(false);
    }
  };

  const handleSaveGenerated = async () => {
    if (!token) return;
    try {
      await bulkSaveFlashcards(
        generatedCards.map((c) => ({ ...c, source_file: selectedDoc })),
        token
      );
      setGeneratedCards([]);
      setActiveTab("review");
      fetchCards();
    } catch (err) {
      console.error("Failed to save cards:", err);
    }
  };

  const filteredCards = cards.filter(
    (c) =>
      c.question.toLowerCase().includes(search.toLowerCase()) ||
      c.answer.toLowerCase().includes(search.toLowerCase())
  );

  if (loading && activeTab === "review") {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-serif text-text-primary mb-6">Flashcards</h1>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {["review", "browse", "generate"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as typeof activeTab)}
            className={`
              px-4 py-2 rounded-lg text-sm capitalize transition-all
              ${activeTab === tab
                ? "bg-accent text-background-primary"
                : "bg-background-secondary text-text-secondary hover:text-text-primary"
              }
            `}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Review Tab */}
      {activeTab === "review" && (
        <div className="max-w-xl mx-auto">
          {cards.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-text-secondary mb-4">No cards due for review!</p>
              <button
                onClick={() => setActiveTab("generate")}
                className="btn-amber"
              >
                <Plus size={18} className="mr-2" />
                Generate Flashcards
              </button>
            </div>
          ) : (
            <>
              {/* Progress */}
              <div className="text-center mb-4 text-sm text-text-secondary">
                Card {currentIndex + 1} of {cards.length}
              </div>

              {/* Card */}
              <div
                className="relative h-80 cursor-pointer"
                style={{ perspective: "1200px" }}
                onClick={() => setIsFlipped(!isFlipped)}
              >
                <motion.div
                  style={{ transformStyle: "preserve-3d" }}
                  animate={{ rotateY: isFlipped ? 180 : 0 }}
                  transition={{ duration: 0.5, ease: "easeInOut" }}
                  className="relative w-full h-full"
                >
                  {/* Front */}
                  <div
                    style={{ backfaceVisibility: "hidden" }}
                    className="absolute inset-0 rounded-2xl bg-background-card border border-border-subtle flex flex-col items-center justify-center p-8"
                  >
                    <p className="text-xl text-center font-serif text-text-primary">
                      {cards[currentIndex]?.question}
                    </p>
                    <p className="absolute bottom-4 text-xs text-text-tertiary">
                      Tap to reveal answer
                    </p>
                  </div>

                  {/* Back */}
                  <div
                    style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
                    className="absolute inset-0 rounded-2xl bg-background-card border border-accent/20 flex flex-col items-center justify-center p-8"
                  >
                    <p className="text-lg text-center text-text-primary">
                      {cards[currentIndex]?.answer}
                    </p>
                  </div>
                </motion.div>
              </div>

              {/* Rating Buttons */}
              {isFlipped && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 grid grid-cols-4 gap-2"
                >
                  {QUALITY_RATINGS.map((rating) => (
                    <button
                      key={rating.value}
                      onClick={() => handleRate(rating.value)}
                      className="flex flex-col items-center gap-1 p-3 rounded-xl bg-background-card border border-border-subtle hover:border-border-default transition-colors"
                    >
                      <span className="text-2xl">{rating.emoji}</span>
                      <span className={`text-xs font-medium ${rating.color}`}>{rating.label}</span>
                    </button>
                  ))}
                </motion.div>
              )}
            </>
          )}
        </div>
      )}

      {/* Browse Tab */}
      {activeTab === "browse" && (
        <div>
          <div className="relative mb-6">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search flashcards..."
              className="input pl-10"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredCards.map((card, index) => (
              <motion.div
                key={card.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <TiltCard className="group relative p-4 rounded-xl card card-glow">
                  <p className="text-sm text-text-primary line-clamp-2 mb-2">{card.question}</p>
                  <p className="text-xs text-text-secondary line-clamp-2">{card.answer}</p>
                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-[10px] text-text-tertiary">
                      Next: {new Date(card.next_review).toLocaleDateString()}
                    </span>
                    <button
                      onClick={() => handleDelete(card.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-text-tertiary hover:text-status-error transition-all"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </TiltCard>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Generate Tab */}
      {activeTab === "generate" && (
        <div className="max-w-2xl mx-auto">
          <div className="card space-y-4 mb-6">
            <div>
              <label className="block text-sm text-text-secondary mb-2">From Document</label>
              <select
                value={selectedDoc}
                onChange={(e) => setSelectedDoc(e.target.value)}
                className="input"
              >
                <option value="">Select a document...</option>
                {documents.map((doc) => (
                  <option key={doc.source_file} value={doc.source_file}>
                    {doc.source_file}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-text-secondary mb-2">
                Number of Cards: {numCards}
              </label>
              <input
                type="range"
                min="5"
                max="50"
                value={numCards}
                onChange={(e) => setNumCards(Number(e.target.value))}
                className="w-full accent-accent"
              />
            </div>

            <motion.button
              onClick={handleGenerate}
              disabled={!selectedDoc || generating}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full btn-amber disabled:opacity-50"
            >
              {generating ? (
                <>
                  <Loader2 size={18} className="animate-spin mr-2" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles size={18} className="mr-2" />
                  Generate Flashcards
                </>
              )}
            </motion.button>
          </div>

          {/* Generated Cards Preview */}
          {generatedCards.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h3 className="text-lg font-serif text-text-primary mb-4">
                Preview ({generatedCards.length} cards)
              </h3>
              <div className="space-y-3 mb-6">
                {generatedCards.map((card, i) => (
                  <div
                    key={i}
                    className="p-4 rounded-xl bg-background-card border border-border-subtle"
                  >
                    <p className="text-sm font-medium text-text-primary mb-1">{card.question}</p>
                    <p className="text-sm text-text-secondary">{card.answer}</p>
                  </div>
                ))}
              </div>
              <motion.button
                onClick={handleSaveGenerated}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full btn-amber"
              >
                <Check size={18} className="mr-2" />
                Save {generatedCards.length} Flashcards
              </motion.button>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
