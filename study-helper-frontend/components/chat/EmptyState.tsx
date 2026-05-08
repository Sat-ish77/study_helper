"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { BookOpen, HelpCircle, FileText, Calendar } from "lucide-react";
import { prepare, layout } from "@chenglou/pretext";

const SUGGESTIONS = [
  { icon: HelpCircle, text: "Explain a concept from my notes", prompt: "Explain the concept of photosynthesis from my Biology notes" },
  { icon: BookOpen, text: "Quiz me on a topic", prompt: "Create a quiz about algorithms and data structures" },
  { icon: FileText, text: "Summarize my lecture PDF", prompt: "Summarize the key points from my lecture PDF" },
  { icon: Calendar, text: "What's due on my calendar?", prompt: "What assignments are due soon based on my calendar?" },
];

const SUBTITLE = "What would you like to learn today?";

interface EmptyStateProps {
  onSuggestionClick: (prompt: string) => void;
}

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  const subtitleRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    // Use pretext to measure subtitle and set exact height to prevent layout shift
    try {
      const prepared = prepare(SUBTITLE, "18px DM Sans");
      const measured = layout(prepared, 400, 24);
      if (subtitleRef.current && measured.height) {
        subtitleRef.current.style.minHeight = `${measured.height}px`;
      }
    } catch {
      // graceful fallback
    }
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-serif shimmer-text mb-3">Study Helper</h1>
        <p ref={subtitleRef} className="text-lg mb-8" style={{ color: "var(--text-secondary)" }}>
          {SUBTITLE}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg mx-auto">
          {SUGGESTIONS.map((suggestion, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSuggestionClick(suggestion.prompt)}
              className="flex items-center gap-3 p-4 rounded-xl glass card-glow text-left"
            >
              <div className="p-2 rounded-lg" style={{ background: "rgba(var(--accent-rgb), 0.1)", color: "var(--accent)" }}>
                <suggestion.icon size={18} />
              </div>
              <span className="text-sm text-text-primary">{suggestion.text}</span>
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
