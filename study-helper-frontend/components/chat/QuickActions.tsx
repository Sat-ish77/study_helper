"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles, Code2, Globe, Volume2, BarChart3, CreditCard,
  Loader2, Check, Copy,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import {
  askQuestion, translateAnswer, synthesizeSpeech, saveFlashcardFromQA,
} from "@/lib/api";

interface QuickActionsProps {
  content: string;
  question?: string;
}

type ActionId = "simpler" | "technical" | "translate" | "listen" | "visualize" | "flashcard";

interface ActionDef {
  id: ActionId;
  icon: typeof Sparkles;
  label: string;
  tooltip: string;
}

const ACTIONS: ActionDef[] = [
  { id: "simpler", icon: Sparkles, label: "Simpler", tooltip: "Rewrite in simpler language" },
  { id: "technical", icon: Code2, label: "Technical", tooltip: "More technical detail" },
  { id: "translate", icon: Globe, label: "Translate", tooltip: "Translate to another language" },
  { id: "listen", icon: Volume2, label: "Listen", tooltip: "Text-to-speech" },
  { id: "flashcard", icon: CreditCard, label: "Save Card", tooltip: "Save as flashcard" },
];

const TRANSLATE_LANGS = [
  "Spanish", "French", "German", "Chinese", "Japanese",
  "Korean", "Arabic", "Portuguese", "Nepali", "Hindi",
];

export function QuickActions({ content, question }: QuickActionsProps) {
  const { token, selectedModel, selectedLanguage } = useAppStore();
  const [loading, setLoading] = useState<ActionId | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [done, setDone] = useState<ActionId | null>(null);
  const [showTranslateMenu, setShowTranslateMenu] = useState(false);

  const handleAction = async (action: ActionId, lang?: string) => {
    if (!token || loading) return;
    setLoading(action);
    setResult(null);
    setShowTranslateMenu(false);

    try {
      switch (action) {
        case "simpler": {
          const data = await askQuestion(
            {
              question: `Rewrite this in simpler, easier-to-understand language. Keep the same meaning but use shorter sentences and common words:\n\n${content}`,
              model: selectedModel,
              mode: "short",
              web_fallback: false,
              language: selectedLanguage,
              history: [],
            },
            token
          );
          setResult(data.answer);
          break;
        }
        case "technical": {
          const data = await askQuestion(
            {
              question: `Rewrite this with more technical depth and precision. Add relevant technical terminology, specific details, and formal academic language:\n\n${content}`,
              model: selectedModel,
              mode: "long",
              web_fallback: false,
              language: selectedLanguage,
              history: [],
            },
            token
          );
          setResult(data.answer);
          break;
        }
        case "translate": {
          const targetLang = lang || "Spanish";
          const data = await translateAnswer(
            { text: content, target_language: targetLang, question },
            token
          ) as any;
          setResult(data.translated_text || data.translation || data.text);
          break;
        }
        case "listen": {
          const data = await synthesizeSpeech(
            { text: content.slice(0, 2000), language: selectedLanguage },
            token
          ) as any;
          if (data.audio_url) {
            const audio = new Audio(data.audio_url);
            audio.play();
          } else if (data.audio_base64) {
            const audio = new Audio(`data:audio/mp3;base64,${data.audio_base64}`);
            audio.play();
          }
          setDone("listen");
          setTimeout(() => setDone(null), 2000);
          break;
        }
        case "flashcard": {
          await saveFlashcardFromQA(
            {
              question: question || content.slice(0, 100),
              answer: content,
            },
            token
          );
          setDone("flashcard");
          setTimeout(() => setDone(null), 2000);
          break;
        }
      }
    } catch (err) {
      console.error(`Quick action ${action} failed:`, err);
    } finally {
      setLoading(null);
    }
  };

  const copyResult = () => {
    if (result) {
      navigator.clipboard.writeText(result);
    }
  };

  return (
    <div className="mt-2">
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-1.5 relative">
        {ACTIONS.map((action) => (
          <div key={action.id} className="relative">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                if (action.id === "translate") {
                  setShowTranslateMenu(!showTranslateMenu);
                } else {
                  handleAction(action.id);
                }
              }}
              disabled={loading !== null}
              className={`
                flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] transition-all
                ${done === action.id
                  ? "bg-status-success/10 text-status-success border border-status-success/20"
                  : loading === action.id
                    ? "bg-accent/10 text-accent border border-accent/20"
                    : "text-text-tertiary hover:text-text-secondary hover:bg-white/[0.04] border border-transparent"
                }
                disabled:opacity-50
              `}
              title={action.tooltip}
            >
              {loading === action.id ? (
                <Loader2 size={12} className="animate-spin" />
              ) : done === action.id ? (
                <Check size={12} />
              ) : (
                <action.icon size={12} />
              )}
              <span>{action.label}</span>
            </motion.button>

            {/* Translate Language Menu */}
            {action.id === "translate" && showTranslateMenu && (
              <AnimatePresence>
                <motion.div
                  initial={{ opacity: 0, y: -4, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.95 }}
                  className="absolute bottom-full left-0 mb-1 p-1.5 rounded-lg bg-background-card border border-border-subtle shadow-lg z-20 min-w-[120px]"
                >
                  {TRANSLATE_LANGS.map((lang) => (
                    <button
                      key={lang}
                      onClick={() => handleAction("translate", lang)}
                      className="block w-full text-left px-2 py-1 rounded text-xs text-text-secondary hover:text-text-primary hover:bg-white/[0.06] transition-colors"
                    >
                      {lang}
                    </button>
                  ))}
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        ))}
      </div>

      {/* Result Panel */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 p-3 rounded-xl bg-accent/5 border border-accent/10"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] text-accent uppercase tracking-wider font-medium">Result</span>
              <button
                onClick={copyResult}
                className="p-1 rounded hover:bg-white/5 text-text-tertiary hover:text-text-secondary transition-colors"
                title="Copy"
              >
                <Copy size={12} />
              </button>
            </div>
            <p className="text-sm text-text-secondary whitespace-pre-wrap">{result}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
