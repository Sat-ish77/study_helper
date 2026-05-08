"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Sparkles } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { AVAILABLE_MODELS, DEFAULT_MODEL } from "@/lib/api";

export function Topbar() {
  const { selectedModel, setSelectedModel } = useAppStore();
  const [isOpen, setIsOpen] = useState(false);

  const currentModel = AVAILABLE_MODELS.find((m) => m.value === selectedModel) || AVAILABLE_MODELS[0];

  return (
    <header className="h-14 glass flex items-center justify-between px-4 sticky top-0 z-40">
      <div />

      {/* Model Selector */}
      <div className="relative">
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background-card border border-border-subtle hover:border-border-default transition-colors"
        >
          <Sparkles size={14} className="text-accent" />
          <span className="text-sm text-text-primary">{currentModel.label}</span>
          <span
            className={`
              text-[10px] px-1.5 py-0.5 rounded font-medium
              ${currentModel.badge === "Free" ? "bg-green-500/20 text-green-400" : "bg-accent/20 text-accent"}
            `}
          >
            {currentModel.badge}
          </span>
          <ChevronDown size={14} className="text-text-tertiary" />
        </motion.button>

        {isOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute right-0 top-full mt-2 w-56 glass rounded-xl shadow-xl z-50 py-1"
            >
              {AVAILABLE_MODELS.map((model) => (
                <button
                  key={model.value}
                  onClick={() => {
                    setSelectedModel(model.value);
                    setIsOpen(false);
                  }}
                  className={`
                    w-full flex items-center justify-between px-3 py-2 text-sm
                    hover:bg-white/5 transition-colors
                    ${selectedModel === model.value ? "text-accent" : "text-text-primary"}
                  `}
                >
                  <span>{model.label}</span>
                  <span
                    className={`
                      text-[10px] px-1.5 py-0.5 rounded font-medium
                      ${model.badge === "Free" ? "bg-green-500/20 text-green-400" : "bg-accent/20 text-accent"}
                    `}
                  >
                    {model.badge}
                  </span>
                </button>
              ))}
            </motion.div>
          </>
        )}
      </div>
    </header>
  );
}
