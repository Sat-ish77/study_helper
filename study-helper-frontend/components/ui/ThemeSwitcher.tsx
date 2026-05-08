"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Palette } from "lucide-react";
import { useAppStore, type ThemeId } from "@/store/useAppStore";

const THEMES: { id: ThemeId; color: string; label: string }[] = [
  { id: "amber", color: "#F59E0B", label: "Night" },
  { id: "cyan", color: "#06B6D4", label: "Ocean" },
  { id: "emerald", color: "#10B981", label: "Forest" },
  { id: "violet", color: "#8B5CF6", label: "Purple" },
];

export function ThemeSwitcher() {
  const { theme, setTheme } = useAppStore();
  const [open, setOpen] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 10 }}
            className="absolute bottom-14 right-0 p-2 rounded-xl glass flex gap-2"
          >
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => { setTheme(t.id); setOpen(false); }}
                title={t.label}
                className="relative w-8 h-8 rounded-full transition-transform hover:scale-110"
                style={{ background: t.color }}
              >
                {theme === t.id && (
                  <motion.div
                    layoutId="theme-ring"
                    className="absolute inset-[-3px] rounded-full border-2"
                    style={{ borderColor: t.color }}
                  />
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => setOpen(!open)}
        className="w-12 h-12 rounded-full glass flex items-center justify-center shadow-lg"
        style={{ color: "var(--accent)" }}
      >
        <Palette size={20} />
      </motion.button>
    </div>
  );
}
