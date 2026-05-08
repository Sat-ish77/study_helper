"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { useAppStore } from "@/store/useAppStore";

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 17) return "Good afternoon";
  if (hour >= 17 && hour < 21) return "Good evening";
  return "Working late";
}

function getFirstName(email: string | null): string {
  if (!email) return "there";
  return email.split("@")[0].split(".")[0].split("_")[0];
}

interface WordProps {
  text: string;
  index: number;
}

function AnimatedWord({ text, index }: WordProps) {
  return (
    <motion.span
      className="inline-block mr-[0.25em]"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.5,
        delay: index * 0.08,
        ease: [0.76, 0, 0.24, 1] as [number, number, number, number],
      }}
    >
      {text}
    </motion.span>
  );
}

export function DashboardGreeting() {
  const { userEmail } = useAppStore();

  const words = useMemo(() => {
    const greeting = getGreeting();
    const firstName = getFirstName(userEmail);
    return [greeting + ",", firstName];
  }, [userEmail]);

  return (
    <div className="mb-6">
      <h2 className="text-2xl font-serif" style={{ color: "var(--text-primary)" }}>
        {words.map((word, i) => (
          <AnimatedWord key={i} text={word} index={i} />
        ))}
      </h2>
    </div>
  );
}
