"use client";

import { motion } from "framer-motion";

export function StreamingCursor() {
  return (
    <motion.span
      animate={{ opacity: [1, 0, 1] }}
      transition={{ repeat: Infinity, duration: 0.75, ease: "linear" }}
      className="inline-block w-[2px] h-4 bg-accent ml-0.5 align-middle rounded-full"
    />
  );
}
