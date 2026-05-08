"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, ExternalLink } from "lucide-react";
import { QuickActions } from "./QuickActions";
import { LearnMorePanel } from "./LearnMorePanel";
import type { Message } from "@/types";

interface MessageBubbleProps {
  message: Message;
  isUser: boolean;
}

export function MessageBubble({ message, isUser }: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      {/* Avatar */}
      <div
        className={`
          w-8 h-8 rounded-full flex items-center justify-center text-sm shrink-0
          ${isUser ? "bg-accent/20 text-accent" : "bg-background-card border border-border-subtle"}
        `}
      >
        {isUser ? "👤" : "🤖"}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? "items-end" : ""}`}>
        <div
          className={`
            inline-block max-w-full rounded-xl p-4
            ${isUser
              ? "bg-accent/10 border border-accent/20 text-text-primary"
              : "bg-background-card border border-border-subtle text-text-primary"
            }
          `}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Quick Actions (only for assistant messages) */}
        {!isUser && message.content && (
          <QuickActions content={message.content} />
        )}

        {/* Learn More (only for assistant messages) */}
        {!isUser && message.content && (
          <LearnMorePanel content={message.content} />
        )}

        {/* Sources (only for assistant messages) */}
        {!isUser && (message.file_sources?.length || message.web_sources?.length) && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-text-tertiary hover:text-text-secondary transition-colors"
            >
              <ChevronDown
                size={14}
                className={`transition-transform ${showSources ? "rotate-180" : ""}`}
              />
              Sources ({(message.file_sources?.length || 0) + (message.web_sources?.length || 0)})
            </button>

            {showSources && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-2 space-y-1.5"
              >
                {message.raw_sources?.map((source, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 py-1.5 px-2 rounded-lg bg-white/3 border border-white/5 text-xs"
                  >
                    <span className="text-accent">📄</span>
                    <span className="text-text-secondary truncate">{source.source_file}</span>
                    {source.page_num && (
                      <span className="text-text-tertiary">Page {source.page_num}</span>
                    )}
                    <span className="text-text-tertiary ml-auto">
                      {(source.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
                {message.web_sources?.map((source, idx) => (
                  <a
                    key={`web-${idx}`}
                    href={source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 py-1.5 px-2 rounded-lg bg-white/3 border border-white/5 text-xs hover:bg-white/5 transition-colors"
                  >
                    <span className="text-accent">🌐</span>
                    <span className="text-text-secondary truncate">{source}</span>
                    <ExternalLink size={10} className="text-text-tertiary ml-auto" />
                  </a>
                ))}
              </motion.div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
