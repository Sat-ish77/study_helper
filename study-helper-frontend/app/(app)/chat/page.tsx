"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Send, Mic, Loader2, Globe } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { useVoiceInput } from "@/hooks/useVoiceInput";
import { createConversation, updateConversation, LANGUAGES } from "@/lib/api";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { EmptyState } from "@/components/chat/EmptyState";
import { StreamingCursor } from "@/components/chat/StreamingCursor";
import type { Message } from "@/types";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const { 
    messages, 
    addMessage, 
    setMessages, 
    token, 
    selectedModel,
    selectedLanguage,
    currentConversationId,
    setCurrentConversation,
  } = useAppStore();
  const { streaming, streamingContent, stream } = useStreamingChat();
  const { recording, transcribing, recordAndTranscribe } = useVoiceInput();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  const handleSend = async () => {
    if (!input.trim() || !token) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
    };

    addMessage(userMessage);
    setInput("");

    try {
      const history = messages.slice(-10).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const result = await stream(
        {
          question: userMessage.content,
          model: selectedModel,
          mode: "medium",
          web_fallback: true,
          language: selectedLanguage,
          history,
        },
        token
      );

      if (result) {
        const assistantMessage: Message = {
          role: "assistant",
          content: result.answer,
          file_sources: result.file_sources,
          web_sources: result.web_sources,
          raw_sources: result.raw_sources,
          used_web: result.used_web,
        };
        addMessage(assistantMessage);

        // Save conversation
        if (!currentConversationId) {
          const title = userMessage.content.slice(0, 50);
          const newConv = await createConversation(
            { title, messages: [...messages, userMessage, assistantMessage] },
            token
          ) as any;
          setCurrentConversation(newConv.id);
        } else {
          await updateConversation(
            currentConversationId,
            [...messages, userMessage, assistantMessage],
            token
          );
        }
      }
    } catch (err) {
      console.error("Failed to send message:", err);
      addMessage({
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
      });
    }
  };

  const handleVoiceInput = async () => {
    if (!token) return;

    const text = await recordAndTranscribe(token);
    if (text) {
      setInput(text);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 px-4">
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={setInput} />
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              isUser={message.role === "user"}
            />
          ))
        )}
        {streaming && (
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-background-card flex items-center justify-center text-sm">🤖</div>
            <div className="flex-1 bg-background-card border border-border-subtle rounded-xl p-4">
              <p className="text-text-primary whitespace-pre-wrap">{streamingContent}</p>
              <StreamingCursor />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="mt-4 px-4">
        <div className="flex items-end gap-2 bg-background-card border border-border-subtle rounded-xl p-2">
          <button
            onClick={handleVoiceInput}
            disabled={transcribing || !token}
            title={recording ? "Click to stop & transcribe" : "Click to record"}
            className={`
              p-2.5 rounded-lg transition-all relative
              ${recording 
                ? "bg-status-error/20 text-status-error" 
                : transcribing
                  ? "bg-accent/20 text-accent"
                  : "text-text-tertiary hover:text-text-primary hover:bg-white/5"
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            {recording && (
              <>
                <motion.div
                  className="absolute inset-0 rounded-lg bg-status-error/20"
                  animate={{ scale: [1, 1.6, 1] }}
                  transition={{ repeat: Infinity, duration: 1.2 }}
                />
                <motion.div
                  className="absolute inset-0 rounded-lg bg-status-error/10"
                  animate={{ scale: [1, 2, 1] }}
                  transition={{ repeat: Infinity, duration: 1.2, delay: 0.2 }}
                />
              </>
            )}
            {transcribing ? <Loader2 size={20} className="animate-spin" /> : <Mic size={20} />}
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask anything about your notes..."
            className="flex-1 bg-transparent text-text-primary placeholder:text-text-tertiary resize-none outline-none py-2.5 min-h-[44px] max-h-32"
            rows={1}
            style={{ height: "auto" }}
          />
          <div className="relative">
            <select
              value={selectedLanguage}
              onChange={(e) => useAppStore.getState().setSelectedLanguage(e.target.value)}
              className="appearance-none bg-transparent text-[11px] text-text-tertiary hover:text-text-secondary pl-6 pr-1 py-2.5 rounded-lg cursor-pointer outline-none transition-colors"
              title="Response language"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang} value={lang} className="bg-background-card text-text-primary">
                  {lang}
                </option>
              ))}
            </select>
            <Globe size={13} className="absolute left-1.5 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none" />
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || streaming || !token}
            className="p-2.5 rounded-lg bg-accent text-background-primary hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {streaming ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
          </button>
        </div>
        <p className="text-center text-xs text-text-tertiary mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
