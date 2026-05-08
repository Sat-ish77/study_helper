"use client";

import { useState, useRef, useCallback } from "react";
import { askStream } from "@/lib/api";
import type { AskRequest, AskResponse } from "@/types";

interface StreamResult {
  answer: string;
  file_sources: string[];
  web_sources: string[];
  raw_sources: AskResponse["raw_sources"];
  used_web: boolean;
}

export function useStreamingChat() {
  const [streaming, setStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const stream = useCallback(
    async (body: AskRequest, token: string): Promise<StreamResult | null> => {
      setStreaming(true);
      setStreamingContent("");
      abortRef.current = new AbortController();

      try {
        const response = await askStream(body, token);
        const reader = response.body?.getReader();
        if (!reader) throw new Error("No readable stream");

        const decoder = new TextDecoder();
        let accumulated = "";
        let finalResult: StreamResult | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const data = JSON.parse(jsonStr);

              if (data.chunk) {
                accumulated += data.chunk;
                setStreamingContent(accumulated);
              }

              if (data.done) {
                finalResult = {
                  answer: accumulated,
                  file_sources: data.file_sources ?? [],
                  web_sources: data.web_sources ?? [],
                  raw_sources: data.raw_sources ?? [],
                  used_web: data.used_web ?? false,
                };
              }
            } catch {
              // skip malformed JSON chunks
            }
          }
        }

        return finalResult;
      } catch (err) {
        if ((err as Error).name === "AbortError") return null;
        throw err;
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    []
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, []);

  return { streaming, streamingContent, stream, abort };
}
