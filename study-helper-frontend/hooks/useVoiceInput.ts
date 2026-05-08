"use client";

import { useState, useRef, useCallback } from "react";
import { transcribeAudio } from "@/lib/api";

const MAX_RECORDING_MS = 30_000;

export function useVoiceInput() {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/mp4";
      const mediaRecorder = new MediaRecorder(stream, { mimeType });

      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(250);
      setRecording(true);

      timeoutRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      }, MAX_RECORDING_MS);
    } catch (err: any) {
      setError(err.message || "Microphone access denied");
      console.error("Mic access error:", err);
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    return new Promise((resolve) => {
      const mr = mediaRecorderRef.current;
      if (!mr || mr.state === "inactive") {
        setRecording(false);
        resolve(new Blob([], { type: "audio/webm" }));
        return;
      }

      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        mr.stream.getTracks().forEach((t) => t.stop());
        mediaRecorderRef.current = null;
        chunksRef.current = [];
        setRecording(false);
        resolve(blob);
      };

      mr.stop();
    });
  }, []);

  const recordAndTranscribe = useCallback(
    async (token: string): Promise<string> => {
      if (recording) {
        const blob = await stopRecording();
        setTranscribing(true);
        try {
          const result = (await transcribeAudio(blob, token)) as {
            text: string;
          };
          return result.text || "";
        } finally {
          setTranscribing(false);
        }
      }

      await startRecording();
      return "";
    },
    [recording, startRecording, stopRecording]
  );

  return { recording, transcribing, error, recordAndTranscribe };
}
