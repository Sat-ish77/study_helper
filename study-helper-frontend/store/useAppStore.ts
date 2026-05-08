"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Conversation, Message } from "@/types";
import { DEFAULT_MODEL } from "@/lib/api";

export type ThemeId = "amber" | "cyan" | "emerald" | "violet";

interface AppState {
  token: string | null;
  userId: string | null;
  userEmail: string | null;
  selectedModel: string;
  selectedLanguage: string;
  theme: ThemeId;
  sidebarCollapsed: boolean;
  currentConversationId: string | null;
  conversations: Conversation[];
  messages: Message[];
}

interface AppActions {
  setToken: (token: string | null) => void;
  setUserId: (userId: string | null) => void;
  setUserEmail: (email: string | null) => void;
  setSelectedModel: (model: string) => void;
  setSelectedLanguage: (language: string) => void;
  setTheme: (theme: ThemeId) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setCurrentConversation: (id: string | null) => void;
  setConversations: (conversations: Conversation[]) => void;
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  clearMessages: () => void;
  logout: () => void;
}

const initialState: AppState = {
  token: null,
  userId: null,
  userEmail: null,
  selectedModel: DEFAULT_MODEL,
  selectedLanguage: "English",
  theme: "amber" as ThemeId,
  sidebarCollapsed: false,
  currentConversationId: null,
  conversations: [],
  messages: [],
};

export const useAppStore = create<AppState & AppActions>()(
  persist(
    (set) => ({
      ...initialState,

      setToken: (token) => set({ token }),
      setUserId: (userId) => set({ userId }),
      setUserEmail: (userEmail) => set({ userEmail }),
      setSelectedModel: (selectedModel) => set({ selectedModel }),
      setSelectedLanguage: (selectedLanguage) => set({ selectedLanguage }),
      setTheme: (theme) => {
        set({ theme });
        if (typeof document !== "undefined") {
          document.documentElement.setAttribute("data-theme", theme);
        }
      },
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
      setCurrentConversation: (currentConversationId) =>
        set({ currentConversationId }),
      setConversations: (conversations) => set({ conversations }),
      addMessage: (message) =>
        set((state) => ({ messages: [...state.messages, message] })),
      setMessages: (messages) => set({ messages }),
      clearMessages: () => set({ messages: [] }),
      logout: () => set({ ...initialState }),
    }),
    {
      name: "study-helper-store",
      partialize: (state) => ({
        token: state.token,
        userId: state.userId,
        userEmail: state.userEmail,
        selectedModel: state.selectedModel,
        selectedLanguage: state.selectedLanguage,
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);
