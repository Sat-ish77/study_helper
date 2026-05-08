"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  MessageSquare,
  Beaker,
  CreditCard,
  Upload,
  BarChart2,
  Calendar,
  Image,
  Shield,
  LogOut,
  Plus,
  Trash2,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { createClient } from "@/lib/supabase-client";
import { getConversations, deleteConversation } from "@/lib/api";
import type { Conversation } from "@/types";

const NAV_ITEMS = [
  { href: "/chat", icon: MessageSquare, label: "Chat" },
  { href: "/quiz", icon: Beaker, label: "Quiz Lab" },
  { href: "/flashcards", icon: CreditCard, label: "Flashcards" },
  { href: "/upload", icon: Upload, label: "Upload Docs" },
  { href: "/dashboard", icon: BarChart2, label: "Dashboard" },
  { href: "/calendar", icon: Calendar, label: "Calendar" },
  { href: "/images", icon: Image, label: "Image Lab" },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { token, userEmail, logout, setConversations, conversations, setCurrentConversation, sidebarCollapsed, setSidebarCollapsed } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // Collapse sidebar when user scrolls down
  useEffect(() => {
    if (isMobile) return;
    let lastY = 0;
    const onScroll = () => {
      const y = window.scrollY;
      if (y > 100 && y > lastY) setSidebarCollapsed(true);
      if (y < 50) setSidebarCollapsed(false);
      lastY = y;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [isMobile, setSidebarCollapsed]);

  const isOpen = !sidebarCollapsed || hovered;
  const width = isOpen ? 240 : 60;

  useEffect(() => {
    if (!token) return;
    fetchConversations();
  }, [token]);

  const fetchConversations = async () => {
    if (!token) return;
    try {
      const data = await getConversations(token);
      setConversations(data.conversations || []);
    } catch (err) {
      console.error("Failed to fetch conversations:", err);
    }
  };

  const handleLogout = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    logout();
    router.push("/login");
  };

  const handleNewChat = () => {
    setCurrentConversation(null);
    router.push("/chat");
  };

  const handleDeleteConversation = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!token) return;
    try {
      await deleteConversation(id, token);
      fetchConversations();
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  };

  // Mobile bottom nav
  if (isMobile) {
    return (
      <nav className="fixed bottom-0 left-0 right-0 h-16 glass z-50 flex items-center justify-around px-2" style={{ borderTop: "1px solid var(--border-subtle)" }}>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <a
              key={item.href}
              href={item.href}
              className={`flex flex-col items-center gap-1 p-2 rounded-lg transition-colors ${
                isActive ? "text-accent" : "text-text-tertiary"
              }`}
            >
              <item.icon size={20} />
              <span className="text-[10px]">{item.label.split(" ")[0]}</span>
            </a>
          );
        })}
      </nav>
    );
  }

  return (
    <aside
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="h-screen glass flex flex-col fixed left-0 top-0 z-40 overflow-hidden"
      style={{
        width,
        borderRight: "1px solid var(--border-subtle)",
        background: "var(--bg-secondary)",
        transition: "width 0.3s cubic-bezier(0.76, 0, 0.24, 1)",
      }}
    >
      {/* Logo */}
      <div className="p-4 border-b border-border-subtle h-14 flex items-center">
        {isOpen ? (
          <h1 className="font-serif text-xl text-gradient-amber whitespace-nowrap">Study Helper</h1>
        ) : (
          <span className="font-serif text-2xl text-gradient-amber mx-auto">S</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="p-2 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <motion.a
              key={item.href}
              href={item.href}
              whileHover={{ x: 2 }}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150
                ${isActive
                  ? "text-accent border-l-2 border-accent"
                  : "text-text-secondary hover:text-text-primary"
                }
                ${!isOpen ? "justify-center px-2" : ""}
              `}
              style={{ background: isActive ? "rgba(var(--accent-rgb), 0.08)" : "transparent" }}
              title={!isOpen ? item.label : undefined}
            >
              <item.icon size={18} />
              {isOpen && <span className="whitespace-nowrap">{item.label}</span>}
            </motion.a>
          );
        })}
      </nav>

      {/* Conversations — hide when collapsed */}
      {isOpen && (
        <div className="flex-1 overflow-hidden flex flex-col border-t border-border-subtle mt-2">
          <div className="flex items-center justify-between px-3 py-2">
            <span className="text-[11px] text-text-tertiary uppercase tracking-wider whitespace-nowrap">Recent Chats</span>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleNewChat}
              className="p-1 rounded hover:bg-white/5 text-text-tertiary hover:text-text-primary"
            >
              <Plus size={14} />
            </motion.button>
          </div>
          <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
            {conversations.slice(0, 10).map((conv) => (
              <motion.div
                key={conv.id}
                onClick={() => {
                  setCurrentConversation(conv.id);
                  router.push("/chat");
                }}
                className="group relative flex items-center px-2 py-1.5 rounded-lg cursor-pointer hover:bg-white/[0.04] transition-colors"
              >
                <span className="flex-1 truncate text-sm text-text-secondary group-hover:text-text-primary transition-colors">
                  {conv.title}
                </span>
                <motion.button
                  onClick={(e) => handleDeleteConversation(e, conv.id)}
                  className="opacity-0 group-hover:opacity-100 text-text-tertiary hover:text-status-error text-xs ml-1 p-1"
                  whileHover={{ scale: 1.1 }}
                >
                  <Trash2 size={12} />
                </motion.button>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Admin */}
      {isOpen && (
        <div className="px-3 pb-1">
          <motion.a
            href="/admin"
            whileHover={{ x: 2 }}
            className={`
              flex items-center gap-3 px-3 py-1.5 rounded-lg text-xs transition-all duration-150
              ${pathname === "/admin"
                ? "text-accent bg-accent/8"
                : "text-text-tertiary hover:text-text-secondary"
              }
            `}
          >
            <Shield size={14} />
            <span className="whitespace-nowrap">Admin</span>
          </motion.a>
        </div>
      )}

      {/* User Section */}
      <div className="p-3 border-t border-border-subtle">
        <div className={`flex items-center gap-3 ${!isOpen ? "justify-center" : ""}`}>
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium shrink-0" style={{ background: "rgba(var(--accent-rgb), 0.2)", color: "var(--accent)" }}>
            {userEmail?.charAt(0).toUpperCase() || "?"}
          </div>
          {isOpen && (
            <>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-text-primary truncate">{userEmail || "Loading..."}</p>
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleLogout}
                className="p-1.5 rounded-lg hover:bg-white/5 text-text-tertiary hover:text-text-primary"
              >
                <LogOut size={16} />
              </motion.button>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
