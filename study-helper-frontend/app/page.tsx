"use client";

import { useEffect, useRef, lazy, Suspense } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import {
  BookOpen, Brain, BarChart2, ArrowRight, Sparkles, Check,
  Upload, MessageSquare, Mic, Image as ImageIcon, Shield,
} from "lucide-react";
import Link from "next/link";
import { prepare, layout } from "@chenglou/pretext";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { MagneticButton } from "@/components/ui/MagneticButton";

const WebGLBackground = lazy(() =>
  import("@/components/landing/WebGLBackground").then((m) => ({ default: m.WebGLBackground }))
);

/* ── Data ──────────────────────────────────────────── */

const FEATURES = [
  {
    icon: BookOpen,
    title: "Ask Your Docs",
    description: "Upload PDFs, DOCX, or PPTX and ask questions in natural language. Get answers from YOUR content.",
  },
  {
    icon: Brain,
    title: "Smart Flashcards",
    description: "Auto-generate flashcards with spaced repetition. Never forget what you learned.",
  },
  {
    icon: BarChart2,
    title: "Track Progress",
    description: "Quiz scores, study streaks, and weak topics. Personalized study plans based on performance.",
  },
  {
    icon: Mic,
    title: "Voice Input & TTS",
    description: "Speak your questions, listen to answers. Full voice workflow for hands-free studying.",
  },
  {
    icon: ImageIcon,
    title: "Image Lab",
    description: "Generate charts, flowcharts, and DALL-E images from your study content.",
  },
  {
    icon: Shield,
    title: "Multi-Model AI",
    description: "Choose between GPT-4o, Llama 3.3, Gemini 2.0 Flash — use what works best for you.",
  },
];

const STEPS = [
  { num: "01", title: "Upload", desc: "Drag & drop your lecture notes, PDFs, or presentations", icon: Upload },
  { num: "02", title: "Ask", desc: "Chat with AI about your content. Voice input supported.", icon: MessageSquare },
  { num: "03", title: "Master", desc: "Quizzes, flashcards, and progress tracking — all in one.", icon: Brain },
];

const HERO_WORDS_L1 = ["Study", "Smarter."];
const HERO_WORDS_L2 = ["Not", "Harder."];

const CHAT_MESSAGES = [
  { role: "user", text: "What is photosynthesis?" },
  { role: "assistant", text: "Photosynthesis is the process by which green plants convert light energy into chemical energy, producing glucose and oxygen from CO₂ and water." },
  { role: "user", text: "Explain it simpler" },
  { role: "assistant", text: "Plants eat sunlight and drink water to make their own food. They breathe out oxygen as a bonus!" },
];

/* ── Word-by-word animated headline ────────────────── */

function AnimatedWord({ word, index, line2 }: { word: string; index: number; line2?: boolean }) {
  return (
    <motion.span
      className={`inline-block mr-[0.3em] ${line2 ? "shimmer-text" : ""}`}
      initial={{ opacity: 0, y: 40, rotateX: -40 }}
      animate={{ opacity: 1, y: 0, rotateX: 0 }}
      transition={{
        duration: 0.8,
        delay: 0.3 + index * 0.12,
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      {word}
    </motion.span>
  );
}

/* ── Floating Chat Preview ─────────────────────────── */

function ChatPreview() {
  const bubbleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Use pretext to measure the longest assistant message
    // and dynamically size the preview container
    try {
      const longestMsg = CHAT_MESSAGES
        .filter((m) => m.role === "assistant")
        .reduce((a, b) => (a.text.length > b.text.length ? a : b));
      const prepared = prepare(longestMsg.text, "13px DM Sans");
      const measured = layout(prepared, 280, 18);
      if (bubbleRef.current && measured.height) {
        // Use measurement to set min-height so container doesn't jump
        bubbleRef.current.style.minHeight = `${measured.height + 24}px`;
      }
    } catch {
      // pretext measurement optional — graceful fallback
    }
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="max-w-sm mx-auto"
    >
      <div className="glass rounded-2xl p-4 space-y-3 shadow-2xl">
        <div className="flex items-center gap-2 mb-3 pb-2" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
          <div className="w-2 h-2 rounded-full" style={{ background: "var(--accent)" }} />
          <span className="text-[11px] font-medium" style={{ color: "var(--text-secondary)" }}>Study Helper Chat</span>
        </div>
        {CHAT_MESSAGES.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: msg.role === "user" ? 20 : -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 + i * 0.2 }}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              ref={msg.role === "assistant" && i === 1 ? bubbleRef : undefined}
              className="max-w-[280px] px-3 py-2 rounded-xl text-[13px]"
              style={{
                background: msg.role === "user"
                  ? "rgba(var(--accent-rgb), 0.15)"
                  : "var(--glass-bg)",
                border: `1px solid ${msg.role === "user" ? "rgba(var(--accent-rgb), 0.2)" : "var(--glass-border)"}`,
                color: "var(--text-primary)",
              }}
            >
              {msg.text}
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

/* ══════════════════════════════════════════════════════
   LANDING PAGE
   ══════════════════════════════════════════════════════ */

export default function LandingPage() {
  const heroRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroY = useTransform(scrollYProgress, [0, 1], [0, 150]);
  const orbY = useTransform(scrollYProgress, [0, 1], [0, 80]);
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.92]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  useScrollReveal();

  return (
    <div className="min-h-screen overflow-hidden relative" style={{ background: "var(--bg-primary)" }}>
      {/* ── Navigation ─────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <Link href="/" className="font-serif text-xl text-gradient-amber">
            Study Helper
          </Link>
          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-sm transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              Sign In
            </Link>
            <MagneticButton
              className="px-4 py-2 rounded-lg text-sm font-medium"
              onClick={() => (window.location.href = "/signup")}
            >
              Get Started
            </MagneticButton>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ───────────────────────────── */}
      <section ref={heroRef} className="relative pt-32 pb-24 px-6 min-h-screen flex items-center justify-center">
        {/* Three.js WebGL background — lazy loaded */}
        <Suspense fallback={null}>
          <WebGLBackground />
        </Suspense>

        {/* Parallax Gradient Orbs */}
        <motion.div
          style={{ y: orbY }}
          className="absolute top-10 left-1/4 w-[500px] h-[500px] rounded-full blur-[160px] pointer-events-none"
          aria-hidden="true"
        >
          <div className="w-full h-full rounded-full" style={{ background: "rgba(var(--accent-rgb), 0.15)" }} />
        </motion.div>
        <motion.div
          style={{ y: useTransform(scrollYProgress, [0, 1], [0, 50]) }}
          className="absolute bottom-20 right-1/4 w-80 h-80 rounded-full blur-[120px] pointer-events-none"
          aria-hidden="true"
        >
          <div className="w-full h-full rounded-full" style={{ background: "rgba(var(--accent-rgb), 0.08)" }} />
        </motion.div>

        <motion.div
          style={{ y: heroY, scale: heroScale, opacity: heroOpacity }}
          className="max-w-4xl mx-auto text-center relative z-10"
        >
          <motion.span
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-medium mb-8"
            style={{ color: "var(--accent)" }}
          >
            <Sparkles size={14} />
            Powered by RAG + Multiple LLMs
          </motion.span>

          {/* Word-by-word animated headline with CSS shimmer */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-serif leading-tight mb-6" style={{ perspective: "600px" }}>
            <span className="block" style={{ color: "var(--text-primary)" }}>
              {HERO_WORDS_L1.map((word, i) => (
                <AnimatedWord key={word} word={word} index={i} />
              ))}
            </span>
            <span className="block">
              {HERO_WORDS_L2.map((word, i) => (
                <AnimatedWord key={word} word={word} index={i + HERO_WORDS_L1.length} line2 />
              ))}
            </span>
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-lg sm:text-xl max-w-2xl mx-auto mb-10"
            style={{ color: "var(--text-secondary)" }}
          >
            Upload your notes. Ask anything. Get instant answers from YOUR documents with
            AI-powered study tools.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.0 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link href="/signup">
              <MagneticButton className="group px-8 py-4 rounded-xl font-medium flex items-center gap-2 text-base btn-amber">
                Start Free
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </MagneticButton>
            </Link>
            <Link
              href="/login"
              className="px-8 py-4 rounded-xl glass font-medium transition-all flex items-center gap-2"
              style={{ color: "var(--text-primary)" }}
            >
              Sign In
            </Link>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.3 }}
            className="text-sm mt-8"
            style={{ color: "var(--text-tertiary)" }}
          >
            No credit card required &nbsp;&bull;&nbsp; Works with any document
          </motion.p>
        </motion.div>
      </section>

      {/* ── Features Section ───────────────────────── */}
      <section className="py-24 px-6 relative z-10" style={{ borderTop: "1px solid var(--border-subtle)" }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 reveal">
            <h2 className="text-3xl sm:text-4xl font-serif mb-4" style={{ color: "var(--text-primary)" }}>
              Everything you need to ace your exams
            </h2>
            <p style={{ color: "var(--text-secondary)" }}>
              Built for students, by students. Trusted by UNT Computer Science.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((feature, i) => (
              <div
                key={i}
                className={`reveal reveal-delay-${(i % 4) + 1} card card-glow p-6 rounded-2xl`}
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: "rgba(var(--accent-rgb), 0.1)" }}
                >
                  <feature.icon size={24} style={{ color: "var(--accent)" }} />
                </div>
                <h3 className="text-lg font-medium mb-2" style={{ color: "var(--text-primary)" }}>
                  {feature.title}
                </h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ───────────────────────────── */}
      <section className="py-24 px-6 relative z-10" style={{ background: "var(--bg-secondary)" }}>
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16 reveal">
            <h2 className="text-3xl sm:text-4xl font-serif mb-4" style={{ color: "var(--text-primary)" }}>
              How It Works
            </h2>
            <p style={{ color: "var(--text-secondary)" }}>Get started in less than 2 minutes</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Animated connectors */}
            <div className="hidden md:block absolute top-12 left-[20%] right-[20%] h-px" style={{ background: "var(--border-subtle)" }}>
              <motion.div
                className="h-full"
                style={{ background: "var(--accent)", transformOrigin: "left" }}
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 1.2, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>

            {STEPS.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 + i * 0.2, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                className="text-center relative"
              >
                <div
                  className="w-16 h-16 rounded-2xl glass flex items-center justify-center mx-auto mb-4"
                >
                  <step.icon size={28} style={{ color: "var(--accent)" }} />
                </div>
                <span className="text-3xl font-serif" style={{ color: "rgba(var(--accent-rgb), 0.2)" }}>
                  {step.num}
                </span>
                <h3 className="text-lg font-medium mt-2 mb-2" style={{ color: "var(--text-primary)" }}>
                  {step.title}
                </h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Chat Demo Preview ──────────────────────── */}
      <section className="py-24 px-6 relative z-10">
        <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="reveal">
            <h2 className="text-3xl sm:text-4xl font-serif mb-4" style={{ color: "var(--text-primary)" }}>
              Ask anything about your notes
            </h2>
            <p className="mb-6" style={{ color: "var(--text-secondary)" }}>
              Upload a document, then chat with AI that understands your content.
              Get simplified explanations, translations, flashcards — all from one conversation.
            </p>
            <ul className="space-y-3">
              {["Voice input & text-to-speech", "6 quick actions per message", "Translate to 10+ languages", "Save answers as flashcards"].map((item, i) => (
                <li key={i} className="flex items-center gap-2 text-sm" style={{ color: "var(--text-secondary)" }}>
                  <Check size={16} style={{ color: "var(--accent)" }} />
                  {item}
                </li>
              ))}
            </ul>
          </div>
          <div className="reveal reveal-delay-2">
            <ChatPreview />
          </div>
        </div>
      </section>

      {/* ── Pricing ────────────────────────────────── */}
      <section className="py-24 px-6 relative z-10" style={{ background: "var(--bg-secondary)" }}>
        <div className="max-w-4xl mx-auto text-center">
          <div className="reveal">
            <h2 className="text-3xl sm:text-4xl font-serif mb-4" style={{ color: "var(--text-primary)" }}>
              Simple Pricing
            </h2>
            <p className="mb-12" style={{ color: "var(--text-secondary)" }}>
              Student-friendly. No credit card required to start.
            </p>
          </div>

          <div className="reveal reveal-delay-1 max-w-sm mx-auto p-8 rounded-2xl glass card-glow" style={{ border: "1px solid rgba(var(--accent-rgb), 0.2)" }}>
            <p className="text-sm uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>Full Access</p>
            <div className="flex items-baseline justify-center gap-1 mb-6">
              <span className="text-5xl font-bold" style={{ color: "var(--text-primary)" }}>$4</span>
              <span style={{ color: "var(--text-secondary)" }}>/month</span>
            </div>

            <ul className="space-y-3 text-left mb-8">
              {[
                "Unlimited chat messages",
                "Upload unlimited documents",
                "Voice input & TTS",
                "Smart flashcards with SRS",
                "Canvas calendar sync",
                "Quiz generation & tracking",
                "Multiple AI models",
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-2 text-sm" style={{ color: "var(--text-secondary)" }}>
                  <Check size={16} style={{ color: "var(--accent)" }} />
                  {item}
                </li>
              ))}
            </ul>

            <Link href="/signup">
              <MagneticButton className="w-full py-3 rounded-xl font-medium btn-amber text-base">
                Start Free Trial
              </MagneticButton>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────── */}
      <footer className="py-12 px-6 relative z-10" style={{ borderTop: "1px solid var(--border-subtle)" }}>
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <p className="font-serif text-lg text-gradient-amber">Study Helper</p>
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Built by Satish Wagle at UNT
            </p>
          </div>
          <div className="flex items-center gap-6 text-sm" style={{ color: "var(--text-secondary)" }}>
            <Link href="/login" className="hover:opacity-80 transition-opacity">Sign In</Link>
            <Link href="/signup" className="hover:opacity-80 transition-opacity">Get Started</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
