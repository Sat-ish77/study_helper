"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { FileText, CreditCard, Target, TrendingUp, Loader2, RefreshCw } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { getDashboardStats, getStudyPlan } from "@/lib/api";
import type { DashboardStats, StudyPlan, RecentScore, StudySession } from "@/types";
import { DashboardGreeting } from "@/components/dashboard/DashboardGreeting";

function AnimatedNumber({ value }: { value: number }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          const start = performance.now();
          const step = (now: number) => {
            const progress = Math.min((now - start) / 2000, 1);
            const eased = 1 - Math.pow(1 - progress, 4);
            setDisplay(Math.round(eased * value));
            if (progress < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }
      },
      { threshold: 0.3 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [value]);

  return <span ref={ref}>{display}</span>;
}

function MetricCard({
  icon: Icon,
  label,
  value,
  sub,
  delay,
}: {
  icon: typeof FileText;
  label: string;
  value: string | number;
  sub?: string;
  delay: number;
}) {
  const numericValue = typeof value === "number" ? value : parseInt(value, 10);
  const isNumeric = !isNaN(numericValue) && typeof value === "number";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="p-5 rounded-xl card card-glow"
    >
      <div className="flex items-center gap-2 mb-3">
        <Icon size={18} style={{ color: "var(--accent)" }} />
        <span className="text-xs text-text-tertiary uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-3xl font-semibold text-text-primary font-sans">
        {isNumeric ? <AnimatedNumber value={numericValue} /> : value}
      </p>
      {sub && <p className="text-xs mt-1" style={{ color: "var(--accent)" }}>{sub}</p>}
    </motion.div>
  );
}

export default function DashboardPage() {
  const { token } = useAppStore();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [studyPlan, setStudyPlan] = useState<StudyPlan | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, [token]);

  const fetchData = async () => {
    if (!token) return;
    setLoading(true);

    try {
      const statsData = await getDashboardStats(token) as any;
      setStats(statsData);

      const planData = await getStudyPlan(token) as any;
      setStudyPlan(planData.plan);
    } catch (err) {
      console.error("Failed to fetch dashboard data:", err);
    } finally {
      setLoading(false);
    }
  };

  const getGradeColor = (percentage: number) => {
    if (percentage >= 70) return "bg-status-success/20 text-status-success";
    if (percentage >= 50) return "bg-status-warning/20 text-status-warning";
    return "bg-status-error/20 text-status-error";
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={32} className="animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <DashboardGreeting />
        <button
          onClick={fetchData}
          className="p-2 rounded-lg hover:bg-white/5 text-text-secondary hover:text-text-primary transition-colors"
        >
          <RefreshCw size={18} />
        </button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          icon={FileText}
          label="Documents"
          value={stats?.documents || 0}
          delay={0}
        />
        <MetricCard
          icon={CreditCard}
          label="Cards Due"
          value={stats?.flashcards_due || 0}
          sub="Review them today"
          delay={0.1}
        />
        <MetricCard
          icon={Target}
          label="Quiz Avg"
          value={`${stats?.quiz_avg_percentage || 0}%`}
          delay={0.2}
        />
        <MetricCard
          icon={TrendingUp}
          label="Quiz Attempts"
          value={stats?.quiz_attempts || 0}
          delay={0.3}
        />
      </div>

      {/* Study Plan */}
      {studyPlan && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card mb-8 border-accent/20"
        >
          <h2 className="text-lg font-serif text-text-primary mb-4">{studyPlan.greeting}</h2>
          <div className="space-y-3 mb-4">
            {studyPlan.sessions.map((session: StudySession, i: number) => (
              <div key={i} className="flex items-center gap-4">
                <span className="px-2 py-1 rounded bg-accent/10 text-accent text-xs font-medium">
                  {session.time}
                </span>
                <span className="text-text-primary">{session.activity}</span>
                <span className="text-text-tertiary text-sm ml-auto">{session.reason}</span>
              </div>
            ))}
          </div>
          <p className="text-sm text-text-secondary italic border-t border-border-subtle pt-3">
            💡 {studyPlan.tip}
          </p>
        </motion.div>
      )}

      {/* Recent Scores */}
      {stats?.recent_scores && stats.recent_scores.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-lg font-serif text-text-primary mb-4">Recent Quiz Scores</h2>
          <div className="space-y-2">
            {stats.recent_scores.map((score: RecentScore, index: number) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.05 }}
                className="flex items-center gap-4 p-3 rounded-xl bg-background-card border border-border-subtle"
              >
                <span className={`px-2 py-1 rounded text-xs font-medium ${getGradeColor(score.percentage)}`}>
                  {score.percentage}%
                </span>
                <span className="flex-1 text-text-primary">{score.topic}</span>
                <span className="text-text-secondary text-sm">
                  {score.score}/{score.total}
                </span>
                <span className="text-text-tertiary text-xs">
                  {new Date(score.created_at).toLocaleDateString()}
                </span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
