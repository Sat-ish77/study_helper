"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Shield, Activity, Database, AlertTriangle, Volume2,
  RefreshCw, Loader2, CheckCircle, XCircle, Cpu,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { checkHealth, AVAILABLE_MODELS } from "@/lib/api";

interface HealthStatus {
  status: string;
  version?: string;
  uptime?: number;
  [key: string]: any;
}

export default function AdminPage() {
  const { token } = useAppStore();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const fetchHealth = async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await checkHealth() as any;
      setHealth(data);
      setLastChecked(new Date());
    } catch (err: any) {
      setHealthError(err.message || "Failed to reach backend");
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  const isHealthy = health?.status === "ok" || health?.status === "healthy";

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <Shield size={24} className="text-accent" />
        <h1 className="text-2xl font-serif text-text-primary">Admin Panel</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Backend Health */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider flex items-center gap-2">
              <Activity size={14} />
              Backend Health
            </h2>
            <motion.button
              onClick={fetchHealth}
              disabled={healthLoading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-1.5 rounded-lg hover:bg-white/5 text-text-tertiary hover:text-text-primary transition-colors"
            >
              {healthLoading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <RefreshCw size={14} />
              )}
            </motion.button>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              {healthError ? (
                <XCircle size={20} className="text-status-error" />
              ) : isHealthy ? (
                <CheckCircle size={20} className="text-status-success" />
              ) : (
                <AlertTriangle size={20} className="text-status-warning" />
              )}
              <div>
                <p className="text-text-primary font-medium">
                  {healthError ? "Unreachable" : isHealthy ? "Healthy" : "Unknown"}
                </p>
                <p className="text-[11px] text-text-tertiary">
                  {healthError || (health?.status ?? "Checking...")}
                </p>
              </div>
            </div>

            {health && (
              <div className="space-y-1.5 pt-2 border-t border-border-subtle">
                {health.version && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-tertiary">Version</span>
                    <span className="text-text-secondary">{health.version}</span>
                  </div>
                )}
                {health.uptime !== undefined && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-tertiary">Uptime</span>
                    <span className="text-text-secondary">
                      {Math.round(health.uptime / 60)} min
                    </span>
                  </div>
                )}
                {Object.entries(health)
                  .filter(([k]) => !["status", "version", "uptime"].includes(k))
                  .slice(0, 6)
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-text-tertiary">{key}</span>
                      <span className="text-text-secondary truncate max-w-[200px]">
                        {typeof value === "object" ? JSON.stringify(value) : String(value)}
                      </span>
                    </div>
                  ))}
              </div>
            )}

            {lastChecked && (
              <p className="text-[10px] text-text-tertiary pt-1">
                Last checked: {lastChecked.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>

        {/* Available Models */}
        <div className="card">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider flex items-center gap-2 mb-4">
            <Cpu size={14} />
            Available Models
          </h2>
          <div className="space-y-2">
            {AVAILABLE_MODELS.map((model) => (
              <div
                key={model.value}
                className="flex items-center justify-between px-3 py-2 rounded-lg bg-background-secondary/50"
              >
                <div>
                  <p className="text-sm text-text-primary">{model.label}</p>
                  <p className="text-[10px] text-text-tertiary">{model.provider}</p>
                </div>
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full ${
                    model.badge === "Free"
                      ? "bg-status-success/10 text-status-success"
                      : "bg-accent/10 text-accent"
                  }`}
                >
                  {model.badge}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* TTS Status */}
        <div className="card">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider flex items-center gap-2 mb-4">
            <Volume2 size={14} />
            TTS Status
          </h2>
          <div className="space-y-2">
            <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-background-secondary/50">
              <span className="text-sm text-text-secondary">Speech Synthesis</span>
              <span className="flex items-center gap-1 text-xs text-status-success">
                <CheckCircle size={12} />
                Available
              </span>
            </div>
            <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-background-secondary/50">
              <span className="text-sm text-text-secondary">Transcription</span>
              <span className="flex items-center gap-1 text-xs text-status-success">
                <CheckCircle size={12} />
                Available
              </span>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="card">
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider flex items-center gap-2 mb-4">
            <Database size={14} />
            System Info
          </h2>
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">Frontend</span>
              <span className="text-text-secondary">Next.js 14</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">Backend</span>
              <span className="text-text-secondary">FastAPI + Python</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">Database</span>
              <span className="text-text-secondary">Supabase (PostgreSQL)</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">Vector Store</span>
              <span className="text-text-secondary">ChromaDB</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">Auth</span>
              <span className="text-text-secondary">Supabase Auth</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-text-tertiary">API URL</span>
              <span className="text-text-secondary truncate max-w-[200px]">
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
