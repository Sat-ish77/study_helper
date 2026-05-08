"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Calendar as CalendarIcon, Link2, Trash2, Loader2, ExternalLink } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { saveCanvasUrl, getCanvasUrl, clearCanvasUrl, getCanvasEvents } from "@/lib/api";
import type { CanvasEvent } from "@/types";

export default function CalendarPage() {
  const { token } = useAppStore();
  const [url, setUrl] = useState("");
  const [savedUrl, setSavedUrl] = useState<string | null>(null);
  const [events, setEvents] = useState<CanvasEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    fetchData();
  }, [token]);

  const fetchData = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [urlData, eventsData] = await Promise.all([
        getCanvasUrl(token),
        getCanvasEvents(token),
      ]) as [any, any];
      setSavedUrl(urlData.url);
      setEvents(eventsData.events || []);
    } catch (err) {
      console.error("Failed to fetch calendar data:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!token || !url.trim()) return;
    setSaving(true);
    try {
      await saveCanvasUrl(url.trim(), token);
      setUrl("");
      fetchData();
    } catch (err) {
      console.error("Failed to save URL:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    if (!token) return;
    try {
      await clearCanvasUrl(token);
      fetchData();
    } catch (err) {
      console.error("Failed to clear URL:", err);
    }
  };

  const getUrgencyColor = (startDate: string) => {
    const days = Math.ceil((new Date(startDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24));
    if (days < 1) return "border-status-error/50 bg-status-error/5";
    if (days < 7) return "border-status-warning/50 bg-status-warning/5";
    return "border-border-subtle";
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  };

  return (
    <div>
      <h1 className="text-2xl font-serif text-text-primary mb-6">Calendar</h1>

      {/* URL Form */}
      <div className="card mb-8">
        <label className="block text-sm text-text-secondary mb-2">
          Canvas Calendar iCal URL
        </label>
        <div className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://canvas.instructure.com/feeds/calendars/..."
            className="input flex-1"
          />
          <motion.button
            onClick={handleSave}
            disabled={saving || !url.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="btn-amber disabled:opacity-50"
          >
            {saving ? <Loader2 size={18} className="animate-spin" /> : <Link2 size={18} />}
          </motion.button>
        </div>

        {savedUrl && (
          <div className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-accent/5 border border-accent/10">
            <Link2 size={16} className="text-accent" />
            <span className="text-sm text-text-secondary truncate flex-1">{savedUrl}</span>
            <button
              onClick={handleClear}
              className="p-1.5 rounded-lg hover:bg-white/5 text-text-tertiary hover:text-status-error transition-colors"
            >
              <Trash2 size={16} />
            </button>
          </div>
        )}

        <p className="mt-4 text-xs text-text-tertiary">
          Find your iCal URL in Canvas: Calendar → Calendar Feed (bottom left)
        </p>
      </div>

      {/* Events List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={32} className="animate-spin text-accent" />
        </div>
      ) : events.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          <CalendarIcon size={48} className="mx-auto mb-4 text-accent/40" />
          <p>No upcoming events.</p>
        </div>
      ) : (
        <div>
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-4">
            Upcoming Events ({events.length})
          </h2>
          <div className="space-y-3">
            {events.map((event, index) => (
              <motion.div
                key={event.uid}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`p-4 rounded-xl bg-background-card border ${getUrgencyColor(event.start)} transition-colors`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="font-medium text-text-primary">{event.title}</h3>
                    <p className="text-sm text-text-secondary mt-1">{formatDate(event.start)}</p>
                    {event.location && (
                      <p className="text-xs text-text-tertiary mt-1">📍 {event.location}</p>
                    )}
                    {event.description && (
                      <p className="text-xs text-text-tertiary mt-2 line-clamp-2">{event.description}</p>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
