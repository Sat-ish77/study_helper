"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, PlayCircle, Globe, ExternalLink, Loader2, ChevronDown } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { getLearnMore } from "@/lib/api";

interface LearnMorePanelProps {
  content: string;
}

interface LearnMoreResult {
  youtube?: { title: string; url: string; thumbnail?: string }[];
  wikipedia?: { title: string; url: string; snippet?: string }[];
  related_topics?: string[];
}

export function LearnMorePanel({ content }: LearnMorePanelProps) {
  const { token } = useAppStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<LearnMoreResult | null>(null);

  const handleOpen = async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (data) return;

    if (!token) return;
    setLoading(true);
    try {
      const topic = content.slice(0, 200).replace(/\n/g, " ");
      const result = await getLearnMore(topic, token) as any;
      setData({
        youtube: result.youtube || result.videos || [],
        wikipedia: result.wikipedia || result.wiki || [],
        related_topics: result.related_topics || [],
      });
    } catch (err) {
      console.error("Learn more failed:", err);
      setData({ youtube: [], wikipedia: [], related_topics: [] });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-1.5">
      <button
        onClick={handleOpen}
        className="flex items-center gap-1 text-[11px] text-text-tertiary hover:text-text-secondary transition-colors"
      >
        <BookOpen size={12} />
        <span>Learn More</span>
        <ChevronDown
          size={12}
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-2 p-3 rounded-xl bg-background-secondary/50 border border-border-subtle space-y-3">
              {loading ? (
                <div className="flex items-center gap-2 text-text-tertiary text-xs">
                  <Loader2 size={14} className="animate-spin" />
                  Finding resources...
                </div>
              ) : (
                <>
                  {/* YouTube */}
                  {data?.youtube && data.youtube.length > 0 && (
                    <div>
                      <p className="text-[10px] text-text-tertiary uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <PlayCircle size={11} className="text-red-400" />
                        Videos
                      </p>
                      <div className="space-y-1.5">
                        {data.youtube.slice(0, 3).map((v, i) => (
                          <a
                            key={i}
                            href={v.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/[0.04] transition-colors group"
                          >
                            <PlayCircle size={14} className="text-red-400 shrink-0" />
                            <span className="text-xs text-text-secondary group-hover:text-text-primary truncate">
                              {v.title}
                            </span>
                            <ExternalLink size={10} className="text-text-tertiary ml-auto shrink-0" />
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Wikipedia */}
                  {data?.wikipedia && data.wikipedia.length > 0 && (
                    <div>
                      <p className="text-[10px] text-text-tertiary uppercase tracking-wider mb-1.5 flex items-center gap-1">
                        <Globe size={11} />
                        Wikipedia
                      </p>
                      <div className="space-y-1.5">
                        {data.wikipedia.slice(0, 3).map((w, i) => (
                          <a
                            key={i}
                            href={w.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-start gap-2 px-2 py-1.5 rounded-lg hover:bg-white/[0.04] transition-colors group"
                          >
                            <Globe size={14} className="text-accent shrink-0 mt-0.5" />
                            <div className="min-w-0">
                              <span className="text-xs text-text-secondary group-hover:text-text-primary block truncate">
                                {w.title}
                              </span>
                              {w.snippet && (
                                <span className="text-[10px] text-text-tertiary line-clamp-2">
                                  {w.snippet}
                                </span>
                              )}
                            </div>
                            <ExternalLink size={10} className="text-text-tertiary ml-auto shrink-0 mt-0.5" />
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* No results */}
                  {(!data?.youtube?.length && !data?.wikipedia?.length) && (
                    <p className="text-xs text-text-tertiary">No additional resources found for this topic.</p>
                  )}
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
