"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Image as ImageIcon, Sparkles, BarChart3, GitBranch,
  Loader2, Download, RefreshCw,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import {
  generateDalleImage, generateChart, generateFlowchart,
} from "@/lib/api";

type Tab = "dalle" | "chart" | "flowchart";

const TABS: { id: Tab; label: string; icon: typeof ImageIcon }[] = [
  { id: "dalle", label: "DALL-E", icon: Sparkles },
  { id: "chart", label: "Charts", icon: BarChart3 },
  { id: "flowchart", label: "Flowcharts", icon: GitBranch },
];

const CHART_TYPES = ["bar", "line", "pie", "scatter"] as const;

export default function ImageLabPage() {
  const { token } = useAppStore();
  const [activeTab, setActiveTab] = useState<Tab>("dalle");
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [usageCount, setUsageCount] = useState(0);

  // DALL-E state
  const [dallePrompt, setDallePrompt] = useState("");

  // Chart state
  const [chartType, setChartType] = useState<string>("bar");
  const [chartTitle, setChartTitle] = useState("");
  const [chartLabels, setChartLabels] = useState("");
  const [chartValues, setChartValues] = useState("");
  const [chartXLabel, setChartXLabel] = useState("");
  const [chartYLabel, setChartYLabel] = useState("");

  // Flowchart state
  const [flowTitle, setFlowTitle] = useState("");
  const [flowSteps, setFlowSteps] = useState("");

  const handleDalle = async () => {
    if (!token || !dallePrompt.trim()) return;
    setLoading(true);
    setImageUrl(null);
    try {
      const data = await generateDalleImage(dallePrompt.trim(), token) as any;
      setImageUrl(data.image_url || data.url || data.image);
      setUsageCount((c) => c + 1);
    } catch (err) {
      console.error("DALL-E generation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleChart = async () => {
    if (!token || !chartTitle.trim()) return;
    setLoading(true);
    setImageUrl(null);
    try {
      const labels = chartLabels.split(",").map((s) => s.trim()).filter(Boolean);
      const values = chartValues.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
      const data = await generateChart(
        {
          chart_type: chartType,
          title: chartTitle.trim(),
          labels,
          values,
          xlabel: chartXLabel || undefined,
          ylabel: chartYLabel || undefined,
        },
        token
      ) as any;
      setImageUrl(data.image_url || data.url || data.image);
      setUsageCount((c) => c + 1);
    } catch (err) {
      console.error("Chart generation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFlowchart = async () => {
    if (!token || !flowTitle.trim() || !flowSteps.trim()) return;
    setLoading(true);
    setImageUrl(null);
    try {
      const steps = flowSteps.split("\n").map((s) => s.trim()).filter(Boolean);
      const data = await generateFlowchart(
        { title: flowTitle.trim(), steps },
        token
      ) as any;
      setImageUrl(data.image_url || data.url || data.image);
      setUsageCount((c) => c + 1);
    } catch (err) {
      console.error("Flowchart generation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!imageUrl) return;
    const a = document.createElement("a");
    a.href = imageUrl;
    a.download = `study-helper-${activeTab}-${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-serif text-text-primary">Image Lab</h1>
        <span className="text-xs text-text-tertiary bg-background-secondary px-2 py-1 rounded-lg">
          {usageCount} generated this session
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-background-secondary rounded-xl mb-6">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => { setActiveTab(tab.id); setImageUrl(null); }}
            className={`
              flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm transition-all
              ${activeTab === tab.id
                ? "bg-accent text-background-primary font-medium"
                : "text-text-secondary hover:text-text-primary"
              }
            `}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="card space-y-4">
          {activeTab === "dalle" && (
            <>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Prompt</label>
                <textarea
                  value={dallePrompt}
                  onChange={(e) => setDallePrompt(e.target.value)}
                  placeholder="A detailed diagram of the human circulatory system..."
                  rows={4}
                  className="input resize-none"
                />
              </div>
              <motion.button
                onClick={handleDalle}
                disabled={loading || !dallePrompt.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full btn-amber disabled:opacity-50"
              >
                {loading ? (
                  <><Loader2 size={16} className="animate-spin mr-2" /> Generating...</>
                ) : (
                  <><Sparkles size={16} className="mr-2" /> Generate Image</>
                )}
              </motion.button>
            </>
          )}

          {activeTab === "chart" && (
            <>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Chart Type</label>
                <div className="flex gap-2">
                  {CHART_TYPES.map((type) => (
                    <button
                      key={type}
                      onClick={() => setChartType(type)}
                      className={`
                        px-3 py-1.5 rounded-lg text-xs capitalize transition-all
                        ${chartType === type
                          ? "bg-accent text-background-primary"
                          : "bg-background-secondary text-text-secondary hover:text-text-primary"
                        }
                      `}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Title</label>
                <input
                  value={chartTitle}
                  onChange={(e) => setChartTitle(e.target.value)}
                  placeholder="Quiz Scores Over Time"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Labels (comma-separated)</label>
                <input
                  value={chartLabels}
                  onChange={(e) => setChartLabels(e.target.value)}
                  placeholder="Week 1, Week 2, Week 3, Week 4"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Values (comma-separated)</label>
                <input
                  value={chartValues}
                  onChange={(e) => setChartValues(e.target.value)}
                  placeholder="75, 82, 90, 95"
                  className="input"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">X-Axis Label</label>
                  <input
                    value={chartXLabel}
                    onChange={(e) => setChartXLabel(e.target.value)}
                    placeholder="Weeks"
                    className="input"
                  />
                </div>
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Y-Axis Label</label>
                  <input
                    value={chartYLabel}
                    onChange={(e) => setChartYLabel(e.target.value)}
                    placeholder="Score %"
                    className="input"
                  />
                </div>
              </div>
              <motion.button
                onClick={handleChart}
                disabled={loading || !chartTitle.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full btn-amber disabled:opacity-50"
              >
                {loading ? (
                  <><Loader2 size={16} className="animate-spin mr-2" /> Generating...</>
                ) : (
                  <><BarChart3 size={16} className="mr-2" /> Generate Chart</>
                )}
              </motion.button>
            </>
          )}

          {activeTab === "flowchart" && (
            <>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Title</label>
                <input
                  value={flowTitle}
                  onChange={(e) => setFlowTitle(e.target.value)}
                  placeholder="Photosynthesis Process"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm text-text-secondary mb-2">Steps (one per line)</label>
                <textarea
                  value={flowSteps}
                  onChange={(e) => setFlowSteps(e.target.value)}
                  placeholder={"Light hits chlorophyll\nWater molecules split\nOxygen released\nGlucose formed"}
                  rows={6}
                  className="input resize-none"
                />
              </div>
              <motion.button
                onClick={handleFlowchart}
                disabled={loading || !flowTitle.trim() || !flowSteps.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full btn-amber disabled:opacity-50"
              >
                {loading ? (
                  <><Loader2 size={16} className="animate-spin mr-2" /> Generating...</>
                ) : (
                  <><GitBranch size={16} className="mr-2" /> Generate Flowchart</>
                )}
              </motion.button>
            </>
          )}
        </div>

        {/* Preview Panel */}
        <div className="card flex flex-col items-center justify-center min-h-[400px]">
          {loading ? (
            <div className="flex flex-col items-center gap-3 text-text-tertiary">
              <Loader2 size={32} className="animate-spin text-accent" />
              <p className="text-sm">Generating your image...</p>
            </div>
          ) : imageUrl ? (
            <div className="w-full space-y-3">
              <img
                src={imageUrl}
                alt="Generated"
                className="w-full rounded-xl border border-border-subtle"
              />
              <div className="flex gap-2">
                <motion.button
                  onClick={handleDownload}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-border-subtle text-text-secondary hover:text-text-primary text-sm transition-colors"
                >
                  <Download size={14} />
                  Download
                </motion.button>
                <motion.button
                  onClick={() => setImageUrl(null)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-border-subtle text-text-secondary hover:text-text-primary text-sm transition-colors"
                >
                  <RefreshCw size={14} />
                </motion.button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 text-text-tertiary">
              <ImageIcon size={48} className="opacity-20" />
              <p className="text-sm">Your generated image will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
