"""
image_generator.py — Study Helper v2
DALL-E 3 image generation + matplotlib diagrams.
No graphviz dependency — Streamlit Cloud safe.
"""
from __future__ import annotations

import io
import os
import tempfile


# ── DALL-E 3 ──────────────────────────────────────────────────────────────────

def generate_dalle_image(prompt: str) -> bytes | None:
    """Returns PNG bytes or None on failure."""
    try:
        import httpx
        from openai import OpenAI

        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url        = response.data[0].url
        img_bytes  = httpx.get(url, timeout=30).content
        return img_bytes
    except ImportError:
        return None
    except Exception as e:
        print(f"[image_generator] DALL-E error: {e}")
        return None


# ── Matplotlib charts ─────────────────────────────────────────────────────────

def generate_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float],
    xlabel: str = "",
    ylabel: str = "",
) -> bytes | None:
    """
    Renders a bar, pie, or line chart.
    Returns PNG bytes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for Streamlit Cloud
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # Dark theme
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#111318")

        AMBER  = "#e8a44a"
        COLORS = ["#e8a44a", "#60a5fa", "#4ade80", "#f87171",
                  "#c084fc", "#34d399", "#fb923c", "#a78bfa"]

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=COLORS[:len(labels)], width=0.6)
            ax.bar_label(bars, fmt="%.1f", color="#e2e4e9", fontsize=10)
            ax.set_xlabel(xlabel, color="#9ca3af", fontsize=11)
            ax.set_ylabel(ylabel, color="#9ca3af", fontsize=11)
            ax.tick_params(colors="#9ca3af")
            ax.spines[:].set_color("#1e2028")

        elif chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=COLORS[:len(labels)],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "#0d0f14", "linewidth": 2},
            )
            for text in texts + autotexts:
                text.set_color("#e2e4e9")

        elif chart_type == "line":
            ax.plot(labels, values, color=AMBER, linewidth=2.5,
                    marker="o", markersize=7, markerfacecolor=AMBER)
            ax.fill_between(range(len(labels)), values,
                            alpha=0.15, color=AMBER)
            ax.set_xlabel(xlabel, color="#9ca3af", fontsize=11)
            ax.set_ylabel(ylabel, color="#9ca3af", fontsize=11)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, color="#9ca3af")
            ax.tick_params(colors="#9ca3af")
            ax.spines[:].set_color("#1e2028")
            ax.grid(True, color="#1e2028", alpha=0.5)

        ax.set_title(title, color="#f0ede8", fontsize=14,
                     fontweight="bold", pad=15)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except ImportError:
        return None
    except Exception as e:
        print(f"[image_generator] Chart error: {e}")
        return None


# ── Flowchart via matplotlib ──────────────────────────────────────────────────

def generate_flowchart(title: str, steps: list[str]) -> bytes | None:
    """
    Simple top-down flowchart using matplotlib patches.
    No graphviz needed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        n      = len(steps)
        height = max(n * 1.4 + 1.5, 4)
        fig, ax = plt.subplots(figsize=(8, height))
        fig.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")
        ax.axis("off")

        BOX_W  = 5.5
        BOX_H  = 0.7
        CX     = 4.0   # center x
        START_Y = height - 1.2

        AMBER  = "#e8a44a"
        BOX_BG = "#111318"
        BOX_BD = "#2a2d36"
        TEXT_C = "#e2e4e9"

        ax.set_xlim(0, 8)
        ax.set_ylim(0, height)

        # Title
        ax.text(CX, height - 0.4, title,
                ha="center", va="center",
                color="#f0ede8", fontsize=13, fontweight="bold")

        for i, step in enumerate(steps):
            y = START_Y - i * 1.4

            # Box
            box = FancyBboxPatch(
                (CX - BOX_W / 2, y - BOX_H / 2),
                BOX_W, BOX_H,
                boxstyle="round,pad=0.1",
                facecolor=BOX_BG,
                edgecolor=AMBER if i == 0 else BOX_BD,
                linewidth=1.5 if i == 0 else 0.8,
            )
            ax.add_patch(box)

            # Step number + text
            ax.text(CX - BOX_W / 2 + 0.35, y,
                    str(i + 1),
                    ha="center", va="center",
                    color=AMBER, fontsize=10, fontweight="bold")
            ax.text(CX - BOX_W / 2 + 0.75, y,
                    step[:55] + ("…" if len(step) > 55 else ""),
                    ha="left", va="center",
                    color=TEXT_C, fontsize=9.5)

            # Arrow to next
            if i < n - 1:
                ax.annotate(
                    "",
                    xy=(CX, y - BOX_H / 2 - 0.38),
                    xytext=(CX, y - BOX_H / 2 - 0.02),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=BOX_BD,
                        lw=1.2,
                    ),
                )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except ImportError:
        return None
    except Exception as e:
        print(f"[image_generator] Flowchart error: {e}")
        return None


# ── Router ────────────────────────────────────────────────────────────────────

def generate_visualization(decision: dict) -> bytes | None:
    """
    Takes image_agent decision dict and routes to the right generator.
    Returns PNG bytes or None.
    """
    viz_type = decision.get("type", "dalle")

    if viz_type == "dalle":
        return generate_dalle_image(decision.get("prompt", ""))

    elif viz_type == "chart":
        return generate_chart(
            chart_type=decision.get("chart_type", "bar"),
            title=decision.get("title", ""),
            labels=decision.get("labels", []),
            values=decision.get("values", []),
            xlabel=decision.get("xlabel", ""),
            ylabel=decision.get("ylabel", ""),
        )

    elif viz_type == "flowchart":
        return generate_flowchart(
            title=decision.get("title", ""),
            steps=decision.get("steps", []),
        )

    return None


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_image_lab_page(llm):
    import streamlit as st
    from agents.image_agent import ImageAgent

    st.markdown(
        '<h2 style="font-family:\'DM Serif Display\',serif; color:#f0ede8;">'
        '🖼️ Image Lab</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#6b7280; margin-bottom:1.5rem;">'
        'Describe a concept and get a visual — image, chart, or flowchart.</p>',
        unsafe_allow_html=True
    )

    concept = st.text_area(
        "What do you want to visualize?",
        placeholder="e.g. Flowchart of how TCP handshake works",
        height=100,
        key="image_concept"
    )

    col1, col2, col3 = st.columns(3)
    auto_btn   = col1.button("🤖 Auto-decide", use_container_width=True, type="primary",  key="btn_auto")
    dalle_btn  = col2.button("🎨 Force image",  use_container_width=True, type="secondary", key="btn_dalle")
    chart_btn  = col3.button("📊 Force chart",  use_container_width=True, type="secondary", key="btn_chart")

    if not (auto_btn or dalle_btn or chart_btn):
        st.markdown(
            '<div style="font-size:0.8rem; color:#4b5563; margin-top:0.5rem;">'
            '🎨 Real image via DALL-E 3 (~$0.04) &nbsp;|&nbsp; '
            '📊 Charts and flowcharts are free</div>',
            unsafe_allow_html=True
        )
        return

    if not concept.strip():
        st.warning("Please describe what you want to visualize.")
        return

    with st.spinner("Generating visualization..."):
        if dalle_btn:
            decision = {"type": "dalle", "prompt": concept}
        elif chart_btn:
            # Default to flowchart for force-chart
            agent    = ImageAgent(llm)
            decision = agent.decide(concept)
            if decision.get("type") == "dalle":
                decision = {"type": "flowchart", "title": concept,
                            "steps": ["Step 1", "Step 2", "Step 3"]}
        else:
            # Auto-decide
            agent    = ImageAgent(llm)
            decision = agent.decide(concept)

        img_bytes = generate_visualization(decision)

    if img_bytes:
        st.image(img_bytes, use_column_width=True)
        viz_type = decision.get("type", "")
        if viz_type == "dalle":
            st.caption("🎨 Generated with DALL-E 3")
        else:
            st.caption("📊 Generated with matplotlib — free")

        st.download_button(
            "⬇️ Download PNG",
            data=img_bytes,
            file_name=f"{concept[:30].replace(' ', '_')}.png",
            mime="image/png",
        )
    else:
        st.error(
            "Generation failed. "
            "For images: check OPENAI_API_KEY. "
            "For charts: check matplotlib is installed."
        )