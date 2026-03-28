"""
image_generator.py — Study Helper v2
DALL-E 3 image generation + matplotlib diagrams.
No graphviz dependency — Streamlit Cloud safe.
"""
from __future__ import annotations

import io
import os
import tempfile
import streamlit as st
from agents.image_agent import ImageAgent

# ── DALL-E 3 ──────────────────────────────────────────────────────────────────

def generate_dalle_image(prompt: str) -> bytes | None:
    """
    Generate image with DALL-E 3.
    IMPORTANT: Always sends English-only prompts regardless of UI language.
    """
    try:
        import httpx
        from openai import OpenAI

        # Force English-only prompt for DALL-E — never inherit UI language
        forced_prompt = (
            "Create this image with ALL text, labels, captions "
            "and annotations strictly in English only. "
            "No other language anywhere in the image. "
            f"Image request: {prompt}"
        )
        
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=forced_prompt,
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
    # Validate inputs - need at least some data
    if not labels or not values:
        print(f"[image_generator] Chart error: empty labels={labels} or values={values}")
        return None
    if len(labels) != len(values):
        # Truncate to shorter length
        min_len = min(len(labels), len(values))
        labels = labels[:min_len]
        values = values[:min_len]
    
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
    # Validate inputs
    if not steps:
        print(f"[image_generator] Flowchart error: empty steps")
        return None
    
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
            # Ensure step is a string
            step_text = str(step) if not isinstance(step, str) else step

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
                    step_text[:55] + ("…" if len(step_text) > 55 else ""),
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

def generate_visualization_from_concept(concept: str, llm) -> dict:
    """Generate a visualization from concept using ImageAgent"""
    from agents.image_agent import ImageAgent
    
    agent = ImageAgent(llm)
    decision = agent.decide(concept)
    
    if decision.get("type") == "dalle":
        # Generate DALL-E image
        image_url = generate_dalle_image(decision.get("prompt", concept))
        return {
            "type": "image",
            "url": image_url,
            "prompt": decision.get("prompt", concept)
        }
    elif decision.get("type") == "chart":
        # Generate matplotlib chart
        chart_data = generate_chart(
            decision.get("chart_type", "bar"),
            decision.get("title", "Chart"),
            decision.get("labels", []),
            decision.get("values", []),
            decision.get("xlabel", ""),
            decision.get("ylabel", "")
        )
        return {
            "type": "chart",
            "data": chart_data,
            "title": decision.get("title", "Chart")
        }
    elif decision.get("type") == "flowchart":
        # Generate flowchart
        flowchart_data = generate_flowchart(
            decision.get("title", "Flowchart"),
            decision.get("steps", [])
        )
        return {
            "type": "flowchart",
            "data": flowchart_data,
            "title": decision.get("title", "Flowchart")
        }
    else:
        # Fallback to DALL-E
        image_url = generate_dalle_image(concept)
        return {
            "type": "image",
            "url": image_url,
            "prompt": concept
        }


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_image_lab_page(llm):
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
    auto_btn   = col1.button("🤖 Auto-decide", width='stretch', type="primary",  key="btn_auto")
    dalle_btn  = col2.button("🎨 Force image",  width='stretch', type="secondary", key="btn_dalle")
    chart_btn  = col3.button("📊 Force chart",  width='stretch', type="secondary", key="btn_chart")

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

    # Diagram detection - check for diagram-related keywords
    DIAGRAM_KEYWORDS = ["flowchart", "diagram", "chart", "flow", "process map", "steps", "workflow"]
    is_diagram = any(keyword in concept.lower() for keyword in DIAGRAM_KEYWORDS)

    with st.spinner("Generating visualization..."):
        if dalle_btn and not is_diagram:
            # Only use DALL-E if not a diagram
            decision = {"type": "dalle", "prompt": concept}
        elif chart_btn or is_diagram:
            # Force chart or auto-detect diagram
            if is_diagram and "flowchart" in concept.lower():
                # Generate basic flowchart
                decision = {"type": "flowchart", "title": concept,
                           "steps": ["Step 1", "Step 2", "Step 3"]}
            else:
                # Use agent for other charts
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
        # Store in session state for follow-up actions
        st.session_state.last_image = img_bytes
        st.session_state.last_concept = concept
        st.session_state.last_decision = decision
        
        st.image(img_bytes, use_container_width=True)
        viz_type = decision.get("type", "")
        if viz_type == "dalle":
            st.caption("🎨 Generated with DALL-E 3")
        else:
            st.caption("📊 Generated with matplotlib — free")

        # Download button
        st.download_button(
            "⬇️ Download PNG",
            data=img_bytes,
            file_name=f"{concept[:30].replace(' ', '_')}.png",
            mime="image/png",
        )
        
        # Follow-up options
        st.markdown("---")
        st.markdown("### 🎨 Want to modify this image?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✏️ Edit", width='stretch', key="btn_edit"):
                st.session_state.show_edit_dialog = True
                st.rerun()
        
        with col2:
            if st.button("🔄 Regenerate", width='stretch', key="btn_regenerate"):
                st.session_state.regenerate_image = True
                st.rerun()
        
        with col3:
            if st.button("❓ Ask Question", width='stretch', key="btn_ask_question"):
                st.session_state.show_question_dialog = True
                st.rerun()
        
        # Edit dialog
        if st.session_state.get("show_edit_dialog"):
            st.markdown("#### ✏️ Edit the image")
            edit_prompt = st.text_area(
                "Describe the changes you want:",
                value=concept,
                key="edit_prompt",
                height=100
            )
            col_save, col_cancel = st.columns(2)
            if col_save.button("Apply Changes", width='stretch'):
                if edit_prompt.strip():
                    with st.spinner("Applying changes..."):
                        new_decision = {"type": "dalle", "prompt": edit_prompt}
                        new_img = generate_visualization(new_decision)
                        if new_img:
                            st.session_state.last_image = new_img
                            st.session_state.last_concept = edit_prompt
                            st.session_state.last_decision = new_decision
                            st.success("✅ Image updated!")
                            st.session_state.show_edit_dialog = False
                            st.rerun()
                        else:
                            st.error("Failed to update image")
            if col_cancel.button("Cancel", width='stretch'):
                st.session_state.show_edit_dialog = False
                st.rerun()
        
        # Question dialog
        if st.session_state.get("show_question_dialog"):
            st.markdown("#### ❓ Ask a question about this image")
            question = st.text_input(
                "What would you like to know?",
                key="image_question",
                placeholder="e.g., What does this diagram show? Can you explain the flow?"
            )
            col_ask, col_cancel = st.columns(2)
            if col_ask.button("Ask", width='stretch'):
                if question.strip():
                    with st.spinner("Analyzing image..."):
                        # Use the concept and decision to answer
                        context = f"The image shows: {concept}\n"
                        if viz_type != "dalle":
                            context += f"It's a {viz_type} diagram.\n"
                        # Here you could integrate with an LLM to answer about the image
                        st.info(f"🤖 This is a {viz_type} visualization about: {concept}")
                        st.session_state.show_question_dialog = False
                        st.rerun()
            if col_cancel.button("Cancel", width='stretch'):
                st.session_state.show_question_dialog = False
                st.rerun()
        
        # Handle regenerate
        if st.session_state.get("regenerate_image"):
            with st.spinner("Regenerating..."):
                new_img = generate_visualization(decision)
                if new_img:
                    st.session_state.last_image = new_img
                    st.success("✅ Image regenerated!")
                    st.session_state.regenerate_image = False
                    st.rerun()
                else:
                    st.error("Failed to regenerate")
                    st.session_state.regenerate_image = False
    else:
        st.error(
            "Generation failed. "
            "For images: check OPENAI_API_KEY. "
            "For charts: check matplotlib is installed."
        )