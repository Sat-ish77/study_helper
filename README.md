# 📚 Study Helper

A personal RAG (Retrieval-Augmented Generation) app that turns your study notes into an intelligent Q&A assistant. Upload PDFs, Word docs, or PowerPoints — then ask questions and get answers with citations.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red)

---

## ✨ Features

### Core Features
- **📄 Multi-format support** — PDF, DOCX, PPTX
- **🔍 Semantic search** — Finds relevant content by meaning, not just keywords
- **📝 Citations** — Every answer includes source file + page/slide number
- **🌐 Web fallback** — Automatically searches the web when files don't have the answer
- **⚙️ Answer modes** — Short, medium, or long answer styles

### Quick Actions
- **😊 Simpler** — Re-explains answers in simple, everyday language
- **🔬 Technical** — Adds more scientific detail and terminology
- **🇳🇵 Nepali** — Explains concepts in Nepali for easier understanding
- **🔊 Listen** — Text-to-speech reads answers aloud
- **💬 Deep Dive** — Follow-up chat panel for deeper exploration

### Additional Features
- **🧪 Quiz Lab** — Auto-generate quizzes (MCQ, True/False, Fill-in-blank)
- **🎨 Themes** — 4 color themes (Night Study, Ocean Blue, Forest Green, Purple Haze)
- **📊 Quiz Stats** — Track your quiz performance over time

---

## 🖼️ Screenshots

### Home Page
Beautiful landing page with feature overview and audio introduction.

### Study Helper
Ask questions, get cited answers, use quick actions for different explanations.

### Quiz Lab
Generate and take quizzes from your study notes.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repo
git clone https://github.com/Sat-ish77/study_helper.git
cd study_helper

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here   # Optional: for web fallback
```

### 3. Add your notes

Drop your study files into the `data/raw/` folder:
- PDFs (`.pdf`)
- Word documents (`.docx`)
- PowerPoint presentations (`.pptx`)

### 4. Build the knowledge base

```bash
uv run python ingest.py
```

### 5. Start the web app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
study-helper/
├── app.py                    # Home page (Streamlit)
├── pages/
│   ├── 1_📚_Study_Helper.py  # Main Q&A interface
│   └── 2_🧪_Quiz_Lab.py      # Quiz generation & grading
├── main.py                   # Backend RAG logic
├── ingest.py                 # Document ingestion script
├── data/
│   └── raw/                  # Put your study files here
├── vectordb/                 # Generated: vector database
├── pyproject.toml            # Project configuration (uv)
└── .env                      # Your API keys (create this)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | [LangChain](https://langchain.com/) |
| LLM | [OpenAI GPT-4](https://openai.com/) |
| Embeddings | OpenAI text-embedding |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| Web Search | [Tavily](https://tavily.com/) |
| UI | [Streamlit](https://streamlit.io/) |
| TTS | [gTTS](https://gtts.readthedocs.io/) |

---

## 🎨 Available Themes

| Theme | Description |
|-------|-------------|
| 🌙 Night Study | Dark with amber accents (default) |
| 🌊 Ocean Blue | Dark navy with cyan accents |
| 🌲 Forest Green | Dark with green accents |
| 🔮 Purple Haze | Dark with purple accents |

---

## 💬 CLI Mode (Optional)

You can also use the CLI interface:

```bash
uv run python main.py
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `:web on` | Enable web search fallback |
| `:web off` | Disable web search fallback |
| `:mode short` | Brief answers |
| `:mode medium` | Balanced answers (default) |
| `:mode long` | Detailed exam-style answers |
| `exit` | Quit the app |

---

## 🔧 Configuration

Tune these settings in `main.py`:

```python
# Retrieval settings
TOP_K = 8                    # Number of chunks to retrieve
FETCH_K = 24                 # Chunks fetched before MMR selection

# Relevance thresholds (higher = stricter)
MIN_AVG_SCORE_SHORT = 0.65
MIN_AVG_SCORE_MED = 0.70
MIN_AVG_SCORE_LONG = 0.75

# Model settings
DEFAULT_MODEL = "gpt-4o"
TEMPERATURE = 0
```

---

## 📋 Requirements

- Python 3.12+
- OpenAI API key
- (Optional) Tavily API key for web fallback

---

## 🗺️ Completed Features

- ✅ Project setup + dependencies
- ✅ Ingestion script (PDF/DOCX/PPTX → vector DB)
- ✅ CLI Q&A with citations
- ✅ Streamlit web UI (multi-page)
- ✅ Answer modes (Short/Medium/Long)
- ✅ Web fallback toggle
- ✅ Quick actions (Simpler, Technical, Nepali)
- ✅ Text-to-Speech
- ✅ Deep Dive chat panel
- ✅ Quiz Lab (MCQ, True/False, Fill-in-blank)
- ✅ Multiple themes
- ✅ Welcome audio introduction

---

## 📄 License

MIT License — feel free to use and modify!

---

## 🙏 Acknowledgments

- Built using LangChain, OpenAI, ChromaDB, and Streamlit
- Development assisted by [Cursor](https://cursor.com/) + Claude AI

---

**Personalized study app built by Satish** 🚀
