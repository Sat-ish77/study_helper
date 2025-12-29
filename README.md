# 📚 Study Helper

A personal RAG (Retrieval-Augmented Generation) app that turns your study notes into an intelligent Q&A assistant. Upload PDFs, Word docs, or PowerPoints — then ask questions and get answers with citations.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange)

---

## ✨ Features

- **📄 Multi-format support** — PDF, DOCX, PPTX
- **🔍 Semantic search** — Finds relevant content by meaning, not just keywords
- **📝 Citations** — Every answer includes source file + page/slide number
- **🌐 Web fallback** — Automatically searches the web when files don't have the answer
- **⚙️ Answer modes** — Short, medium, or long answer styles
- **🧠 Smart detection** — Detects comparison questions ("A vs B") and checks both sides exist

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repo
git clone https://github.com/yourusername/study-helper.git
cd study-helper

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

This reads your files, splits them into chunks, and saves them to `vectordb/`.

### 5. Start asking questions

```bash
uv run python main.py
```

---

## 💬 Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `:web on` | Enable web search fallback |
| `:web off` | Disable web search fallback |
| `:mode short` | Brief answers (3-6 bullet points) |
| `:mode medium` | Balanced answers (default) |
| `:mode long` | Detailed exam-style answers |
| `exit` | Quit the app |

### Example Session

```
📚 Study Helper (files-first) ready.

Q: :web on
✅ Web fallback is now: ON

Q: What are the three main components of the cytoskeleton?

A: The cytoskeleton consists of three main components:
   - **Microtubules**: Long tubes formed by tubulin protein... [S1]
   - **Microfilaments**: Actin polymers that maintain cell structure... [S2]
   - **Intermediate filaments**: Tissue-specific structures... [S3]

File Sources:
- [S1] notes of actin filament.pdf (page 1)
- [S2] notes of actin filament.pdf (page 3)
- [S3] notes of actin filament.pdf (page 3)
```

---

## 📁 Project Structure

```
study-helper/
├── data/
│   └── raw/              # Put your study files here
├── vectordb/             # Generated: vector database storage
├── ingest.py             # Builds the knowledge base from files
├── main.py               # CLI Q&A interface
├── app.py                # (Coming soon) Streamlit web UI
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration (uv)
└── .env                  # Your API keys (create this)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | [LangChain](https://langchain.com/) |
| LLM | [OpenAI GPT-4](https://openai.com/) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| Web Search | [Tavily](https://tavily.com/) |
| UI (Day 4) | [Streamlit](https://streamlit.io/) |

---

## 🔧 Configuration

You can tune these settings in `main.py`:

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

## 🗺️ Roadmap

- [x] Day 1: Project setup + dependencies
- [x] Day 2: Ingestion script (PDF/DOCX/PPTX → vector DB)
- [x] Day 3: CLI Q&A with citations
- [ ] Day 4: Streamlit web UI
- [ ] Day 5: Answer modes (explain simply, MCQs, summaries)
- [ ] Day 6: Web fallback toggle in UI
- [ ] Day 7: Polish + demo

---

## 📄 License

MIT License — feel free to use and modify!

---

## 🙏 Acknowledgments

Built using LangChain, OpenAI, and ChromaDB.

