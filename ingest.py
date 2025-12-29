from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma

from pptx import Presentation  # comes from python-pptx


# -----------------------
# Config (edit if needed)
# -----------------------
RAW_DIR = Path("data/raw")
DB_DIR = Path("vectordb")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# -----------------------
# Helpers: load files
# -----------------------
def load_pdf(path: Path) -> List[Document]:
    # PyPDFLoader returns one Document per page with metadata including "page"
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # Add filename for citations
    for d in docs:
        d.metadata["filename"] = path.name
        d.metadata["filetype"] = "pdf"
    return docs


def load_docx(path: Path) -> List[Document]:
    # Docx2txtLoader returns one Document (no page numbers usually)
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["filename"] = path.name
        d.metadata["filetype"] = "docx"
    return docs


def load_pptx(path: Path) -> List[Document]:
    # We parse slides ourselves (no extra unstructured dependency)
    prs = Presentation(str(path))
    docs: List[Document] = []

    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            # Some shapes have text frames
            if hasattr(shape, "text") and shape.text:
                txt = shape.text.strip()
                if txt:
                    parts.append(txt)

        slide_text = "\n".join(parts).strip()
        if slide_text:
            docs.append(
                Document(
                    page_content=slide_text,
                    metadata={
                        "filename": path.name,
                        "filetype": "pptx",
                        "slide": i,  # slide number for citations
                    },
                )
            )
    return docs


def load_all_files(raw_dir: Path) -> List[Document]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing folder: {raw_dir.resolve()}")

    docs: List[Document] = []
    files = sorted([p for p in raw_dir.rglob("*") if p.is_file()])

    if not files:
        print(f" No files found in {raw_dir.resolve()}")
        return docs

    for path in files:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                docs.extend(load_pdf(path))
                print(f" Loaded PDF: {path.name}")
            elif ext == ".docx":
                docs.extend(load_docx(path))
                print(f" Loaded DOCX: {path.name}")
            elif ext == ".pptx":
                docs.extend(load_pptx(path))
                print(f" Loaded PPTX: {path.name}")
            else:
                print(f" Skipped (unsupported): {path.name}")
        except Exception as e:
            print(f" Failed to load {path.name}: {e}")

    return docs


# -----------------------
# Build / update Chroma
# -----------------------
def build_or_update_db(docs: List[Document], reset_db: bool = False) -> None:
    if not docs:
        print(" No documents loaded. Nothing to index.")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Embeddings (needs OPENAI_API_KEY in .env)
    embeddings = OpenAIEmbeddings()

    # Reset DB if requested
    if reset_db and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    # Create or append
    if DB_DIR.exists():
        # Load existing DB and append
        vectordb = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
        )
        vectordb.add_documents(chunks)
        print(f" Updated existing DB: {DB_DIR.resolve()}")
    else:
        # Create new DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
        )
        vectordb.persist()
        print(f" Created new DB: {DB_DIR.resolve()}")

    print(f" Loaded docs: {len(docs)}")
    print(f" Stored chunks: {len(chunks)}")


def main():
    # Quick key check
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in your .env file.")

    print(" Day 2: Ingesting files and building memory...")
    print(f"Raw folder: {RAW_DIR.resolve()}")
    print(f"DB folder:  {DB_DIR.resolve()}")

    docs = load_all_files(RAW_DIR)

    # For now: keep reset_db=False so you can re-run and append.
    # If you ever want a fresh rebuild, set reset_db=True.
    build_or_update_db(docs, reset_db=False)

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
