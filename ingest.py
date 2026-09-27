"""Step 1: read your docs -> split into chunks -> embed -> save a FAISS index.
Run once, and again whenever you add or change files in data/.

    python ingest.py
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents():
    docs = []
    for path in Path(config.DATA_DIR).rglob("*"):
        if path.suffix.lower() not in config.FILE_TYPES:
            continue
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = path.name   # short name, shown as the source in answers
        docs.extend(loaded)
        print(f"Loaded {path.name} ({len(loaded)} part(s))")
    return docs


def main():
    docs = load_documents()
    if not docs:
        print(f"No files found in '{config.DATA_DIR}/'. Add your LRS-Nav files and run again.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} document(s) into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.INDEX_DIR)
    print(f"Saved index to '{config.INDEX_DIR}/'")


if __name__ == "__main__":
    main()
