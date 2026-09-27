"""Incremental ingestion: only NEW or CHANGED files are embedded.

    python ingest.py            # normal run: only processes what changed
    python ingest.py --rebuild  # delete the database and rebuild everything

How it knows what changed:
    each file gets a fingerprint (SHA-256 hash). Fingerprints are saved in
    chroma_db/manifest.json. Same fingerprint as last time -> file skipped.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import config
from store import get_vector_db

MANIFEST = Path(config.DB_DIR) / "manifest.json"

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)
CODE_SPLITTER = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)


# ---------- small helpers ----------
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(manifest):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def find_files():
    """Returns {"folder/file.py": Path(...)} for every supported file in data/."""
    return {
        str(p.relative_to(config.DATA_DIR)): p
        for p in sorted(Path(config.DATA_DIR).rglob("*"))
        if p.is_file() and p.suffix.lower() in config.FILE_TYPES
    }


def load_file(path, source):
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = source
    return docs


def split(docs, source):
    splitter = CODE_SPLITTER if source.endswith(".py") else TEXT_SPLITTER
    return splitter.split_documents(docs)


def delete_file_chunks(db, source):
    ids = db.get(where={"source": source})["ids"]
    if ids:
        db.delete(ids=ids)
    return len(ids)


# ---------- main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="delete the database and start fresh")
    args = parser.parse_args()

    if args.rebuild and Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)
        print("Deleted old database, rebuilding from scratch")

    files = find_files()
    if not files:
        print(f"No files found in '{config.DATA_DIR}/'. Add your files and run again.")
        return

    db = get_vector_db()
    manifest = load_manifest()
    stats = {"new_or_changed": 0, "unchanged": 0, "removed": 0, "chunks_added": 0, "duplicates_skipped": 0}

    # 1. Files deleted from data/ -> remove their chunks
    for source in list(manifest):
        if source not in files:
            n = delete_file_chunks(db, source)
            del manifest[source]
            stats["removed"] += 1
            print(f"Removed {source} ({n} chunks)")

    # 2. New or changed files -> (re)embed; unchanged -> skip
    seen_in_this_run = set()
    for source, path in files.items():
        fingerprint = file_hash(path)
        if manifest.get(source) == fingerprint:
            stats["unchanged"] += 1
            continue

        delete_file_chunks(db, source)  # if the file changed, remove its old chunks first
        try:
            docs = load_file(path, source)
        except Exception as e:  # broken or unreadable file: log it and move on
            print(f"Skipped {source}: {e}")
            continue

        chunks, ids = [], []
        for i, chunk in enumerate(split(docs, source)):
            if not chunk.page_content.strip():
                continue  # empty chunk
            h = text_hash(chunk.page_content)
            already_stored = db.get(where={"content_hash": h}, limit=1)["ids"]
            if h in seen_in_this_run or already_stored:
                stats["duplicates_skipped"] += 1  # same text already stored (copy-pasted code)
                continue
            seen_in_this_run.add(h)
            chunk.metadata["content_hash"] = h
            chunk.page_content = f"File: {source}\n{chunk.page_content}"
            chunks.append(chunk)
            ids.append(f"{source}::{i}")

        if chunks:
            db.add_documents(filter_complex_metadata(chunks), ids=ids)
        manifest[source] = fingerprint
        save_manifest(manifest)  # save after every file, so a crash loses nothing
        stats["new_or_changed"] += 1
        stats["chunks_added"] += len(chunks)
        print(f"Indexed {source} ({len(chunks)} chunks)")

    total = len(db.get()["ids"])
    print("\nSummary")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  total chunks in database: {total}")


if __name__ == "__main__":
    main()