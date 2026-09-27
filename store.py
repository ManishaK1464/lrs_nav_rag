"""One place that opens the vector database, used by both ingest.py and rag.py."""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_db(embeddings=None):
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=config.DB_DIR,               # saved on disk automatically
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
    )