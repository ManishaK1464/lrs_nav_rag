"""All settings in one place. Change values here, not inside the other files."""

# Folders
DATA_DIR = "data"            # put your LRS-Nav files here (report PDF, .md, .py, .yaml)
INDEX_DIR = "faiss_index"    # the vector database is saved here after ingest.py

# File types we read
FILE_TYPES = [".pdf", ".md", ".txt", ".py", ".yaml", ".yml"]

# Chunking
CHUNK_SIZE = 800             # characters per chunk
CHUNK_OVERLAP = 100          # characters shared between neighbouring chunks

# Models (both downloaded automatically from Hugging Face on first run)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # small, fast on CPU
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"    # try "Qwen/Qwen2.5-0.5B-Instruct" if too slow

# Retrieval + generation
TOP_K = 4                    # how many chunks to give the LLM
MAX_NEW_TOKENS = 300         # max length of the answer
