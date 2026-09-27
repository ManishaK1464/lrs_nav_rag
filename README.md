# LRS-Nav Docs Assistant (RAG, CPU-only)

A question-answering assistant over my project documents, built with an
open-source Hugging Face LLM and a Chroma vector database. Runs fully locally on a CPU: no API keys, no cloud.

## How it works
**Ingestion pipeline (`ingest.py`): a small ETL process**
1. **Extract**: reads PDF, Markdown, Python and YAML files from `data/` (including subfolders)
2. **Transform**: splits Python at classes/functions and text by paragraphs, removes duplicate chunks, and adds the file name to each chunk
3. **Load**: embeds chunks (`BAAI/bge-small-en-v1.5`) and stores them in **Chroma**
4. **Incremental**: each file gets a SHA-256 fingerprint; only new or changed files are re-embedded, and deleted files are removed from the database

**Question answering (`rag.py`)**
1. Retrieves the top 4 chunks with **MMR** (relevant *and* diverse)
2. `Qwen2.5-1.5B-Instruct` answers using only those chunks
3. Shows the source files with every answer

```
data/ -> load -> split -> dedupe -> embed -> Chroma
question -> embed -> MMR search -> top-k chunks -> Qwen LLM -> answer + sources
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run
```bash
python ingest.py                                   # build or update the database
python ingest.py --rebuild                         # rebuild from scratch
python rag.py "What does the Reviewer agent do?"   # quick test
streamlit run app.py                               # chat UI
```

## Design choices
- **Small models** so it runs on a normal laptop CPU
- **Chroma** vector DB: persistent, supports updates/deletes by ID and metadata filters
- **Incremental ingestion** so only changed files are processed
- **Answers only from context**, with sources shown, to reduce hallucination
- **Documents are not in the repo** (`data/` is git-ignored)

## Known limits
- "List everything" questions are hard, because the model only sees the top-k chunks

## Next steps
- Azure OpenAI + Azure AI Search for a cloud version
- Scheduled or event-triggered ingestion
- MCP server so agents can query tabular results
- Evaluation set (questions + expected answers)