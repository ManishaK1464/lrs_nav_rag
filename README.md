# LRS-Nav Docs Assistant (RAG, CPU-only)

A question-answering assistant over my project documents, built with an
open-source Hugging Face LLM. Runs fully locally on a CPU: no API keys, no cloud.

## How it works
1. **Load**: reads PDF, Markdown, Python and YAML files from `data/`
2. **Chunk**: splits them into ~800-character pieces
3. **Embed**: turns each chunk into a vector (`BAAI/bge-small-en-v1.5`)
4. **Store**: saves the vectors in a FAISS index
5. **Retrieve**: finds the top 4 chunks for a question
6. **Generate**: `Qwen2.5-1.5B-Instruct` answers using only those chunks, and shows its sources

```
question -> embed -> FAISS search -> top-k chunks -> prompt -> Qwen LLM -> answer + sources
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
# 1. put your documents in data/
python ingest.py                                   # build the index
python rag.py "What does the Reviewer agent do?"   # quick test
streamlit run app.py                               # chat UI
```

## Design choices
- **Small models** so it runs on a normal laptop CPU
- **Answers only from context**, and says "I don't know" otherwise, to reduce hallucination
- **Sources shown** with every answer, so answers can be checked
- **Documents are not in the repo** (`data/` is git-ignored)

## Next steps
- Swap to Azure OpenAI + Azure AI Search for a cloud version
- Add an evaluation set (questions + expected answers)
- Hybrid search (keyword + vector)
