"""Step 2: question -> find relevant chunks -> Hugging Face LLM answers from them.

Test from the terminal:
    python rag.py "What does the Reviewer agent do?"
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from store import get_vector_db

SYSTEM_PROMPT = (
    "You answer questions about the LRS-Nav project. "
    "Use ONLY the context provided. "
    "Answer the question directly and briefly. "
    "Do NOT explain code or describe how to find the answer unless the user asks for that. "
    "If the answer is not in the context, say: \"I don't know based on the documents.\""
)


class RAG:
    def __init__(self):
        # 1. Open the Chroma vector database built by ingest.py
        self.db = get_vector_db()
        if not self.db.get(limit=1)["ids"]:
            raise RuntimeError("The database is empty. Run: python ingest.py")
        # 2. Load the LLM on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=torch.float32)
        self.model.eval()

    def retrieve(self, question):
        # MMR = pick chunks that are relevant AND different from each other
        return self.db.similarity_search(question, k=config.TOP_K)

    def answer(self, question):
        docs = self.retrieve(question)
        context = "\n\n".join(
            f"[{i + 1}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
            for i, d in enumerate(docs)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)

        new_tokens = output[0][inputs["input_ids"].shape[1]:]   # keep only the answer, not the prompt
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, docs


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is LRS-Nav?"
    rag = RAG()
    ans, sources = rag.answer(q)
    print("\nAnswer:\n", ans)
    print("\nSources:", sorted({d.metadata.get("source", "?") for d in sources}))