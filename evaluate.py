"""Measure retrieval quality with Recall@k and MRR. No LLM needed, so it runs fast.

    python evaluate.py

eval/questions.csv has one question per row plus the file(s) that SHOULD be found.
Several acceptable files are separated by "|". A match means the expected text
appears anywhere in the retrieved file path (e.g. "env_patch.py").

Recall@k : in how many questions was a correct file in the top k?      (0 to 1)
MRR      : average of 1/rank of the first correct file (1st=1, 2nd=0.5) (0 to 1)
"""
import csv

import config
from store import get_vector_db

QUESTIONS_FILE = "eval/questions.csv"
K = config.TOP_K


def normalize(path):
    return path.replace("\\", "/").lower()


def first_correct_rank(docs, expected):
    """Position (1, 2, 3...) of the first correct source, or None if not found."""
    for rank, d in enumerate(docs, start=1):
        source = normalize(d.metadata.get("source", ""))
        if any(normalize(e) in source for e in expected):
            return rank
    return None


def evaluate(db, rows, method):
    hits, reciprocal_ranks = 0, []
    print(f"\n=== {method} (k={K}) ===")
    for row in rows:
        question = row["question"]
        expected = [e.strip() for e in row["expected_sources"].split("|")]
        if method == "similarity":
            docs = db.similarity_search(question, k=K)
        else:
            docs = db.max_marginal_relevance_search(question, k=K, fetch_k=20)

        rank = first_correct_rank(docs, expected)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0)
        mark = f"rank {rank}" if rank else "MISS  "
        print(f"  [{mark}] {question}")
        if rank is None:  # show what was found instead, to help debugging
            found = [normalize(d.metadata.get("source", "?")).split("/")[-1] for d in docs]
            print(f"           expected {expected}, got {found}")

    recall = hits / len(rows)
    mrr = sum(reciprocal_ranks) / len(rows)
    print(f"  Recall@{K}: {recall:.2f}   MRR: {mrr:.2f}")
    return recall, mrr


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    db = get_vector_db()

    results = {m: evaluate(db, rows, m) for m in ["similarity", "mmr"]}

    print("\n=== Summary ===")
    print(f"  {'method':<12}{'Recall@' + str(K):<12}MRR")
    for m, (recall, mrr) in results.items():
        print(f"  {m:<12}{recall:<12.2f}{mrr:.2f}")


if __name__ == "__main__":
    main()