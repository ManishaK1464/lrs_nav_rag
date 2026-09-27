"""A small knowledge graph of the project: things (nodes) + relations (arrows).

Facts come from a CSV of triples:  subject, relation, object
    Hovering, is_a, Environment
    Reviewer, sends_REVISE_feedback_to, Coder

Used in two ways by rag.py:
  1. Direct answers for list / count questions ("Which environments are there?")
     -> exact, no guessing, no LLM needed
  2. Extra facts added to the LLM context when the question mentions a known thing
     (this "graph + vector search" combination is the idea behind GraphRAG)

Try it:
    python kg.py "How many environments are there?"
"""
import csv
import sys
from pathlib import Path

import networkx as nx

import config

LIST_WORDS = ["list", "name", "which", "what are", "how many", "all "]
TYPE_ALIASES = {"environment": ["environment", "env"]}  # extra words that mean the same type


def normalize(text):
    return text.lower().replace("-", " ").replace("_", " ")


class KnowledgeGraph:
    def __init__(self, path=config.KG_FILE):
        self.graph = nx.MultiDiGraph()
        self.enabled = Path(path).exists()
        if not self.enabled:
            print(f"(knowledge graph file '{path}' not found, graph disabled)")
            return
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s, r, o = row["subject"].strip(), row["relation"].strip(), row["object"].strip()
                self.graph.add_edge(s, o, relation=r)

    # ---------- basic queries ----------
    def nodes_of_type(self, type_name):
        """All things with an 'is_a <type_name>' arrow, in file order."""
        return [
            s for s, o, d in self.graph.edges(data=True)
            if d["relation"] == "is_a" and o.lower() == type_name.lower()
        ]

    def types(self):
        return sorted({o for _, o, d in self.graph.edges(data=True) if d["relation"] == "is_a"})

    def facts_about(self, node):
        facts = [f"{node} {d['relation'].replace('_', ' ')} {o}" for _, o, d in self.graph.out_edges(node, data=True)]
        facts += [f"{s} {d['relation'].replace('_', ' ')} {node}" for s, _, d in self.graph.in_edges(node, data=True)]
        return facts

    def find_entities(self, text):
        """Known nodes whose name appears in the text (e.g. 'Reviewer', 'Figure-8')."""
        t = normalize(text)
        return [n for n in self.graph.nodes if len(n) > 2 and normalize(n) in t]

    # ---------- used by rag.py ----------
    def try_answer(self, question):
        """Answer list/count questions directly from the graph. Returns None if it can't."""
        if not self.enabled:
            return None
        q = normalize(question)
        if not any(w in q for w in LIST_WORDS):
            return None
        for type_name in self.types():
            words = TYPE_ALIASES.get(type_name.lower(), [type_name.lower()])
            if any(w in q for w in words):
                items = self.nodes_of_type(type_name)
                if "how many" in q:
                    return f"There are {len(items)} {type_name.lower()}s: {', '.join(items)}."
                return f"The {type_name.lower()}s are: {', '.join(items)}."
        return None

    def context_for(self, question):
        """Graph facts about things mentioned in the question, as text for the LLM."""
        if not self.enabled:
            return ""
        facts = []
        types = {t.lower() for t in self.types()}
        for entity in self.find_entities(question):
            if entity.lower() in types:
                continue  # skip broad words like "Agent" / "Environment" (too many facts, too little signal)
            facts += self.facts_about(entity)
        return "\n".join(dict.fromkeys(facts))  # remove duplicates, keep order


if __name__ == "__main__":
    kg = KnowledgeGraph()
    q = " ".join(sys.argv[1:]) or "Which environments are there?"
    print("Types in graph:", kg.types())
    print("Direct answer :", kg.try_answer(q))
    print("Facts found   :\n" + (kg.context_for(q) or "(none)"))