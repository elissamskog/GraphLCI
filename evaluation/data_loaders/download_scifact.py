"""
Download SciFact dataset to local JSONL files.
Location: evaluation/data_loaders/download_scifact.py
Owner: ROOT Instance

Usage:
    python -m evaluation.data_loaders.download_scifact

Writes:
    data/raw/scifact/corpus.jsonl
    data/raw/scifact/claims.jsonl
"""

from __future__ import annotations

import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import load_dataset


_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "scifact")


def main() -> None:
    print("Loading SciFact corpus from HuggingFace...")
    corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)

    print("Loading SciFact claims from HuggingFace...")
    claims_ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)

    os.makedirs(_DATA_DIR, exist_ok=True)

    # Write corpus.jsonl
    corpus_path = os.path.join(_DATA_DIR, "corpus.jsonl")
    doc_count = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for split in corpus_ds:
            for row in corpus_ds[split]:
                record = {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "structured": row["structured"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                doc_count += 1

    # Write claims.jsonl
    claims_path = os.path.join(_DATA_DIR, "claims.jsonl")
    claim_count = 0
    with open(claims_path, "w", encoding="utf-8") as f:
        for split in claims_ds:
            for row in claims_ds[split]:
                record = {
                    "split": split,
                    "id": row["id"],
                    "claim": row["claim"],
                    "evidence": row.get("evidence", {}),
                    "cited_doc_ids": row.get("cited_doc_ids", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                claim_count += 1

    print(f"Wrote {doc_count} documents to {corpus_path}")
    print(f"Wrote {claim_count} claims to {claims_path}")
    print("Done.")


if __name__ == "__main__":
    main()
