import argparse

from app.ingest import load_runbooks, build_index
from app.retriever import retrieve_with_scores
from app.qa import answer

CONFIDENCE_THRESHOLD = 0.20


def top_source_only(retrieved):
    best_score, best_rec = max(retrieved, key=lambda x: x[0])
    return f"{best_rec['runbook']} ({best_score:.3f})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runbook AI (CLI)")
    parser.add_argument("--source", action="store_true", help="Print top runbook source")
    args = parser.parse_args()

    runbooks = load_runbooks("data/runbooks")
    index_matrix, chunk_records, model = build_index(runbooks)

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        retrieved = retrieve_with_scores(question, index_matrix, chunk_records, model, k=6)
        top_score = retrieved[0][0] if retrieved else 0.0

        if top_score < CONFIDENCE_THRESHOLD:
            print("Not found in runbook.")
            continue

        context_chunks = [rec["text"] for score, rec in retrieved]
        response = answer(question, context_chunks)
        response = response.replace("Not found in runbook.", "").strip()

        print(response)

        if args.source:
            print(f"Source: {top_source_only(retrieved)}")

