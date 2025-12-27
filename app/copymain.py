from ingest import load_runbooks, build_index
from retriever import retrieve
from qa import answer

if __name__ == "__main__":
    docs = load_runbooks("data/runbooks")
    index, chunks, model = build_index(docs)

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break

        context = retrieve(q, index, chunks, model)
        response = answer(q, context)

        print("\nAnswer:")
        print(response)

