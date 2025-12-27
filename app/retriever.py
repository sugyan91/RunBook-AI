import numpy as np


def retrieve_with_scores(question, index_matrix, chunk_records, model, k=6, category=None):
    q = model.encode([question]).astype("float32")[0]
    q = q / (np.linalg.norm(q) + 1e-12)

    sims = index_matrix @ q  # cosine similarities for all chunks

    # Filter indices by category if provided
    if category and category != "all":
        valid_idx = [i for i, rec in enumerate(chunk_records) if rec["category"] == category]
        if not valid_idx:
            return []
        sims_sub = sims[valid_idx]
        top_local = np.argsort(-sims_sub)[:k]
        top_idx = [valid_idx[i] for i in top_local]
    else:
        top_idx = np.argsort(-sims)[:k]

    results = []
    seen = set()
    for idx in top_idx:
        rec = chunk_records[int(idx)]
        key = (rec["runbook"], rec["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append((float(sims[int(idx)]), rec))

    return results

