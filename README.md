# Runbook AI

AI-assisted operational runbook reader for production incident triage.

Runbook AI helps engineers quickly find the correct operational steps during incidents by semantically searching runbooks and generating grounded, explainable answers.

---

## Problem

Production incidents often involve:
- dozens of runbooks spread across files and tools
- inconsistent naming and structure
- engineers searching under pressure

Traditional keyword search and scripts break down when questions are vague or wording differs from documentation.

---

## Solution

Runbook AI uses **semantic retrieval (RAG)** to:
- understand natural language questions
- find the most relevant runbook sections
- generate concise, grounded answers
- fail safely when information is not present

The system is designed as a **read-only decision support tool**, not an automation engine.

---

## Key Features

- Loads multiple runbooks from `data/runbooks/`
- Semantic search using sentence embeddings
- Grounded answers generated only from retrieved runbook context
- Confidence threshold with safe fallback: **"Not found in runbook"**
- Optional source attribution (runbook name + similarity score)
- Streamlit GUI for interactive use
- CLI mode for quick terminal access

---

## Tech Stack

- **Python**
- **Hugging Face**
  - `sentence-transformers` for embeddings
  - `transformers` for local LLM inference

