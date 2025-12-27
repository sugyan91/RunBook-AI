# RunBook-AI
Your Ai enabled Runbook for every PRODUCTION Incidents
# Runbook AI (Side Project)

AI-assisted operational runbook reader. Uses semantic retrieval + a lightweight LLM to answer questions grounded in runbook content.

## What it does
- Loads multiple runbooks from `data/runbooks/`
- Retrieves relevant runbook chunks using embeddings (semantic search)
- Generates a grounded answer using an instruction-tuned LLM
- Safe fallback: returns "Not found in runbook" when confidence is low

## Tech
- Python
- Hugging Face: sentence-transformers for embeddings + transformers for LLM inference
- NumPy cosine similarity retrieval (simple + stable on macOS)
- Streamlit GUI

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

>>>>>>> 0a3868e (Initial commit: RunBook-AI)
