from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ingest import load_runbooks, build_index
from app.retriever import retrieve_with_scores
from app.qa import answer

CONFIDENCE_THRESHOLD = 0.35

app = FastAPI(title="AI Runbook Reader API")

# Load once at startup
_runbooks = load_runbooks("data/runbooks")
_index_matrix, _chunk_records, _embed_model = build_index(_runbooks)


class AskRequest(BaseModel):
    question: str
    category: str | None = "all"
    short_answer: bool | None = True


class AskResponse(BaseModel):
    answer: str
    source: str | None = None
    score: float | None = None


def top_source_only(retrieved):
    best_score, best_rec = max(retrieved, key=lambda x: x[0])
    return best_rec["runbook"], best_score


@app.get("/", response_class=HTMLResponse)
def home():
    # A tiny web page that calls /ask
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>AI Runbook Reader</title>
  <style>
    body { font-family: -apple-system, system-ui, Arial; margin: 40px; max-width: 900px; }
    input, select, button, textarea { font-size: 16px; padding: 10px; }
    textarea { width: 100%; height: 140px; }
    .row { display: flex; gap: 10px; margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>AI Runbook Reader</h1>
  <div class="row">
    <input id="q" style="flex: 1" placeholder="Ask a question..."/>
    <select id="cat">
      <option value="all">all</option>
      <option value="auth">auth</option>
      <option value="network">network</option>
      <option value="loadbalancer">loadbalancer</option>
      <option value="general">general</option>
    </select>
    <button onclick="ask()">Ask</button>
  </div>

  <label><input type="checkbox" id="short" checked/> Short answer</label>
  <p><strong>Answer</strong></p>
  <textarea id="ans" readonly></textarea>
  <p id="src"></p>

<script>
async function ask() {
  const question = document.getElementById("q").value.trim();
  const category = document.getElementById("cat").value;
  const short_answer = document.getElementById("short").checked;
  if (!question) return;

  document.getElementById("ans").value = "Thinking...";
  document.getElementById("src").textContent = "";

  const resp = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({question, category, short_answer})
  });
  const data = await resp.json();
  document.getElementById("ans").value = data.answer || "";
  if (data.source) {
    document.getElementById("src").textContent = `Source: ${data.source} (score=${data.score.toFixed(3)})`;
  }
}
</script>
</body>
</html>
"""


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        return AskResponse(answer="Please provide a question.")

    retrieved = retrieve_with_scores(
        question,
        _index_matrix,
        _chunk_records,
        _embed_model,
        k=8,
        category=req.category or "all",
    )
    top_score = retrieved[0][0] if retrieved else 0.0

    if top_score < CONFIDENCE_THRESHOLD:
        return AskResponse(answer="Not found in runbook.")

    context_chunks = [rec["text"] for score, rec in retrieved]

    q_for_model = question
    if req.short_answer:
        q_for_model = f"{question}\n\nAnswer in 1-2 short sentences. No extra commentary."

    resp = answer(q_for_model, context_chunks).replace("Not found in runbook.", "").strip()

    src, s = top_source_only(retrieved)
    return AskResponse(answer=resp, source=src, score=s)

