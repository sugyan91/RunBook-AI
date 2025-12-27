# --- FIX STREAMLIT IMPORT PATH ISSUE (DO NOT REMOVE) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import re
import streamlit as st

from app.ingest import load_runbooks, build_index
from app.retriever import retrieve_with_scores
from app.qa import answer

CONFIDENCE_THRESHOLD = 0.35


@st.cache_resource
def init_engine():
    runbooks = load_runbooks("data/runbooks")
    index_matrix, chunk_records, model = build_index(runbooks)
    categories = sorted({rec["category"] for rec in chunk_records})
    return index_matrix, chunk_records, model, categories


def top_source_only(retrieved):
    best_score, best_rec = max(retrieved, key=lambda x: x[0])
    return best_rec["runbook"], best_score


def detect_category(question: str, available_categories: list[str]) -> str:
    """
    Heuristic category detection. Simple, explainable, and good enough for a prototype.
    Returns "all" if unsure.
    """
    q = question.lower().strip()

    # Keyword map -> category (update to match your runbook naming conventions)
    patterns = [
        ("auth", [r"\blogin\b", r"\bauth\b", r"\btoken\b", r"\b401\b", r"\b403\b", r"\bsso\b", r"\boauth\b"]),
        ("network", [r"\bdns\b", r"\btimeout\b", r"\bpacket\b", r"\blatency\b", r"\broute\b", r"\bconnect(ion)?\b",
                    r"\bnxdomain\b", r"\brefused\b", r"\btls\b", r"\bssl\b"]),
        ("loadbalancer", [r"\blb\b", r"\bload balancer\b", r"\bingress\b", r"\bproxy\b", r"\b502\b", r"\b503\b",
                          r"\b504\b", r"\bhealth check\b", r"\btarget\b"]),
        ("database", [r"\bdb\b", r"\bdatabase\b", r"\bpostgres\b", r"\bmysql\b", r"\bredis\b", r"\bcassandra\b",
                      r"\bdynamo\b", r"\bconnection pool\b", r"\bdeadlock\b"]),
    ]

    # Only consider categories you actually have indexed
    available = set(available_categories)

    for cat, regexes in patterns:
        if cat in available:
            for rgx in regexes:
                if re.search(rgx, q):
                    return cat

    return "all"


SUGGESTIONS = {
    "auth": [
        "First step to troubleshoot login failures",
        "What should I check when I see a spike in 401/403?",
        "Users can't sign in after a deployment — what do I do first?",
    ],
    "network": [
        "DNS resolution failing — first checks?",
        "Intermittent timeouts between services — what should I verify?",
        "How do I troubleshoot sudden latency increases?",
    ],
    "loadbalancer": [
        "Sudden spike in 502/503 from the load balancer — what to check first?",
        "Backends failing health checks — what are initial checks?",
        "504 timeouts at the LB — where do I start?",
    ],
    "database": [
        "Database connection timeout errors — first steps?",
        "Redis latency spike — what should I look at?",
        "Connection pool saturation — what do I check?",
    ],
    "all": [
        "First step to troubleshoot login issues",
        "What are the initial checks for 503 errors?",
        "What should I verify after a recent deployment?",
    ],
}


def render_suggestions(detected: str):
    st.markdown("**Suggestions**")
    suggestions = SUGGESTIONS.get(detected, SUGGESTIONS["all"])
    cols = st.columns(min(3, len(suggestions)))
    for i, s in enumerate(suggestions[:3]):
        if cols[i].button(s, use_container_width=True):
            st.session_state.question_prefill = s


# ---------- UI ----------
st.set_page_config(page_title="AI Runbook Reader", layout="wide")
st.title("AI Runbook Reader")
st.caption("Ask questions against your operational runbooks (local RAG prototype).")

index_matrix, chunk_records, model, categories = init_engine()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "question_prefill" not in st.session_state:
    st.session_state.question_prefill = ""

# Input
question_default = st.session_state.question_prefill
question = st.text_input(
    "Ask a question",
    value=question_default,
    placeholder="e.g., First step to troubleshoot login failures",
)
# Once used, clear prefill so it doesn't keep reappearing
st.session_state.question_prefill = ""

detected = detect_category(question, categories) if question.strip() else "all"

# Sidebar: history + detected category
st.sidebar.header("Auto-detected category")
st.sidebar.write(detected)

st.sidebar.header("Recent questions")
for q in st.session_state.history[-10:][::-1]:
    st.sidebar.write(f"• {q}")

# Optional override (kept tucked away)
with st.expander("Advanced (optional)"):
    override = st.selectbox("Override category", ["auto"] + ["all"] + categories, index=0)
    show_source = st.checkbox("Show source", value=True)
    short_answer = st.checkbox("Short answer", value=True)
    show_debug = st.checkbox("Show debug", value=False)

# Defaults if user never opens Advanced
if "show_source" not in locals():
    show_source = True
    short_answer = True
    show_debug = False
if "override" not in locals():
    override = "auto"

# Suggestions (based on detected category)
render_suggestions(detected)

# Action
if st.button("Get answer", type="primary", disabled=not question.strip()):
    st.session_state.history.append(question.strip())

    category_to_use = detected if override == "auto" else override

    retrieved = retrieve_with_scores(
        question,
        index_matrix,
        chunk_records,
        model,
        k=8,
        category=category_to_use,
    )

    top_score = retrieved[0][0] if retrieved else 0.0

    if top_score < CONFIDENCE_THRESHOLD:
        st.warning("Not found in runbook.")
    else:
        context_chunks = [rec["text"] for score, rec in retrieved]

        q_for_model = question
        if short_answer:
            q_for_model = f"{question}\n\nAnswer in 1–2 short sentences. No extra commentary."

        response = answer(q_for_model, context_chunks)
        response = response.replace("Not found in runbook.", "").strip()

        st.success(response)
        st.text_area("Answer (copy if needed)", value=response, height=120)

        if show_source:
            rb, s = top_source_only(retrieved)
            st.caption(f"Source: {rb} (score={s:.3f})")

        if show_debug:
            st.subheader("Top matches")
            for score, rec in retrieved:
                st.write(
                    f"- score={score:.3f} | {rec['category']} | {rec['runbook']} | chunk {rec['chunk_id']}"
                )

