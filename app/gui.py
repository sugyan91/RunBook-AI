# --- FIX STREAMLIT IMPORT PATH + WORKING DIRECTORY (DO NOT REMOVE) ---
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force CWD to project root so relative paths work in Streamlit
os.chdir(ROOT)
# ------------------------------------------------------

import re
import difflib
from collections import defaultdict

import streamlit as st

from app.ingest import load_runbooks, build_index
from app.retriever import retrieve_with_scores
from app.qa import answer

CONFIDENCE_THRESHOLD = 0.20


def get_runbook_signature(runbooks_dir: Path):
    files = sorted([p for p in runbooks_dir.glob("*") if p.suffix in (".md", ".txt")])
    return tuple((p.name, int(p.stat().st_mtime)) for p in files)


def extract_h1_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def best_runbook_match(question: str, runbooks: list[dict], min_ratio: float = 0.55):
    """
    If user mentions a runbook by name/title, route directly.
    Uses fuzzy match against:
      - filename (without extension)
      - markdown H1 title (if present)
      - full filename
    Returns: (matched_runbook_name, match_ratio) or (None, 0.0)
    """
    qn = normalize(question)

    best_name = None
    best_ratio = 0.0

    for rb in runbooks:
        fname = rb["name"]
        base = os.path.splitext(fname)[0]
        h1 = extract_h1_title(rb["text"])

        candidates = [fname, base, h1]
        for c in candidates:
            cn = normalize(c)
            if not cn:
                continue
            ratio = difflib.SequenceMatcher(None, qn, cn).ratio()

            # Also boost if candidate words appear inside question (common for "X - Y" titles)
            if cn and cn in qn:
                ratio = max(ratio, 0.85)

            if ratio > best_ratio:
                best_ratio = ratio
                best_name = fname

    if best_ratio >= min_ratio:
        return best_name, best_ratio
    return None, 0.0


def extract_sections(text: str) -> set[str]:
    sections = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            sections.add(line[3:].strip().lower())
    return sections


def build_suggestions_from_runbooks(runbooks, max_per_category=3):
    sug = defaultdict(list)

    for rb in runbooks:
        title = extract_h1_title(rb["text"]) or os.path.splitext(rb["name"])[0].replace("-", " ")
        secs = extract_sections(rb["text"])
        cat = rb.get("category", "general")

        candidates = [
            f"What are the initial checks for {title}?",
            f"First step to troubleshoot {title}",
            f"When should I escalate for {title}?",
        ]
        if "remediation" in secs:
            candidates.append(f"How do I remediate {title}?")
        if "symptoms" in secs:
            candidates.append(f"What are the symptoms of {title}?")

        for c in candidates:
            if len(sug[cat]) < max_per_category and c not in sug[cat]:
                sug[cat].append(c)

    sug["all"] = []
    for cat in sorted(sug.keys()):
        for s in sug[cat]:
            if len(sug["all"]) < 6 and s not in sug["all"]:
                sug["all"].append(s)

    return dict(sug)


def detect_category(question: str, available_categories: list[str]) -> str:
    q = question.lower().strip()
    available = set(available_categories)

    patterns = [
        ("auth", [r"\blogin\b", r"\bauth\b", r"\btoken\b", r"\b401\b", r"\b403\b", r"\bsso\b", r"\boauth\b"]),
        ("network", [r"\bdns\b", r"\btimeout\b", r"\blatency\b", r"\broute\b",
                    r"\bconnect(ion)?\b", r"\bnxdomain\b", r"\btls\b", r"\bssl\b"]),
        ("loadbalancer", [r"\blb\b", r"\bload balancer\b", r"\bingress\b", r"\bproxy\b",
                          r"\b502\b", r"\b503\b", r"\b504\b", r"\bhealth check\b"]),
        ("database", [r"\bdb\b", r"\bdatabase\b", r"\bpostgres\b", r"\bmysql\b", r"\bredis\b",
                      r"\bcassandra\b", r"\bdynamo\b", r"\bconnection pool\b"]),
    ]

    for cat, regexes in patterns:
        if cat in available:
            for rgx in regexes:
                if re.search(rgx, q):
                    return cat
    return "all"


def top_source_only(retrieved):
    best_score, best_rec = max(retrieved, key=lambda x: x[0])
    return best_rec["runbook"], best_score


@st.cache_resource
def init_engine(_signature: tuple, runbooks_dir_str: str):
    runbooks = load_runbooks(runbooks_dir_str)
    index_matrix, chunk_records, model = build_index(runbooks)
    categories = sorted({rec["category"] for rec in chunk_records})
    suggestions_map = build_suggestions_from_runbooks(runbooks, max_per_category=3)
    return runbooks, index_matrix, chunk_records, model, categories, suggestions_map


# ---------- UI ----------
st.set_page_config(page_title="Runbook AI", layout="wide")
st.title("Runbook AI")
st.caption("Ask questions against your operational runbooks (local RAG prototype).")

RUNBOOKS_DIR = ROOT / "data" / "runbooks"
sig = get_runbook_signature(RUNBOOKS_DIR)

runbooks, index_matrix, chunk_records, model, categories, suggestions_map = init_engine(sig, str(RUNBOOKS_DIR))

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "question_prefill" not in st.session_state:
    st.session_state.question_prefill = ""

# Sidebar sanity
st.sidebar.caption(f"runbooks={len(runbooks)}  chunks={len(chunk_records)}")

question = st.text_input(
    "Ask a question",
    value=st.session_state.question_prefill,
    placeholder="e.g., first step to troubleshoot login issues",
)
st.session_state.question_prefill = ""

detected = detect_category(question, categories) if question.strip() else "all"

st.sidebar.header("Detected category")
st.sidebar.write(detected)

st.sidebar.header("Recent questions")
for q in st.session_state.history[-10:][::-1]:
    st.sidebar.write(f"• {q}")

# Suggestions derived from actual runbooks
st.markdown("**Suggested questions (based on your runbooks)**")
use_cat = detected if detected in suggestions_map else "all"
sugs = suggestions_map.get(use_cat, suggestions_map.get("all", []))

if sugs:
    cols = st.columns(min(3, len(sugs)))
    for i, s in enumerate(sugs[:3]):
        if cols[i].button(s, use_container_width=True):
            st.session_state.question_prefill = s
            st.rerun()

with st.expander("Advanced (optional)"):
    show_source = st.checkbox("Show source", value=True)
    short_answer = st.checkbox("Short answer", value=True)
    show_debug = st.checkbox("Show debug", value=True)

if st.button("Get answer", type="primary", disabled=not question.strip()):
    st.session_state.history.append(question.strip())

    # 0) If user appears to name a runbook, route directly (bypass similarity gate)
    matched_rb, rb_ratio = best_runbook_match(question, runbooks, min_ratio=0.55)

    if matched_rb:
        # Use only chunks from that runbook (still grounded, deterministic)
        rb_chunks = [rec["text"] for rec in chunk_records if rec["runbook"] == matched_rb]

        q_for_model = question
        if short_answer:
            q_for_model = f"{question}\n\nAnswer in 1–2 short sentences. No extra commentary."

        resp = answer(q_for_model, rb_chunks)
        resp = resp.replace("Not found in runbook.", "").strip()

        st.success(resp)
        st.text_area("Answer (copy if needed)", value=resp, height=120)

        if show_source:
            st.caption(f"Source: {matched_rb} (routed, match={rb_ratio:.3f})")

        if show_debug:
            st.caption(f"debug: routed_runbook={matched_rb}, match_ratio={rb_ratio:.3f}, chunks_used={len(rb_chunks)}")

    else:
        # 1) Normal semantic retrieval: detected category -> fallback to all
        category_used = detected

        retrieved = retrieve_with_scores(
            question, index_matrix, chunk_records, model, k=8, category=category_used
        )
        top_score = retrieved[0][0] if retrieved else 0.0

        if top_score < CONFIDENCE_THRESHOLD and category_used != "all":
            retrieved = retrieve_with_scores(
                question, index_matrix, chunk_records, model, k=8, category="all"
            )
            top_score = retrieved[0][0] if retrieved else 0.0
            category_used = "all"

        if show_debug:
            st.caption(f"debug: category_used={category_used}, top_score={top_score:.3f}")

        if top_score < CONFIDENCE_THRESHOLD:
            st.warning("Not found in runbook.")
        else:
            context_chunks = [rec["text"] for score, rec in retrieved]

            q_for_model = question
            if short_answer:
                q_for_model = f"{question}\n\nAnswer in 1–2 short sentences. No extra commentary."

            resp = answer(q_for_model, context_chunks)
            resp = resp.replace("Not found in runbook.", "").strip()

            st.success(resp)
            st.text_area("Answer (copy if needed)", value=resp, height=120)

            if show_source:
                rb, s = top_source_only(retrieved)
                st.caption(f"Source: {rb} (score={s:.3f})")

