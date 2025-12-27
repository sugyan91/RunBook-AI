import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_TEMPLATE = """You are an SRE assistant.
Answer ONLY using the runbook context below.
If the answer is not present, say "Not found in runbook."

Context:
{context}

Question:
{question}

Answer:
"""

def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # macOS Metal (MPS) if available, else CPU
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"

    # Use dtype (not torch_dtype) to avoid deprecation warning
    dtype = torch.float16 if use_mps else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,   # "mps" or "cpu"
    )

    return tokenizer, model

_TOKENIZER, _MODEL = _load_model()

def answer(question: str, context_chunks: list[str]) -> str:
    prompt = PROMPT_TEMPLATE.format(
        context="\n".join(context_chunks),
        question=question
    )

    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,   # deterministic, cleaner
        )

    text = _TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()

