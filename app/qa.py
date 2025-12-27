import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Smaller model = fewer "offloaded to disk" issues on Mac
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)


def answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks[:6])

    prompt = f"""You are an on-call runbook assistant.
Answer ONLY using the provided runbook context.
If the answer is not present in the context, respond exactly: Not found in runbook.

Runbook context:
{context}

Question:
{question}

Answer:"""

    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,  # deterministic
        )

    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    return text.strip()

