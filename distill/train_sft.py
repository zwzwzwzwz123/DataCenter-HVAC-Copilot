"""Stage 2: supervised fine-tuning (SFT) of the planner student model.

Fine-tunes a small instruct model (Qwen2.5-1.5B-Instruct by default) with
QLoRA so it reproduces the route planner's decisions, using the hand-labeled
+ teacher-generated ``{messages, completion}`` data from stage 1.

Design choices (see distillation_plan.md, stage 2):
- **Chat template alignment**: prompts are rendered with the model's own
  ``tokenizer.apply_chat_template`` so the student trains on exactly the format
  it will see at inference. The label is the ``completion`` JSON rendered as the
  assistant turn.
- **Completion-only loss**: loss is masked to the assistant response via TRL's
  ``DataCollatorForCompletionOnlyLM``, so the model is not trained to reproduce
  the (fixed) system/user prompt.
- **QLoRA (4-bit)** by default for single 12-16G GPU; ``--no-quantize`` for
  full-precision LoRA when memory allows.
- **Model is configurable** (``--model``) so the same script trains 1.5B now
  and 3B/7B later without edits.
- After training, validation-set **plan legality** is measured with the SAME
  ``validate_plan_steps`` guard the live system uses (stage 2 acceptance).

Training deps are optional and GPU-oriented; install them only where you train::

    pip install -e '.[train]'

Example (on a rented GPU box, from repo root)::

    python -m distill.train_sft \\
        --train distill/data/gold_sft_train.jsonl \\
        --val   distill/data/gold_sft_val.jsonl \\
        --output distill/checkpoints/sft-qwen1.5b

Run as a module (``-m``) from the repo root so ``src`` is importable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

# NOTE: heavy ML deps (torch/transformers/trl/peft) are imported lazily inside
# functions so this file stays importable (and lint-checkable) on machines
# without a GPU or the [train] extra installed.


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Qwen chat responses are wrapped as <|im_start|>assistant\n ... ; this marker
# is what the completion-only collator uses to locate the label span.
ASSISTANT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    train_path: str = "distill/data/gold_sft_train.jsonl"
    val_path: str = "distill/data/gold_sft_val.jsonl"
    output_dir: str = "distill/checkpoints/sft-qwen1.5b"
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    quantize: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Attention/MLP projections common to Llama/Qwen-style decoders.
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    seed: int = 42


def _require_deps():
    """Import training deps, failing with an actionable message if missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
        import peft  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"missing training dependency ({exc.name}). "
            "Install the training extra on your GPU box:\n"
            "    pip install -e '.[train]'"
        ) from exc


def load_records(path: str) -> list[dict]:
    """Load {messages, completion} rows from a stage-1 SFT jsonl file."""
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no samples found in {path}")
    return records


def build_text(record: dict, tokenizer) -> str:
    """Render one sample to a full chat string ending with the assistant label.

    Uses the tokenizer's own chat template so the student sees exactly the
    inference-time format. The completion (assistant turn) is appended so the
    completion-only collator can mask everything before it.
    """
    messages = list(record["messages"]) + [
        {"role": "assistant", "content": record["completion"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def measure_val_legality(model, tokenizer, val_records: list[dict], max_new_tokens: int = 256) -> dict:
    """Generate plans for val questions and score legality with the live guard.

    This is the stage-2 acceptance metric: the fraction of generated plans that
    parse and pass the guard via ``_decision_from_llm_payload`` — the exact same
    function the online ``LLMRoutePlanner`` uses to turn model output into a
    validated ``PlanDecision``. Also reports how often the generated plan's route
    sequence exactly matches the gold label.
    """
    import torch

    from src.agent.planner import _decision_from_llm_payload

    def _routes(payload: str) -> list[str] | None:
        try:
            decision = _decision_from_llm_payload(content=payload, planner="student")
            return [s.route for s in decision.steps]
        except Exception:
            return None

    model.eval()
    total = len(val_records)
    legal = 0
    exact = 0
    for rec in val_records:
        prompt = tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        gen_routes = _routes(gen)
        if gen_routes is None:
            continue
        legal += 1
        gold_routes = _routes(rec["completion"])
        if gold_routes is not None and gen_routes == gold_routes:
            exact += 1
    return {
        "val_size": total,
        "legal_rate": round(legal / total, 4) if total else 0.0,
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
    }


def train(cfg: TrainConfig) -> None:
    _require_deps()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_targets,
    )

    train_records = load_records(cfg.train_path)
    val_records = load_records(cfg.val_path)
    train_ds = Dataset.from_dict(
        {"text": [build_text(r, tokenizer) for r in train_records]}
    )
    val_ds = Dataset.from_dict(
        {"text": [build_text(r, tokenizer) for r in val_records]}
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=ASSISTANT_RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=cfg.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Stage-2 acceptance: measure plan legality on the validation set.
    metrics = measure_val_legality(trainer.model, tokenizer, val_records)
    print(
        f"[eval] val legality={metrics['legal_rate']:.2%} "
        f"exact_match={metrics['exact_match_rate']:.2%} (n={metrics['val_size']})"
    )

    card = {
        "model": cfg.model,
        "quantize_4bit": cfg.quantize,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "effective_batch": cfg.batch_size * cfg.grad_accum,
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "targets": cfg.lora_targets},
        "train_size": len(train_records),
        "val_size": len(val_records),
        **metrics,
    }
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft_train_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[write] {out}/sft_train_card.json")
    print(f"[done] LoRA adapter saved to {cfg.output_dir}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="base instruct model (HF id or local path)")
    p.add_argument("--train", dest="train_path", default="distill/data/gold_sft_train.jsonl")
    p.add_argument("--val", dest="val_path", default="distill/data/gold_sft_val.jsonl")
    p.add_argument("--output", dest="output_dir", default="distill/checkpoints/sft-qwen1.5b")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--no-quantize", dest="quantize", action="store_false", help="full-precision LoRA instead of 4-bit QLoRA")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return TrainConfig(
        model=a.model,
        train_path=a.train_path,
        val_path=a.val_path,
        output_dir=a.output_dir,
        epochs=a.epochs,
        lr=a.lr,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        max_seq_len=a.max_seq_len,
        quantize=a.quantize,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        seed=a.seed,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
