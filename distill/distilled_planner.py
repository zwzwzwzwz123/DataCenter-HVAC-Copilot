"""Stage 4: wrap the SFT-distilled model as a RoutePlanner.

`DistilledRoutePlanner` loads the base model + trained LoRA adapter and plans
exactly like the online `LLMRoutePlanner`, but runs locally instead of calling
a cloud API:

    build_planner_messages(question)      # same prompt construction as online
      -> tokenizer.apply_chat_template
      -> model.generate
      -> _decision_from_llm_payload(...)  # same parser/guard as online
      -> PlanDecision   (falls back to deterministic on any failure)

Because prompt construction and output parsing reuse the *same* functions as
the production planner, the distilled model plugs into the existing evaluation
pipeline with no format drift.

Heavy deps (torch/transformers/peft) are imported lazily so this module stays
importable on machines without a GPU or the `.[train]` extra. Loading the model
needs those deps + the base weights (~3GB); the deterministic fallback path
needs none of them.
"""

from __future__ import annotations

from typing import Any

from src.agent.planner import (
    DeterministicRoutePlanner,
    PlanDecision,
    PlanStep,
    RoutePlanner,
    SUPPORTED_ROUTES,
    _decision_from_llm_payload,
    build_planner_messages,
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class DistilledRoutePlanner:
    """Local SFT/LoRA planner implementing the RoutePlanner protocol."""

    name = "distilled_sft"

    def __init__(
        self,
        adapter_dir: str,
        base_model: str = DEFAULT_BASE_MODEL,
        max_new_tokens: int = 256,
        quantize: bool = True,
        fallback: RoutePlanner | None = None,
    ) -> None:
        self.adapter_dir = adapter_dir
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.quantize = quantize
        self.fallback = fallback or DeterministicRoutePlanner()
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazily load base model + LoRA adapter on first use."""
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-pad for batched decode: generation must continue from the last
        # real token of each row, so padding has to sit on the left.
        tokenizer.padding_side = "left"

        quant_config = None
        if self.quantize:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, self.adapter_dir)
        # Fold the LoRA adapter into the base weights so inference runs a single
        # fused matmul per layer instead of base + adapter. Only possible on the
        # unquantized path (4-bit weights can't be merged in place); big speedup
        # for the eval loop. Falls back gracefully if merge is unavailable.
        if not self.quantize:
            try:
                model = model.merge_and_unload()
            except Exception:  # pragma: no cover - defensive, keep adapter form
                pass
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def _generate(self, question: str, conversation_context: dict[str, Any] | None) -> str:
        import torch

        messages = build_planner_messages(question, conversation_context)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def plan(
        self,
        question: str,
        task_type: str | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> PlanDecision:
        # Mirror LLMRoutePlanner: an explicit eval task_type short-circuits to
        # the deterministic router (single-route case), so distillation is only
        # exercised on the free-form / compound path.
        if task_type in SUPPORTED_ROUTES:
            return self.fallback.plan(question, task_type=task_type)

        try:
            self._ensure_loaded()
            content = self._generate(question, conversation_context)
            return _decision_from_llm_payload(
                content=content,
                planner=f"distilled:{self.adapter_dir}",
            )
        except Exception as exc:
            fb = self.fallback.plan(question, task_type=task_type)
            return PlanDecision(
                steps=[
                    PlanStep(
                        route=s.route,
                        reason=f"distilled planning failed ({exc}); {s.reason}",
                        tool=s.tool,
                        metric_name=s.metric_name,
                        zone_id=s.zone_id,
                        time_window=s.time_window,
                    )
                    for s in fb.steps
                ],
                planner=fb.planner,
                confidence=fb.confidence,
                fallback_used=True,
            )

    def _fallback_decision(self, question: str, exc: Exception) -> PlanDecision:
        """Wrap the deterministic fallback, tagging the failure reason."""
        fb = self.fallback.plan(question)
        return PlanDecision(
            steps=[
                PlanStep(
                    route=s.route,
                    reason=f"distilled planning failed ({exc}); {s.reason}",
                    tool=s.tool,
                    metric_name=s.metric_name,
                    zone_id=s.zone_id,
                    time_window=s.time_window,
                )
                for s in fb.steps
            ],
            planner=fb.planner,
            confidence=fb.confidence,
            fallback_used=True,
        )

    def plan_batch(
        self,
        questions: list[str],
        batch_size: int = 16,
        progress: Any = None,
    ) -> list[PlanDecision]:
        """Plan many questions with batched GPU decode.

        Equivalent to calling ``plan(q)`` per question but generates
        ``batch_size`` prompts in one ``model.generate`` call, which is far
        faster than the one-at-a-time loop on GPU (decode is bandwidth-bound;
        batching amortises the per-step overhead). Prompt construction and
        output parsing reuse the same online functions, so results match the
        single-item path. Any per-row failure falls back to deterministic.

        ``progress`` is an optional callable ``(done, total) -> None`` for
        incremental reporting so a long run is observable.
        """
        import torch

        self._ensure_loaded()
        total = len(questions)
        decisions: list[PlanDecision] = []
        for start in range(0, total, batch_size):
            chunk = questions[start : start + batch_size]
            prompts = [
                self._tokenizer.apply_chat_template(
                    build_planner_messages(q, None),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in chunk
            ]
            try:
                inputs = self._tokenizer(
                    prompts, return_tensors="pt", padding=True
                ).to(self._model.device)
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id
                        or self._tokenizer.eos_token_id,
                    )
                gen = out[:, inputs["input_ids"].shape[1] :]
                texts = self._tokenizer.batch_decode(gen, skip_special_tokens=True)
            except Exception as exc:  # whole-batch failure -> per-row fallback
                for q in chunk:
                    decisions.append(self._fallback_decision(q, exc))
                if progress:
                    progress(len(decisions), total)
                continue

            for q, text in zip(chunk, texts):
                try:
                    decisions.append(
                        _decision_from_llm_payload(
                            content=text.strip(),
                            planner=f"distilled:{self.adapter_dir}",
                        )
                    )
                except Exception as exc:
                    decisions.append(self._fallback_decision(q, exc))
            if progress:
                progress(len(decisions), total)
        return decisions
