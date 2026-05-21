# DROPT Checkpoint Policy Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, optional policy adapter that can load the uploaded Guided-DiffFNO `.pth` checkpoint, produce deterministic policy outputs when the model is available, and fall back cleanly to the existing rule-based policy when it is not.

**Architecture:** Keep the existing LLM, retrieval, and eval pipeline unchanged. Add a small policy-loading layer that knows how to reconstruct the external DiffFNO/Diffusion modules, validate checkpoint compatibility, map BEAR state into the model’s expected input shape, and return a `PolicyResult` without ever letting the LLM write control actions.

**Tech Stack:** Python, PyTorch, BEAR adapter, existing `src/policies` interface, pytest.

---

### Task 1: Lock down the adapter contract with tests

**Files:**
- Modify: `tests/test_policies.py`
- Modify: `tests/test_agent_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from src.policies.dropt_adapter import DROPTCheckpointPolicy

def test_dropt_adapter_falls_back_when_checkpoint_is_missing():
    policy = DROPTCheckpointPolicy(model_path=None)
    result = policy.run(
        {
            "state_id": "episode_001_step_024",
            "zone_temperature": 29.0,
            "comfort_upper_bound": 26.0,
            "current_action": [0.0, 0.0],
        }
    )

    assert result.policy_name == "rule_based_fallback"
    assert result.baseline == "rule_based"
    assert len(result.recommended_action) == 2

def test_dropt_adapter_rejects_incompatible_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(b"not a pytorch checkpoint")

    with pytest.raises(RuntimeError, match="failed to load"):
        DROPTCheckpointPolicy(model_path=checkpoint)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py::test_dropt_adapter_falls_back_when_checkpoint_is_missing tests/test_policies.py::test_dropt_adapter_rejects_incompatible_checkpoint -q`
Expected: FAIL because `src/policies/dropt_adapter.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class DROPTCheckpointPolicy:
    def __init__(self, model_path=None):
        ...
    def run(self, state):
        return run_rule_based_policy(state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_policies.py::test_dropt_adapter_falls_back_when_checkpoint_is_missing tests/test_policies.py::test_dropt_adapter_rejects_incompatible_checkpoint -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_policies.py src/policies/dropt_adapter.py
git commit -m "feat: add dropt checkpoint adapter contract"
```

### Task 2: Implement the local Guided-DiffFNO inference adapter

**Files:**
- Create: `src/policies/dropt_adapter.py`
- Modify: `src/policies/__init__.py`
- Modify: `tests/test_policies.py`

- [ ] **Step 1: Write the failing test**

```python
from src.policies.dropt_adapter import DROPTCheckpointPolicy

def test_dropt_adapter_loads_checkpoint_and_returns_policy_result():
    policy = DROPTCheckpointPolicy(model_path="policy_best_fno_guided.pth")
    result = policy.run(
        {
            "state_id": "episode_001_step_024",
            "zone_temperature": 29.0,
            "comfort_upper_bound": 26.0,
            "current_action": [0.0, 0.0],
        }
    )

    assert result.input_state_id == "episode_001_step_024"
    assert result.policy_name.startswith("guided_diffno")
    assert len(result.recommended_action) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_policies.py::test_dropt_adapter_loads_checkpoint_and_returns_policy_result -q`
Expected: FAIL until the adapter can reconstruct the external model structure and load the checkpoint.

- [ ] **Step 3: Write minimal implementation**

```python
class DROPTCheckpointPolicy:
    def __init__(self, model_path=None, fallback_policy=None, device="cpu"):
        self.fallback_policy = fallback_policy or run_rule_based_policy
        self.model_path = Path(model_path) if model_path else None
        self._backend = self._try_build_backend()

    def run(self, state):
        if self._backend is None:
            return self.fallback_policy(state)
        action = self._backend.predict(state)
        return PolicyResult(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_policies.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/policies/dropt_adapter.py src/policies/__init__.py tests/test_policies.py
git commit -m "feat: wire guided diffusion checkpoint policy adapter"
```

### Task 3: Expose the adapter in the demo and docs

**Files:**
- Modify: `src/api/demo_factory.py`
- Modify: `README.md`
- Modify: `docs/system_design.md`
- Modify: `docs/stage_1_handoff.md`

- [ ] **Step 1: Write the failing test**

```python
def test_demo_orchestrator_can_use_dropt_checkpoint_when_present(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_app.py::test_demo_orchestrator_can_use_dropt_checkpoint_when_present -q`
Expected: FAIL until `build_demo_orchestrator()` can choose the new adapter.

- [ ] **Step 3: Write minimal implementation**

```python
policy_runner = build_dropt_policy_runner(project_root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_app.py tests/test_policies.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/demo_factory.py README.md docs/system_design.md docs/stage_1_handoff.md tests/test_api_app.py
git commit -m "docs: describe dropt checkpoint policy integration"
```

### Task 4: Re-verify the full project

**Files:**
- None

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: All tests pass.

- [ ] **Step 2: Run the eval pipeline**

Run: `python scripts/run_eval.py`
Expected: Regenerate report artifacts without changing deterministic baseline behavior.

- [ ] **Step 3: Check the generated outputs**

Confirm the report still matches the repo docs and that the new adapter is explicitly described as optional and offline.

