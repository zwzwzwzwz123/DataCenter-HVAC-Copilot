import json
from pathlib import Path

import pytest

from src.policies.dropt_adapter import DROPTCheckpointPolicy
from src.policies.diffusion_adapter import DiffusionPolicyAdapter
from src.policies.mpc_like import run_mpc_like_policy
from src.policies.offline_replay import OfflineReplayPolicy
from src.policies.rule_based import run_rule_based_policy


def test_rule_based_policy_recommends_cooling_when_temperature_is_high():
    result = run_rule_based_policy(
        {
            "state_id": "episode_001_step_002",
            "zone_temperature": 29.0,
            "comfort_upper_bound": 26.0,
            "current_action": [0.0, 0.0],
        }
    )

    assert result.policy_name == "rule_based"
    assert result.input_state_id == "episode_001_step_002"
    assert result.recommended_action == [-0.1, -0.1]
    assert "rule-based" in result.notes.lower()


def test_mpc_like_policy_returns_structured_tradeoff_estimate():
    result = run_mpc_like_policy(
        {
            "state_id": "episode_001_step_003",
            "zone_temperature": 25.0,
            "comfort_upper_bound": 26.0,
            "hvac_power": 120.0,
            "current_action": [0.2, 0.2],
        },
        horizon=4,
    )

    assert result.policy_name == "mpc_like"
    assert result.input_state_id == "episode_001_step_003"
    assert len(result.recommended_action) == 2
    assert result.estimated_energy == 456.0
    assert result.estimated_comfort_violations == 0.0


def test_diffusion_policy_adapter_fails_explicitly_without_backend():
    adapter = DiffusionPolicyAdapter(model_path=None)

    with pytest.raises(NotImplementedError, match="Diffusion policy backend is not configured"):
        adapter.run({"state_id": "episode_001_step_004"})


def test_dropt_checkpoint_policy_falls_back_without_checkpoint():
    policy = DROPTCheckpointPolicy(model_path=None)

    result = policy.run(
        {
            "state_id": "episode_001_step_024",
            "zone_temperature": 29.0,
            "comfort_upper_bound": 26.0,
            "current_action": [0.0, 0.0],
        }
    )

    assert result.policy_name == "dropt_checkpoint_fallback"
    assert result.baseline == "rule_based"
    assert len(result.recommended_action) == 2


def test_dropt_checkpoint_policy_loads_real_checkpoint():
    checkpoint_path = Path("policy_best_fno_guided.pth")
    assert checkpoint_path.exists()

    policy = DROPTCheckpointPolicy(model_path=checkpoint_path)
    result = policy.run(
        {
            "state_id": "episode_001_step_024",
            "bear_state_vector": [
                23.0,
                23.4,
                23.8,
                24.2,
                24.5,
                24.8,
                31.0,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                18.0,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
            ],
            "current_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    assert result.policy_name == "dropt_guided_diffno_checkpoint"
    assert result.input_state_id == "episode_001_step_024"
    assert len(result.recommended_action) == 6


def test_offline_replay_policy_reads_saved_result(tmp_path):
    replay_file = tmp_path / "offline_policy_results.json"
    replay_file.write_text(
        json.dumps(
            [
                {
                    "policy_name": "guided_diffno_offline_replay",
                    "input_state_id": "episode_001_step_024",
                    "recommended_action": [-0.2, -0.1],
                    "estimated_energy": 901.3,
                    "estimated_comfort_violations": 0.886,
                    "mean_action_change": 0.0402,
                    "baseline": "diffusion_mlp",
                    "notes": "Values come from offline replay.",
                }
            ]
        ),
        encoding="utf-8",
    )

    policy = OfflineReplayPolicy(replay_file)
    result = policy.run({"state_id": "episode_001_step_024"})

    assert result.policy_name == "guided_diffno_offline_replay"
    assert result.recommended_action == [-0.2, -0.1]
    assert result.baseline == "diffusion_mlp"

