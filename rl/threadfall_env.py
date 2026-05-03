from __future__ import annotations

"""
Minimal Gymnasium wrapper (scaffolding) for Threadfall rival training.

This is intentionally lightweight and does NOT yet share the Streamlit session state.
For reliable RL training, we should refactor core game logic into a pure-state transition
module (no st.session_state). This file is a starting point for that refactor.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Gymnasium is required for RL(B). Install with: pip install -r requirements-rl.txt"
    ) from exc


@dataclass
class EnvConfig:
    max_steps: int = 80
    seed: int | None = None


class ThreadfallRivalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()

        # Action = mission type index: standard/survey/extraction/containment
        self.action_space = spaces.Discrete(4)

        # Observation: threat_clock, supplies, stability, intel, artifacts
        self.observation_space = spaces.Box(low=0.0, high=400.0, shape=(5,), dtype=np.float32)

        self._step = 0
        self._state: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        from threadfall.rl_core import initial_state

        self._step = 0
        used_seed = seed if seed is not None else self.cfg.seed
        self._state = initial_state(seed=used_seed)
        return self._obs(), {}

    def step(self, action: int):
        from threadfall.rl_core import build_mission, resolve_mission, reward_from_info

        self._step += 1
        action = int(action)
        mission_type = ["standard", "survey", "extraction", "containment"][action]
        mission = build_mission(self._state, squad_name="Rival Unit", mission_type=mission_type)
        self._state, info = resolve_mission(
            self._state,
            mission=mission,
            committed_supplies=0,
            advance_cycle=True,
        )

        r = reward_from_info(info)
        terminated = bool(self._state["world"]["intel"] >= 200 and self._state.get("inventory_artifacts", 0) >= 8)
        truncated = self._step >= self.cfg.max_steps
        return self._obs(), float(r), terminated, truncated, info

    def _obs(self):
        w = self._state["world"]
        return np.array(
            [
                float(w["threat_clock"]),
                float(w["supplies"]),
                float(w["sector_stability"]),
                float(w["intel"]),
                float(self._state.get("inventory_artifacts", 0)),
            ],
            dtype=np.float32,
        )

