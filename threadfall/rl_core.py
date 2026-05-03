from __future__ import annotations

"""
Pure-state transition core for RL training.

This module intentionally avoids Streamlit and `st.session_state`.
It reuses content tables (sectors, mission configs) from `threadfall.game`,
but all mutation happens on an explicit state dict you pass in/out.
"""

from dataclasses import dataclass
import random
from typing import Any

from threadfall.game import MISSION_TYPES, SECTORS, SECTOR_TAGS, _default_stages


@dataclass(frozen=True)
class CoreConfig:
    max_commit_supplies: int = 2


def initial_state(*, seed: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    sector_control = {
        sector: {
            "controller": "Neutral",
            "stability": rng.randint(45, 70),
            "heat": rng.randint(10, 30),
        }
        for sector in SECTORS
    }
    return {
        "rng_seed": seed,
        "world": {"cycle": 1, "intel": 0, "supplies": 6, "threat_clock": 18, "sector_stability": 62},
        "sector_control": sector_control,
        "inventory_artifacts": 0,
    }


def _bucket(value: int, *, low: int, high: int) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def observation(state: dict[str, Any]) -> dict[str, str]:
    world = state["world"]
    return {
        "threat": _bucket(int(world["threat_clock"]), low=25, high=70),
        "supplies": _bucket(int(world["supplies"]), low=2, high=6),
        "stability": _bucket(int(world["sector_stability"]), low=25, high=70),
    }


def build_mission(
    state: dict[str, Any],
    *,
    squad_name: str,
    mission_type: str,
    sector: str | None = None,
) -> dict[str, Any]:
    rng = random.Random(state.get("rng_seed"))
    cfg = MISSION_TYPES.get(mission_type, MISSION_TYPES["standard"])
    sector = sector or rng.choice(SECTORS)
    sector_tags = SECTOR_TAGS.get(sector, [])

    stability_low, stability_high = cfg["stability_cost_range"]
    intel_low, intel_high = cfg["reward_intel_range"]
    world = state["world"]

    objective = rng.choice(list(cfg.get("objectives", [])) or [])
    if not objective:
        objective = "Unknown objective"

    return {
        "title": f"{sector} // {objective}",
        "sector": sector,
        "sector_tags": sector_tags,
        "objective": objective,
        "difficulty": rng.choice(["Low", "Elevated", "Severe"]),
        "stability_cost": rng.randint(int(stability_low), int(stability_high)),
        "reward_intel": rng.randint(int(intel_low), int(intel_high)) + int(world.get("cycle", 1)) + int(cfg.get("intel_bonus", 0)),
        "mission_type": mission_type,
        "stages": _default_stages(),
        "artifact_chance": float(cfg.get("artifact_chance", 0.55)),
        "heat_delta": int(cfg.get("heat_delta", 8)),
        "threat_delta_success": cfg.get("threat_delta_success", (-2, -1)),
        "threat_delta_fail": cfg.get("threat_delta_fail", (3, 7)),
        "stability_bonus_success": int(cfg.get("stability_bonus_success", 0)),
        "stage_mods": dict(cfg.get("stage_mods", {})),
    }


def resolve_mission(
    state: dict[str, Any],
    *,
    mission: dict[str, Any],
    committed_supplies: int = 0,
    advance_cycle: bool = True,
    cfg: CoreConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = cfg or CoreConfig()
    # Copy-on-write
    state = {
        **state,
        "world": dict(state["world"]),
        "sector_control": {k: dict(v) for k, v in state["sector_control"].items()},
    }

    world = state["world"]
    sector_state = state["sector_control"][mission["sector"]]

    committed = max(0, min(int(committed_supplies), int(world["supplies"]), cfg.max_commit_supplies))

    difficulty_weight = {"Low": 0.85, "Elevated": 0.7, "Severe": 0.52, "Terminal": 0.35}
    base_success = difficulty_weight.get(mission.get("difficulty", "Elevated"), 0.7)
    base_success += min(int(world["intel"]), 30) / 100.0
    base_success += min(int(state.get("inventory_artifacts", 0)), 6) * 0.025
    base_success -= min(max(int(sector_state.get("heat", 0)), 0), 100) / 200.0
    base_success += committed * 0.05
    base_success = max(0.06, min(base_success, 0.95))

    rng = random.Random(state.get("rng_seed"))
    stages = mission.get("stages") or _default_stages()
    stage_mods = dict(mission.get("stage_mods", {}))
    stage_results: list[dict[str, Any]] = []
    staged_success = True
    for stage in stages:
        name = str(stage.get("name", "Stage"))
        mod = float(stage_mods.get(name, -0.02))
        chance = max(0.05, min(0.97, base_success + mod))
        ok = rng.random() < chance
        stage_results.append({"stage": name, "success": ok, "chance": round(chance, 3)})
        if not ok:
            staged_success = False
            break

    if advance_cycle:
        world["cycle"] += 1
    world["supplies"] = max(0, int(world["supplies"]) - 1 - committed)
    world["sector_stability"] = max(0, int(world["sector_stability"]) - int(mission["stability_cost"]))
    sector_state["heat"] = min(100, int(sector_state["heat"]) + int(mission.get("heat_delta", 8)))
    sector_state["stability"] = max(0, int(sector_state["stability"]) - int(mission["stability_cost"]))

    artifact_found = 0
    if staged_success:
        world["intel"] = int(world["intel"]) + int(mission["reward_intel"])
        td0, td1 = mission.get("threat_delta_success", (-2, -1))
        world["threat_clock"] = max(0, min(100, int(world["threat_clock"]) + rng.randint(int(td0), int(td1))))
        stability_bonus = int(mission.get("stability_bonus_success", 0))
        if stability_bonus:
            world["sector_stability"] = min(100, int(world["sector_stability"]) + stability_bonus)
            sector_state["stability"] = min(100, int(sector_state["stability"]) + stability_bonus)
        if rng.random() < float(mission.get("artifact_chance", 0.55)):
            state["inventory_artifacts"] = int(state.get("inventory_artifacts", 0)) + 1
            artifact_found = 1
        outcome = "Success"
    else:
        td0, td1 = mission.get("threat_delta_fail", (3, 7))
        world["threat_clock"] = max(0, min(100, int(world["threat_clock"]) + rng.randint(int(td0), int(td1))))
        world["sector_stability"] = max(0, int(world["sector_stability"]) - 4)
        outcome = "Compromised"

    info = {
        "outcome": outcome,
        "stage_results": stage_results,
        "reward_intel": int(mission["reward_intel"]),
        "artifact_found": artifact_found,
        "committed_supplies": committed,
        "mission_type": mission.get("mission_type", "standard"),
        "sector": mission["sector"],
    }
    return state, info


def reward_from_info(info: dict[str, Any]) -> float:
    r = float(info.get("reward_intel", 0))
    r += 20.0 * float(info.get("artifact_found", 0))
    r -= 2.0 * float(info.get("committed_supplies", 0))
    if info.get("outcome") != "Success":
        r -= 4.0
    return r

