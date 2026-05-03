"""
Multi-agent scheduling + per-faction decision policies (MAS layer).

Factions still share one world state, but each has:
- an independent policy mode (LLM / heuristic / bandit)
- factor weights shaping bandit reward from expedition-style effect deltas
- turn-order scheduling (rotation / shuffle / fixed)
"""

from __future__ import annotations

import random
from typing import Any, Optional

import streamlit as st

from threadfall.llm import DecisionProposal, OllamaDecisionEngine

MAS_DEFAULT_POLICY = "bandit"


def _sanitize_bandit(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"counts": {}, "values": {}}
    counts = raw.get("counts")
    values = raw.get("values")
    out_counts: dict[str, Any] = {}
    out_values: dict[str, Any] = {}
    if isinstance(counts, dict):
        for ok, ov in counts.items():
            if not isinstance(ov, dict):
                continue
            inner: dict[str, int] = {}
            for ak, av in ov.items():
                try:
                    inner[str(ak)] = int(av)
                except (TypeError, ValueError):
                    continue
            out_counts[str(ok)] = inner
    if isinstance(values, dict):
        for ok, ov in values.items():
            if not isinstance(ov, dict):
                continue
            inner_v: dict[str, float] = {}
            for ak, av in ov.items():
                try:
                    inner_v[str(ak)] = float(av)
                except (TypeError, ValueError):
                    continue
            out_values[str(ok)] = inner_v
    return {"counts": out_counts, "values": out_values}


def _sanitize_agent_slot(slot: dict[str, Any]) -> dict[str, Any]:
    policy = str(slot.get("policy", MAS_DEFAULT_POLICY))
    if policy not in ("llm", "heuristic", "bandit"):
        policy = MAS_DEFAULT_POLICY
    try:
        epsilon = float(slot.get("epsilon", 0.12))
    except (TypeError, ValueError):
        epsilon = 0.12
    epsilon = max(0.0, min(0.95, epsilon))
    factors = slot.get("factors")
    if not isinstance(factors, dict):
        factors = {}
    factors = {str(k): float(v) for k, v in factors.items() if isinstance(v, (int, float))}
    return {
        "policy": policy,
        "epsilon": epsilon,
        "factors": factors,
        "bandit": _sanitize_bandit(slot.get("bandit")),
    }


def _ensure_agent_defaults(slot: dict[str, Any], interest: str) -> None:
    slot.setdefault("policy", MAS_DEFAULT_POLICY)
    if str(slot.get("policy")) not in ("llm", "heuristic", "bandit"):
        slot["policy"] = MAS_DEFAULT_POLICY
    try:
        slot["epsilon"] = max(0.0, min(0.95, float(slot.get("epsilon", 0.12))))
    except (TypeError, ValueError):
        slot["epsilon"] = 0.12
    if not isinstance(slot.get("factors"), dict) or not slot["factors"]:
        slot["factors"] = default_factors_for_interest(interest)
    slot.setdefault("bandit", {"counts": {}, "values": {}})
    if not isinstance(slot["bandit"], dict):
        slot["bandit"] = {"counts": {}, "values": {}}
    slot["bandit"].setdefault("counts", {})
    slot["bandit"].setdefault("values", {})


def merge_mas_from_save(loaded: Optional[dict[str, Any]], factions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a normalized MAS dict for session state or save files."""
    order = "rotation"
    rotation_idx = 0
    agents: dict[str, Any] = {}
    if isinstance(loaded, dict):
        o = str(loaded.get("order", "rotation"))
        if o in ("rotation", "shuffle", "fixed"):
            order = o
        try:
            rotation_idx = max(0, int(loaded.get("rotation_idx", 0)))
        except (TypeError, ValueError):
            rotation_idx = 0
        raw_agents = loaded.get("agents")
        if isinstance(raw_agents, dict):
            for name, slot in raw_agents.items():
                if isinstance(slot, dict):
                    agents[str(name)] = _sanitize_agent_slot(slot)
    for faction in factions:
        name = str(faction.get("name", ""))
        if not name:
            continue
        interest = str(faction.get("interest", ""))
        if name in agents:
            _ensure_agent_defaults(agents[name], interest)
        else:
            agents[name] = {
                "policy": MAS_DEFAULT_POLICY,
                "epsilon": 0.12,
                "factors": default_factors_for_interest(interest),
                "bandit": {"counts": {}, "values": {}},
            }
    return {"order": order, "rotation_idx": rotation_idx, "agents": agents}


def init_mas() -> None:
    """Merge disk/session MAS data with the current faction roster."""
    factions = st.session_state.get("factions", [])
    raw = st.session_state.get("mas")
    st.session_state["mas"] = merge_mas_from_save(raw if isinstance(raw, dict) else None, factions)


def default_factors_for_interest(interest: str) -> dict[str, float]:
    base = {
        "intel": 1.0,
        "leverage": 1.0,
        "threat": -0.35,
        "stability": 0.45,
        "supplies": 0.25,
        "pressure": -0.25,
    }
    if interest == "Artifact extraction":
        base["intel"] += 0.35
        base["leverage"] += 0.25
    elif interest == "Knowledge capture":
        base["intel"] += 0.45
        base["stability"] += 0.25
    else:
        base["leverage"] += 0.2
        base["pressure"] -= 0.15
    return base


def mas_state() -> dict[str, Any]:
    init_mas()
    return st.session_state["mas"]


def ordered_factions(factions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    f = list(factions)
    if not f:
        return f
    root = mas_state()
    order = str(root.get("order", "rotation"))
    if order == "shuffle":
        random.shuffle(f)
        return f
    if order == "fixed":
        return f
    idx = int(root.get("rotation_idx", 0)) % len(f)
    return f[idx:] + f[:idx]


def advance_turn_order(*, num_factions: int) -> None:
    root = mas_state()
    if str(root.get("order", "rotation")) != "rotation":
        return
    n = max(1, int(num_factions))
    root["rotation_idx"] = (int(root.get("rotation_idx", 0)) + 1) % n


def _bucket(value: int, *, low: int, high: int) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def faction_observation_key(faction: dict[str, Any], world: dict[str, Any]) -> str:
    sc = st.session_state.get("sector_control", {})
    fname = str(faction.get("name", ""))
    controls = sum(1 for _, info in sc.items() if str(info.get("controller", "")) == fname)
    ctrl_b = "low" if controls <= 2 else "high" if controls >= 5 else "mid"
    return (
        f"thr={_bucket(int(world.get('threat_clock', 0)), low=25, high=70)}"
        f"|sup={_bucket(int(world.get('supplies', 0)), low=2, high=6)}"
        f"|stab={_bucket(int(world.get('sector_stability', 0)), low=25, high=70)}"
        f"|p={_bucket(int(faction.get('pressure', 0)), low=30, high=65)}"
        f"|l={_bucket(int(faction.get('leverage', 0)), low=20, high=55)}"
        f"|c={ctrl_b}"
    )


def _faction_policy(name: str) -> str:
    return str(mas_state()["agents"].get(name, {}).get("policy", MAS_DEFAULT_POLICY))


def _bandit_pick_action_target(faction: dict[str, Any], obs_key: str) -> tuple[str, str]:
    from threadfall.game import _allowed_targets

    actions = ["extract", "research", "sabotage", "fortify", "resupply", "rewire"]
    fname = str(faction["name"])
    slot = mas_state()["agents"][fname]
    eps = float(slot.get("epsilon", 0.12))
    bandit = slot.setdefault("bandit", {"counts": {}, "values": {}})
    values = bandit.setdefault("values", {}).setdefault(obs_key, {})

    if random.random() < eps:
        shuffled = list(actions)
        random.shuffle(shuffled)
        for action in shuffled:
            opts = _allowed_targets(action, faction)
            if opts:
                return action, random.choice(opts)
        return "resupply", "Expedition Hub"

    best_score = None
    best_action = None
    for action in actions:
        score = float(values.get(action, 0.0))
        if best_score is None or score > best_score:
            best_score = score
            best_action = action
    assert best_action is not None
    opts = _allowed_targets(best_action, faction)
    if not opts:
        return best_action, random.choice(_allowed_targets("resupply", faction))
    return best_action, random.choice(opts)


def reward_from_effects(effects: dict[str, int], weights: dict[str, float]) -> float:
    mapping = {
        "intel": int(effects.get("intel_delta", 0) or 0),
        "leverage": int(effects.get("leverage_delta", 0) or 0),
        "threat": int(effects.get("threat_delta", 0) or 0),
        "stability": int(effects.get("stability_delta", 0) or 0),
        "supplies": int(effects.get("supplies_delta", 0) or 0),
        "pressure": int(effects.get("pressure_delta", 0) or 0),
    }
    r = 0.0
    for k, v in mapping.items():
        r += float(v) * float(weights.get(k, 0.0))
    return max(-18.0, min(18.0, r))


def update_bandit_after_action(*, faction_name: str, obs_key: str, action: str, effects: dict[str, int]) -> None:
    slot = mas_state()["agents"].get(faction_name)
    if not slot or str(slot.get("policy")) != "bandit":
        return
    weights = slot.get("factors") or default_factors_for_interest("")
    reward = reward_from_effects(effects, weights if isinstance(weights, dict) else default_factors_for_interest(""))
    bandit = slot.setdefault("bandit", {"counts": {}, "values": {}})
    counts = bandit.setdefault("counts", {}).setdefault(obs_key, {})
    values = bandit.setdefault("values", {}).setdefault(obs_key, {})
    counts[action] = int(counts.get(action, 0)) + 1
    n = int(counts[action])
    old = float(values.get(action, 0.0))
    values[action] = old + (reward - old) / float(max(1, n))


def decide_faction_turn(
    *,
    decision_engine: OllamaDecisionEngine,
    memory: Any,
    faction: dict[str, Any],
    world: dict[str, Any],
) -> tuple[DecisionProposal, str, list[Any], str, str]:
    """
    Returns: proposal, status, related_memories, obs_key, policy_used
    """
    from threadfall.game import _build_faction_memory_query, _choose_action, _validated_llm_decision

    obs_key = faction_observation_key(faction, world)
    fname = str(faction["name"])
    policy = _faction_policy(fname)
    query = _build_faction_memory_query(faction, world)
    related = memory.search(query, limit=5)

    if policy == "heuristic":
        a, t = _choose_action(faction, world)
        prop = DecisionProposal(
            action=a,
            target=t,
            reasoning=f"{fname} MAS heuristic policy.",
            confidence=0.55,
            source="mas_heuristic",
        )
        return prop, "ok", related, obs_key, policy

    if policy == "bandit":
        a, t = _bandit_pick_action_target(faction, obs_key)
        prop = DecisionProposal(
            action=a,
            target=t,
            reasoning=f"{fname} MAS bandit policy (obs={obs_key}).",
            confidence=0.55,
            source="mas_bandit",
        )
        return prop, "ok", related, obs_key, policy

    prop, status = _validated_llm_decision(
        decision_engine=decision_engine,
        faction=faction,
        world=world,
        related_memories=related,
    )
    if prop.source == "heuristic":
        prop = DecisionProposal(
            action=prop.action,
            target=prop.target,
            reasoning=prop.reasoning,
            confidence=prop.confidence,
            source="mas_llm_fallback",
        )
    else:
        prop = DecisionProposal(
            action=prop.action,
            target=prop.target,
            reasoning=prop.reasoning,
            confidence=prop.confidence,
            source="mas_llm",
        )
    return prop, status, related, obs_key, policy
