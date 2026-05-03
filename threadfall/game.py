from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

import streamlit as st

from threadfall.config import SETTINGS
from threadfall.embedder import Embedder
from threadfall.llm import DecisionProposal, OllamaDecisionEngine, propose_expedition_commit_supplies
from threadfall.mas import (
    advance_turn_order,
    decide_faction_turn,
    init_mas,
    merge_mas_from_save,
    ordered_factions,
    update_bandit_after_action,
)
from threadfall.memory import create_memory_store


SECTORS = [
    "Archive Chamber Theta",
    "Inversion Causeway",
    "Flooded Transit Gallery",
    "Vault Spine 3",
    "Thermal Lens Cathedral",
    "Resonance Stairwell",
    "Signal Orchard",
    "Mirror Fault Atrium",
    "Tide-Locked Rotunda",
    "Obsidian Switchyard",
    "Mnemonic Scriptorium",
    "Null Pump Station",
    "Gravimetric Choir",
]

# Topology: sector adjacency for the interactive map.
# Keep this small and readable; it’s a gameplay scaffold for traversal and “front lines”.
# Agents may mutate `st.session_state["sector_graph"]` via the `rewire` action; this dict is the boot template only.
SECTOR_GRAPH_DEFAULT: dict[str, list[str]] = {
    "Archive Chamber Theta": ["Mnemonic Scriptorium", "Mirror Fault Atrium", "Signal Orchard"],
    "Mnemonic Scriptorium": ["Archive Chamber Theta", "Resonance Stairwell", "Vault Spine 3"],
    "Signal Orchard": ["Archive Chamber Theta", "Mirror Fault Atrium", "Null Pump Station"],
    "Mirror Fault Atrium": ["Archive Chamber Theta", "Signal Orchard", "Inversion Causeway", "Thermal Lens Cathedral"],
    "Inversion Causeway": ["Mirror Fault Atrium", "Vault Spine 3", "Gravimetric Choir"],
    "Vault Spine 3": ["Mnemonic Scriptorium", "Inversion Causeway", "Obsidian Switchyard"],
    "Thermal Lens Cathedral": ["Mirror Fault Atrium", "Obsidian Switchyard", "Gravimetric Choir"],
    "Obsidian Switchyard": ["Vault Spine 3", "Thermal Lens Cathedral", "Flooded Transit Gallery"],
    "Flooded Transit Gallery": ["Obsidian Switchyard", "Tide-Locked Rotunda", "Null Pump Station"],
    "Tide-Locked Rotunda": ["Flooded Transit Gallery", "Resonance Stairwell"],
    "Resonance Stairwell": ["Tide-Locked Rotunda", "Mnemonic Scriptorium", "Gravimetric Choir"],
    "Null Pump Station": ["Flooded Transit Gallery", "Signal Orchard"],
    "Gravimetric Choir": ["Inversion Causeway", "Thermal Lens Cathedral", "Resonance Stairwell"],
}
# Back-compat name: read-only template; live graph is `current_sector_graph()`.
SECTOR_GRAPH = SECTOR_GRAPH_DEFAULT


def _copy_sector_graph(src: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    base = src or SECTOR_GRAPH_DEFAULT
    return {s: list(base.get(s, [])) for s in SECTORS}


def _normalize_sector_graph(loaded: Any) -> dict[str, list[str]]:
    if not isinstance(loaded, dict) or not loaded:
        return _copy_sector_graph(SECTOR_GRAPH_DEFAULT)
    out: dict[str, list[str]] = {s: [] for s in SECTORS}
    for sector in SECTORS:
        nbrs = loaded.get(sector, SECTOR_GRAPH_DEFAULT.get(sector, []))
        if not isinstance(nbrs, list):
            nbrs = list(SECTOR_GRAPH_DEFAULT.get(sector, []))
        seen: set[str] = set()
        for b in nbrs:
            if b in SECTORS and b != sector and b not in seen:
                seen.add(b)
                out[sector].append(b)
    for a in SECTORS:
        for b in list(out[a]):
            if a not in out[b]:
                out[b].append(a)
    return out


def current_sector_graph() -> dict[str, list[str]]:
    """Adjacency used for spillover, sabotage routing, and the topology map."""
    g = st.session_state.get("sector_graph")
    if isinstance(g, dict) and all(s in g for s in SECTORS):
        return g
    return SECTOR_GRAPH_DEFAULT


def _neighbors(sector: str) -> list[str]:
    return list(current_sector_graph().get(sector, []))


def _graph_distance(a: str, b: str) -> int | None:
    if a == b:
        return 0
    g = current_sector_graph()
    if a not in g or b not in g:
        return None
    q: list[tuple[str, int]] = [(a, 0)]
    seen = {a}
    while q:
        node, dist = q.pop(0)
        for nxt in _neighbors(node):
            if nxt in seen:
                continue
            if nxt == b:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def _pick_nearest_sector(candidates: list[str], goal: str) -> str:
    best: tuple[int | None, str] | None = None
    for cand in candidates:
        d = _graph_distance(cand, goal)
        if d is None:
            continue
        if best is None or d < best[0]:
            best = (d, cand)
    if best:
        return best[1]
    return random.choice(candidates)


def _strongest_interest_sector(rival_name: str) -> str | None:
    sector_control = st.session_state.get("sector_control", {})
    best_sector = None
    best_score = -1
    for sec, info in sector_control.items():
        score = int(info.get("interest", {}).get(rival_name, 0))
        if score > best_score:
            best_score = score
            best_sector = sec
    return best_sector


def _mark_sector_activity(sector: str, kind: str, detail: str = "") -> None:
    world = st.session_state.get("world_state", {})
    st.session_state.setdefault("sector_activity", {})
    st.session_state["sector_activity"][sector] = {
        "cycle": int(world.get("cycle", 0)),
        "kind": kind,
        "detail": detail,
    }


SECTOR_TAGS: dict[str, list[str]] = {
    "Archive Chamber Theta": ["archival", "mirrors", "acoustic"],
    "Inversion Causeway": ["gravity", "mechanical", "mirrors"],
    "Flooded Transit Gallery": ["flooded", "thermal", "acoustic"],
    "Vault Spine 3": ["mechanical", "archival", "thermal"],
    "Thermal Lens Cathedral": ["thermal", "mirrors", "gravity"],
    "Resonance Stairwell": ["acoustic", "gravity", "liminal"],
    "Signal Orchard": ["archival", "shadow", "liminal"],
    "Mirror Fault Atrium": ["mirrors", "material", "gravity"],
    "Tide-Locked Rotunda": ["flooded", "liminal", "acoustic"],
    "Obsidian Switchyard": ["mechanical", "shadow", "thermal"],
    "Mnemonic Scriptorium": ["archival", "material", "acoustic"],
    "Null Pump Station": ["flooded", "mechanical", "material"],
    "Gravimetric Choir": ["gravity", "acoustic", "thermal"],
}

THREATS = [
    "Shard swarm",
    "Witness echo",
    "Guardian engine",
    "Silent stalker",
]

OBJECTIVES = [
    "Recover unstable artifact",
    "Map anomaly corridor",
    "Rescue missing survey team",
    "Restore extraction relay",
]

ANOMALIES = [
    "Gravity pulls sideways toward illuminated surfaces.",
    "Sound briefly manifests as visible light trails.",
    "Unobserved doors change destination after ten seconds.",
    "Metal objects duplicate after thermal shock.",
    "Shadows persist even after their source moves away.",
]

MISSION_TYPES: dict[str, dict[str, Any]] = {
    "standard": {
        "label": "Standard Expedition",
        "objectives": OBJECTIVES,
        "intel_bonus": 0,
        "artifact_chance": 0.55,
        "stability_cost_range": (3, 8),
        "reward_intel_range": (4, 9),
        "heat_delta": 8,
        "threat_delta_success": (-2, -1),
        "threat_delta_fail": (3, 7),
        "stability_bonus_success": 0,
        "stage_mods": {"Approach": 0.0, "Contact": -0.02, "Objective": -0.05, "Extraction": -0.02},
    },
    "survey": {
        "label": "Survey Run",
        "objectives": ["Map anomaly corridor", "Document anomaly rules", "Calibrate observation relay"],
        "intel_bonus": 3,
        "artifact_chance": 0.18,
        "stability_cost_range": (2, 6),
        "reward_intel_range": (7, 12),
        "heat_delta": 5,
        "threat_delta_success": (-3, -1),
        "threat_delta_fail": (2, 5),
        "stability_bonus_success": 2,
        "stage_mods": {"Approach": 0.02, "Contact": 0.0, "Objective": -0.03, "Extraction": 0.01},
    },
    "extraction": {
        "label": "Extraction Push",
        "objectives": ["Recover unstable artifact", "Secure relic cache", "Strip salvage from a guardian husk"],
        "intel_bonus": 0,
        "artifact_chance": 0.8,
        "stability_cost_range": (4, 10),
        "reward_intel_range": (3, 8),
        "heat_delta": 12,
        "threat_delta_success": (-2, 0),
        "threat_delta_fail": (4, 9),
        "stability_bonus_success": 0,
        "stage_mods": {"Approach": -0.01, "Contact": -0.03, "Objective": -0.07, "Extraction": -0.05},
    },
    "containment": {
        "label": "Containment Operation",
        "objectives": ["Stabilize a fracture zone", "Reinforce extraction relay", "Purge a hostile echo imprint"],
        "intel_bonus": 1,
        "artifact_chance": 0.12,
        "stability_cost_range": (2, 7),
        "reward_intel_range": (4, 9),
        "heat_delta": 3,
        "threat_delta_success": (-6, -3),
        "threat_delta_fail": (2, 6),
        "stability_bonus_success": 6,
        "stage_mods": {"Approach": 0.01, "Contact": -0.01, "Objective": -0.04, "Extraction": 0.02},
    },
    "parley": {
        "label": "Diplomatic Parley",
        "objectives": ["Broker a corridor truce", "Trade extraction priorities", "Secure a temporary alignment"],
        "intel_bonus": 2,
        "artifact_chance": 0.05,
        "stability_cost_range": (1, 4),
        "reward_intel_range": (6, 11),
        "heat_delta": 2,
        "threat_delta_success": (-2, -1),
        "threat_delta_fail": (1, 4),
        "stability_bonus_success": 1,
        "stage_mods": {"Approach": 0.03, "Contact": 0.02, "Objective": -0.01, "Extraction": 0.02},
    },
    "deep_dive": {
        "label": "Deep Dive",
        "objectives": ["Reach the inner vault boundary", "Extract a core anomaly rule", "Map the forbidden loop"],
        "intel_bonus": 4,
        "artifact_chance": 0.35,
        "stability_cost_range": (6, 12),
        "reward_intel_range": (10, 18),
        "heat_delta": 14,
        "threat_delta_success": (-1, 1),
        "threat_delta_fail": (6, 12),
        "stability_bonus_success": 0,
        "stage_mods": {"Approach": -0.02, "Contact": -0.04, "Objective": -0.09, "Extraction": -0.05},
    },
    "counter": {
        "label": "Counter-Mission",
        "objectives": OBJECTIVES,
        "intel_bonus": 0,
        "artifact_chance": 0.45,
        "stability_cost_range": (3, 9),
        "reward_intel_range": (4, 9),
        "heat_delta": 10,
        "threat_delta_success": (-2, 0),
        "threat_delta_fail": (3, 7),
        "stability_bonus_success": 0,
        "stage_mods": {"Approach": 0.0, "Contact": -0.03, "Objective": -0.06, "Extraction": -0.03},
    },
}

RESPONSES = [
    "Deploy tether anchors before crossing unstable slabs.",
    "Use scanner pings sparingly to avoid strengthening the anomaly.",
    "Split roles into navigator, observer, and retrieval specialist.",
    "Extract early if two anomaly rules begin to overlap.",
    "Tag every geometry shift to improve future mission retrieval.",
]

FACTIONS = [
    "Helios Prospectors",
    "Glass Archive",
    "Morrow Division",
]

ARTIFACTS = [
    {
        "name": "Lattice Prism",
        "effect": "Improves anomaly scan clarity during archive missions.",
        "rarity": "Rare",
    },
    {
        "name": "Echo Spindle",
        "effect": "Lets the squad recover one extra field memory after a success.",
        "rarity": "Uncommon",
    },
    {
        "name": "Null Coil",
        "effect": "Reduces faction pressure created by unstable recoveries.",
        "rarity": "Rare",
    },
    {
        "name": "Witness Lens",
        "effect": "Boosts insight gained from memory-heavy sectors.",
        "rarity": "Legendary",
    },
]

FACTION_TEMPLATES = [
    {
        "name": "Helios Prospectors",
        "interest": "Artifact extraction",
        "goal": "Secure high-yield relics before the ruin fully destabilizes.",
        "preferred_objectives": ["Recover unstable artifact", "Restore extraction relay"],
        "preferred_sectors": ["Vault Spine 3", "Inversion Causeway"],
        "rival": "Glass Archive",
    },
    {
        "name": "Glass Archive",
        "interest": "Knowledge capture",
        "goal": "Catalog anomaly behavior and preserve non-repeatable discoveries.",
        "preferred_objectives": ["Map anomaly corridor", "Rescue missing survey team"],
        "preferred_sectors": ["Archive Chamber Theta", "Thermal Lens Cathedral"],
        "rival": "Morrow Division",
    },
    {
        "name": "Morrow Division",
        "interest": "Territorial denial",
        "goal": "Deny access to unstable sectors and force all rivals into retreat.",
        "preferred_objectives": ["Restore extraction relay", "Recover unstable artifact"],
        "preferred_sectors": ["Flooded Transit Gallery", "Vault Spine 3"],
        "rival": "Helios Prospectors",
    },
]


@dataclass
class Services:
    embedder: Embedder
    memory: object
    decision_engine: OllamaDecisionEngine


def _infer_anomaly_tags(anomaly: str) -> list[str]:
    text = anomaly.lower()
    tags: list[str] = []
    if "gravity" in text:
        tags.append("gravity")
    if "sound" in text or "echo" in text:
        tags.append("acoustic")
    if "door" in text:
        tags.append("liminal")
    if "metal" in text:
        tags.append("material")
    if "thermal" in text or "heat" in text:
        tags.append("thermal")
    if "shadow" in text:
        tags.append("shadow")
    return tags or ["unknown"]


def _default_stages() -> list[dict[str, Any]]:
    return [
        {"name": "Approach", "focus": "Reach the entry geometry without triggering escalation."},
        {"name": "Contact", "focus": "Observe the anomaly rules and survive first contact."},
        {"name": "Objective", "focus": "Complete the main task while pressure spikes."},
        {"name": "Extraction", "focus": "Exit with logs and salvage before the sector shifts."},
    ]


RIVAL_ACTIONS = ["standard", "survey", "extraction", "containment"]


def _bucket(value: int, *, low: int, high: int) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def rival_observation() -> dict[str, str]:
    world = st.session_state["world_state"]
    return {
        "threat": _bucket(int(world.get("threat_clock", 0)), low=25, high=70),
        "supplies": _bucket(int(world.get("supplies", 0)), low=2, high=6),
        "stability": _bucket(int(world.get("sector_stability", 0)), low=25, high=70),
    }


def _obs_key(obs: dict[str, str]) -> str:
    return f"threat={obs['threat']}|supplies={obs['supplies']}|stability={obs['stability']}"


def init_rival_learning() -> None:
    if "rival_policy" not in st.session_state:
        st.session_state["rival_policy"] = {
            "mode": "bandit",  # heuristic | bandit | qlearn — default bandit for autonomous ML rivals
            "epsilon": 0.09,
            "alpha": 0.07,
            "gamma": 0.86,
            # Caps single-tick learning signal for AI competitors (slows snowball).
            "competitor_reward_cap": 22.0,
            "q_value_clip": 35.0,
        }
    if "rival_bandit" not in st.session_state:
        st.session_state["rival_bandit"] = {"counts": {}, "values": {}}
    if "rival_q" not in st.session_state:
        st.session_state["rival_q"] = {}
    rp = st.session_state["rival_policy"]
    rp.setdefault("mode", "bandit")
    rp.setdefault("competitor_reward_cap", 22.0)
    rp.setdefault("q_value_clip", 35.0)


def choose_rival_action(obs: dict[str, str]) -> str:
    init_rival_learning()
    mode = st.session_state["rival_policy"]["mode"]
    eps = float(st.session_state["rival_policy"].get("epsilon", 0.09))
    key = _obs_key(obs)

    if random.random() < eps:
        return random.choice(RIVAL_ACTIONS)

    if mode == "bandit":
        values = st.session_state["rival_bandit"]["values"].get(key, {})
        if not values:
            return random.choice(RIVAL_ACTIONS)
        return max(RIVAL_ACTIONS, key=lambda a: float(values.get(a, 0.0)))

    if mode == "qlearn":
        qrow = st.session_state["rival_q"].get(key, {})
        if not qrow:
            return random.choice(RIVAL_ACTIONS)
        return max(RIVAL_ACTIONS, key=lambda a: float(qrow.get(a, 0.0)))

    # heuristic
    if obs["threat"] == "high":
        return "containment"
    if obs["supplies"] == "low":
        return "survey"
    if obs["stability"] == "low":
        return "containment"
    return random.choice(["survey", "extraction", "standard"])


def _clip_rival_value(v: float, *, clip: float) -> float:
    c = max(1.0, float(clip))
    return max(-c, min(c, v))


def update_rival_learning(obs: dict[str, str], action: str, reward: float, next_obs: dict[str, str]) -> None:
    init_rival_learning()
    mode = st.session_state["rival_policy"]["mode"]
    key = _obs_key(obs)
    clip = float(st.session_state["rival_policy"].get("q_value_clip", 35.0))

    if mode == "bandit":
        bandit = st.session_state["rival_bandit"]
        counts = bandit["counts"].setdefault(key, {})
        values = bandit["values"].setdefault(key, {})
        counts[action] = int(counts.get(action, 0)) + 1
        n = counts[action]
        old = float(values.get(action, 0.0))
        new_v = old + (reward - old) / float(max(1, n))
        values[action] = _clip_rival_value(new_v, clip=clip)
        return

    if mode == "qlearn":
        alpha = float(st.session_state["rival_policy"].get("alpha", 0.07))
        gamma = float(st.session_state["rival_policy"].get("gamma", 0.86))
        q = st.session_state["rival_q"]
        row = q.setdefault(key, {})
        old = float(row.get(action, 0.0))
        next_key = _obs_key(next_obs)
        next_row = q.get(next_key, {})
        best_next = max([float(next_row.get(a, 0.0)) for a in RIVAL_ACTIONS], default=0.0)
        new_q = old + alpha * (reward + gamma * best_next - old)
        row[action] = _clip_rival_value(new_q, clip=clip)
        for a in RIVAL_ACTIONS:
            if a in row:
                row[a] = _clip_rival_value(float(row[a]), clip=clip)
        return


def configure_ai_agents(num_agents: int) -> None:
    num_agents = max(0, min(6, int(num_agents)))
    names = [
        "Aegis Unit",
        "Vanta Cell",
        "Orchid Team",
        "Cinder Crew",
        "Kestrel Group",
        "Pale Lanterns",
    ]
    agents: list[dict[str, Any]] = []
    for idx in range(num_agents):
        name = names[idx] if idx < len(names) else f"Rival Unit {idx + 1}"
        agents.append(
            {
                "name": name,
                "intel": 0,
                "artifacts": 0,
                "last_outcome": "None",
                "last_mission_type": "standard",
                "last_sector": None,
                "last_cycle": None,
            }
        )

    st.session_state["ai_config"] = {"num_agents": num_agents}
    st.session_state["ai_agents"] = agents


def _competitor_generate_mission(agent_name: str) -> dict[str, object]:
    # Competitors ignore queued counter-missions; those are player-facing crises.
    saved = list(st.session_state.get("counter_missions", []))
    st.session_state["counter_missions"] = []
    try:
        return build_mission(agent_name)
    finally:
        st.session_state["counter_missions"] = saved


def build_mission_custom(
    squad_name: str,
    *,
    forced_mission_type: str | None = None,
    forced_sector: str | None = None,
) -> dict[str, object]:
    mission = build_mission(squad_name)
    if forced_sector:
        mission["sector"] = forced_sector
        mission["sector_tags"] = SECTOR_TAGS.get(forced_sector, [])

    if forced_mission_type and forced_mission_type in MISSION_TYPES:
        config = MISSION_TYPES[forced_mission_type]
        mission["mission_type"] = forced_mission_type
        mission["mission_label"] = config.get("label", forced_mission_type)
        objective_pool = list(config.get("objectives", OBJECTIVES))
        mission["objective"] = random.choice(objective_pool)
        mission["title"] = f"{mission['sector']} // {mission['objective']}"
        stability_low, stability_high = config["stability_cost_range"]
        intel_low, intel_high = config["reward_intel_range"]
        world = st.session_state.get("world_state", _default_world_state())
        mission["stability_cost"] = random.randint(int(stability_low), int(stability_high))
        mission["reward_intel"] = (
            random.randint(int(intel_low), int(intel_high))
            + int(world.get("cycle", 1))
            + int(config.get("intel_bonus", 0))
        )
        mission["stages"] = _default_stages()

    return mission


def _record_competitor_result(agent: dict[str, Any], history_item: dict[str, Any]) -> None:
    if history_item.get("outcome") == "Success":
        agent["intel"] = int(agent.get("intel", 0)) + int(history_item.get("reward_intel", 0))
        agent["artifacts"] = int(agent.get("artifacts", 0)) + int(history_item.get("artifact_found", 0))
    agent["last_outcome"] = history_item.get("outcome", "None")
    agent["last_mission_type"] = history_item.get("mission_type", "standard")
    agent["last_sector"] = history_item.get("sector")
    agent["last_cycle"] = history_item.get("cycle")


def simulate_ai_competitors(memory, *, per_tick: int = 1) -> list[dict[str, Any]]:
    if not st.session_state.get("ai_agents"):
        return []
    if st.session_state["endgame"]["ended"]:
        return []

    results: list[dict[str, Any]] = []
    for _ in range(max(1, int(per_tick))):
        for agent in st.session_state.get("ai_agents", []):
            obs = rival_observation()
            action = choose_rival_action(obs)
            mission = build_mission_custom(agent["name"], forced_mission_type=action)
            history_item = _resolve_mission_for_actor(
                memory=memory,
                mission=mission,
                actor_name=agent["name"],
                committed_supplies=0,
                history_type="ai_expedition",
                advance_cycle=False,
                forced_outcome=None,
                outcome_bonus_intel=0,
                artifact_bonus=0,
            )
            # Reward: bias toward intel + artifacts, but penalize escalating threat/heat.
            reward = float(history_item.get("reward_intel", 0))
            reward += 20.0 * float(history_item.get("artifact_found", 0))
            if history_item.get("outcome") != "Success":
                reward -= 4.0
            # Sub-goals: nudge long-arc exploration without hard-coding endgame.
            world = st.session_state.get("world_state", _default_world_state())
            intel = int(agent.get("intel", 0))
            arts = int(agent.get("artifacts", 0))
            if action == "survey" and intel < 38 and history_item.get("outcome") == "Success":
                reward += 2.8
            if action == "extraction" and arts < 2 and history_item.get("outcome") == "Success":
                reward += 2.2
            if action == "containment" and int(world.get("threat_clock", 0)) >= 62:
                reward += 1.4 if history_item.get("outcome") == "Success" else -0.8
            init_rival_learning()
            cap = float(st.session_state["rival_policy"].get("competitor_reward_cap", 22.0))
            reward = max(-cap, min(cap, reward))
            next_obs = rival_observation()
            update_rival_learning(obs, action, reward, next_obs)
            _record_competitor_result(agent, history_item)
            results.append(history_item)
        persist_state()
    return results


def _default_world_state() -> dict[str, Any]:
    return {
        "cycle": 1,
        "intel": 0,
        "supplies": 6,
        "threat_clock": 18,
        "sector_stability": 62,
    }


def _default_sector_control() -> dict[str, Any]:
    return {
        sector: {
            "controller": "Neutral",
            "stability": random.randint(45, 70),
            "heat": random.randint(10, 30),
            "interest": {name: 0 for name in FACTIONS},
        }
        for sector in SECTORS
    }


def _default_factions() -> list[dict[str, Any]]:
    defaults = []
    for template, pressure, stance, leverage in [
        (FACTION_TEMPLATES[0], 28, "Transactional", 34),
        (FACTION_TEMPLATES[1], 21, "Curious", 29),
        (FACTION_TEMPLATES[2], 39, "Hostile", 42),
    ]:
        defaults.append(
            {
                "name": template["name"],
                "stance": stance,
                "pressure": pressure,
                "interest": template["interest"],
                "goal": template["goal"],
                "preferred_objectives": template["preferred_objectives"],
                "preferred_sectors": template["preferred_sectors"],
                "rival": template["rival"],
                "leverage": leverage,
                "recent_memory": [],
                "last_outcome": "None",
                "campaign_arc": "",
            }
        )
    return defaults


def _empty_persistent_state() -> dict[str, Any]:
    factions = _default_factions()
    return {
        "world_state": _default_world_state(),
        "sector_control": _default_sector_control(),
        "sector_graph": _normalize_sector_graph(None),
        "factions": factions,
        "mas": merge_mas_from_save(None, factions),
        "inventory": [],
        "history": [],
        "activities": [],
        "events": [],
        "counter_missions": [],
        "negotiations": [],
        "snapshots": [],
        "ai_config": {"num_agents": 3},
        "ai_agents": [],
        "autonomous_stack": {
            "enabled": True,
            "player_expeditions_per_tick": 1,
            "faction_rounds_per_tick": 1,
            "use_ollama_player_commit": True,
        },
        "endgame": {
            "ended": False,
            "result": "Active",
            "winner": "None",
            "trigger_cycle": 0,
            "summary": "",
            "details": [],
        },
    }


def _load_persistent_state() -> dict[str, Any]:
    if SETTINGS.save_path.exists():
        try:
            loaded = json.loads(SETTINGS.save_path.read_text())
            return _normalize_persistent_state(loaded)
        except json.JSONDecodeError:
            return _empty_persistent_state()
    return _empty_persistent_state()


def _normalize_persistent_state(loaded: dict[str, Any]) -> dict[str, Any]:
    fresh = _empty_persistent_state()
    world_state = {**fresh["world_state"], **loaded.get("world_state", {})}

    saved_factions = {item.get("name"): item for item in loaded.get("factions", [])}
    factions = []
    for default in fresh["factions"]:
        merged = {**default, **saved_factions.get(default["name"], {})}
        merged["recent_memory"] = list(merged.get("recent_memory", []))[-4:]
        factions.append(merged)

    return {
        "world_state": world_state,
        "sector_control": _normalize_sector_control(loaded.get("sector_control", {})),
        "sector_graph": _normalize_sector_graph(loaded.get("sector_graph")),
        "factions": factions,
        "inventory": list(loaded.get("inventory", [])),
        "history": list(loaded.get("history", [])),
        "activities": list(loaded.get("activities", [])),
        "events": list(loaded.get("events", []))[-12:],
        "counter_missions": list(loaded.get("counter_missions", []))[-8:],
        "negotiations": list(loaded.get("negotiations", []))[-40:],
        "snapshots": list(loaded.get("snapshots", []))[-120:],
        "ai_config": {**fresh["ai_config"], **loaded.get("ai_config", {})},
        "ai_agents": list(loaded.get("ai_agents", [])),
        "rival_policy": dict(loaded.get("rival_policy", {})),
        "rival_bandit": dict(loaded.get("rival_bandit", {})),
        "rival_q": dict(loaded.get("rival_q", {})),
        "endgame": {**fresh["endgame"], **loaded.get("endgame", {})},
        "mas": merge_mas_from_save(
            loaded.get("mas") if isinstance(loaded.get("mas"), dict) else None,
            factions,
        ),
        "autonomous_stack": {
            **fresh["autonomous_stack"],
            **(
                dict(loaded["autonomous_stack"])
                if isinstance(loaded.get("autonomous_stack"), dict)
                else {}
            ),
        },
    }


def _normalize_sector_control(loaded: dict[str, Any]) -> dict[str, Any]:
    fresh = _default_sector_control()
    normalized: dict[str, Any] = {}
    for sector, default in fresh.items():
        merged = {**default, **loaded.get(sector, {})}
        merged_interest = {name: 0 for name in FACTIONS}
        merged_interest.update(merged.get("interest", {}))
        merged["interest"] = merged_interest
        normalized[sector] = merged
    return normalized


def persist_state() -> None:
    snapshot = {
        "world_state": st.session_state["world_state"],
        "sector_control": st.session_state["sector_control"],
        "sector_graph": st.session_state.get("sector_graph") or _normalize_sector_graph(None),
        "factions": st.session_state["factions"],
        "inventory": st.session_state["inventory"],
        "history": st.session_state["history"],
        "activities": st.session_state["activities"],
        "events": st.session_state["events"],
        "counter_missions": st.session_state["counter_missions"],
        "negotiations": st.session_state["negotiations"],
        "snapshots": st.session_state["snapshots"],
        "ai_config": st.session_state.get("ai_config", {"num_agents": 0}),
        "ai_agents": st.session_state.get("ai_agents", []),
        "rival_policy": st.session_state.get("rival_policy", {}),
        "rival_bandit": st.session_state.get("rival_bandit", {}),
        "rival_q": st.session_state.get("rival_q", {}),
        "endgame": st.session_state["endgame"],
        "mas": st.session_state.get("mas", merge_mas_from_save(None, st.session_state.get("factions", []))),
        "autonomous_stack": st.session_state.get(
            "autonomous_stack",
            {
                "enabled": True,
                "player_expeditions_per_tick": 1,
                "faction_rounds_per_tick": 1,
                "use_ollama_player_commit": True,
            },
        ),
    }
    SETTINGS.save_path.write_text(json.dumps(snapshot, indent=2))


def snapshot_for_undo(reason: str) -> None:
    keys = [
        "world_state",
        "sector_control",
        "sector_graph",
        "factions",
        "inventory",
        "history",
        "activities",
        "events",
        "counter_missions",
        "negotiations",
        "snapshots",
        "mas",
        "endgame",
        "current_mission",
        "log",
        "autoplay_enabled",
        "autoplay_remaining_ticks",
        "autoplay_rounds_per_tick",
        "autoplay_delay_seconds",
        "autonomous_stack",
    ]
    payload: dict[str, Any] = {}
    for key in keys:
        if key in st.session_state:
            payload[key] = st.session_state[key]

    stack = st.session_state.get("_undo_stack", [])
    stack.append(
        {
            "reason": reason,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cycle": st.session_state.get("world_state", {}).get("cycle", 0),
            "state": json.loads(json.dumps(payload)),
        }
    )
    st.session_state["_undo_stack"] = stack[-12:]


def undo_last_tick() -> tuple[bool, str]:
    stack = st.session_state.get("_undo_stack", [])
    if not stack:
        return False, "No undo snapshot available yet."

    last = stack.pop()
    st.session_state["_undo_stack"] = stack
    state = last.get("state", {})
    for key, value in state.items():
        st.session_state[key] = value

    persist_state()
    return True, f"Restored snapshot ({last.get('reason', 'undo')})."


def bootstrap_state() -> None:
    if "_persistent_loaded" not in st.session_state:
        saved = _load_persistent_state()
        st.session_state["world_state"] = saved["world_state"]
        st.session_state["sector_control"] = saved["sector_control"]
        st.session_state["sector_graph"] = saved["sector_graph"]
        st.session_state["factions"] = saved["factions"]
        st.session_state["inventory"] = saved["inventory"]
        st.session_state["history"] = saved["history"]
        st.session_state["activities"] = saved["activities"]
        st.session_state["events"] = saved["events"]
        st.session_state["counter_missions"] = saved["counter_missions"]
        st.session_state["negotiations"] = saved["negotiations"]
        st.session_state["snapshots"] = saved["snapshots"]
        st.session_state["ai_config"] = saved.get("ai_config", {"num_agents": 0})
        st.session_state["ai_agents"] = saved.get("ai_agents", [])
        st.session_state["rival_policy"] = saved.get("rival_policy", {})
        st.session_state["rival_bandit"] = saved.get("rival_bandit", {})
        st.session_state["rival_q"] = saved.get("rival_q", {})
        st.session_state["endgame"] = saved["endgame"]
        st.session_state["mas"] = saved.get("mas", merge_mas_from_save(None, saved["factions"]))
        st.session_state["autonomous_stack"] = saved.get(
            "autonomous_stack",
            {
                "enabled": True,
                "player_expeditions_per_tick": 1,
                "faction_rounds_per_tick": 1,
                "use_ollama_player_commit": True,
            },
        )
        st.session_state["_persistent_loaded"] = True
    if "sector_graph" not in st.session_state or not isinstance(st.session_state.get("sector_graph"), dict):
        st.session_state["sector_graph"] = _normalize_sector_graph(st.session_state.get("sector_graph"))
    if "squad_name" not in st.session_state:
        st.session_state["squad_name"] = "Orpheus Unit"
    if "operator_name" not in st.session_state:
        st.session_state["operator_name"] = "Lead Researcher"
    if "log" not in st.session_state:
        st.session_state["log"] = ["Threadfall systems booted."]
    if "_undo_stack" not in st.session_state:
        st.session_state["_undo_stack"] = []
    if "current_mission" not in st.session_state:
        st.session_state["current_mission"] = build_mission(st.session_state["squad_name"])
    if "memory_results" not in st.session_state:
        st.session_state["memory_results"] = []
    if "last_resolution" not in st.session_state:
        st.session_state["last_resolution"] = None
    if "mission_commit_supplies" not in st.session_state:
        st.session_state["mission_commit_supplies"] = 0
    if "pending_encounter" not in st.session_state:
        st.session_state["pending_encounter"] = None
    if "sector_activity" not in st.session_state:
        st.session_state["sector_activity"] = {}
    if "ai_config" not in st.session_state:
        st.session_state["ai_config"] = {"num_agents": 0}
    if "ai_agents" not in st.session_state:
        st.session_state["ai_agents"] = []
    init_rival_learning()
    init_mas()
    ensure_autonomous_stack_defaults()
    n_agents_cfg = int(st.session_state.get("ai_config", {}).get("num_agents", 0))
    if n_agents_cfg > 0 and len(st.session_state.get("ai_agents", [])) != n_agents_cfg:
        configure_ai_agents(n_agents_cfg)
    if "autoplay_remaining_ticks" not in st.session_state:
        st.session_state["autoplay_enabled"] = True
        st.session_state["autoplay_remaining_ticks"] = 500
        st.session_state["autoplay_rounds_per_tick"] = 1
        st.session_state["autoplay_delay_seconds"] = 0.65
    if "activities" not in st.session_state:
        st.session_state["activities"] = []
    if "events" not in st.session_state:
        st.session_state["events"] = []
    if "counter_missions" not in st.session_state:
        st.session_state["counter_missions"] = []
    if "negotiations" not in st.session_state:
        st.session_state["negotiations"] = []
    if "snapshots" not in st.session_state:
        st.session_state["snapshots"] = []
    if "endgame" not in st.session_state:
        st.session_state["endgame"] = _empty_persistent_state()["endgame"]
    if not st.session_state["snapshots"]:
        record_snapshot("campaign_boot")


@st.cache_resource(show_spinner=False)
def init_services() -> Services:
    embedder = Embedder.create(
        dim=SETTINGS.embed_dim,
        requested_device=SETTINGS.requested_device,
        ollama_enabled=SETTINGS.ollama_enabled,
        ollama_url=SETTINGS.ollama_url,
        ollama_model=SETTINGS.ollama_embed_model,
        ollama_timeout_seconds=SETTINGS.ollama_timeout_seconds,
    )
    memory = create_memory_store(embedder)
    decision_engine = OllamaDecisionEngine(
        base_url=SETTINGS.ollama_url,
        model=SETTINGS.ollama_model,
        timeout_seconds=SETTINGS.ollama_timeout_seconds,
        enabled=SETTINGS.ollama_enabled,
    )
    return Services(embedder=embedder, memory=memory, decision_engine=decision_engine)


def current_device_label(embedder: Embedder) -> str:
    return embedder.device.upper()


def _find_faction(name: str) -> dict[str, Any]:
    for faction in st.session_state.get("factions", []):
        if faction["name"] == name:
            return faction
    return _default_factions()[0]


def _choose_sponsor() -> dict[str, Any]:
    factions = st.session_state.get("factions", _default_factions())
    weights = [max(1, faction["pressure"] + faction["leverage"] // 2) for faction in factions]
    return random.choices(factions, weights=weights, k=1)[0]


def _objective_for_faction(faction: dict[str, Any]) -> str:
    pool = list(OBJECTIVES)
    weighted = pool + faction["preferred_objectives"] + faction["preferred_objectives"]
    return random.choice(weighted)


def _sector_for_faction(faction: dict[str, Any]) -> str:
    sector_state = st.session_state.get("sector_control", _default_sector_control())
    weighted: list[str] = []
    for sector in SECTORS:
        info = sector_state.get(sector, {})
        interest = info.get("interest", {}).get(faction["name"], 0)
        weight = 1 + interest // 8
        if sector in faction["preferred_sectors"]:
            weight += 3
        if info.get("controller") == faction["rival"]:
            weight += 2
        if info.get("heat", 0) >= 45:
            weight += 1
        weighted.extend([sector] * max(1, weight))
    return random.choice(weighted)


def _threat_for_faction(faction: dict[str, Any]) -> str:
    if faction["name"] == "Morrow Division":
        return random.choice(["Guardian engine", "Silent stalker", "Witness echo"])
    if faction["name"] == "Glass Archive":
        return random.choice(["Witness echo", "Shard swarm", "Guardian engine"])
    return random.choice(["Shard swarm", "Silent stalker", "Guardian engine"])


def _mission_pressure_tags(sponsor: dict[str, Any]) -> list[str]:
    tags = []
    if sponsor["pressure"] >= 60:
        tags.append("High rival interference expected.")
    if sponsor["leverage"] >= 50:
        tags.append("Sponsor has enough leverage to alter extraction priorities.")
    if sponsor["last_outcome"] == "Compromised":
        tags.append("Sponsor is forcing a redemption cycle after the last failed push.")
    elif sponsor["last_outcome"] == "Success":
        tags.append("Sponsor is pressing the advantage after the last successful incursion.")
    return tags or ["No major faction escalations detected."]


def _active_counter_mission() -> dict[str, Any] | None:
    counter_missions = st.session_state.get("counter_missions", [])
    if not counter_missions:
        return None
    return counter_missions[0]


def _build_counter_mission(squad_name: str, counter: dict[str, Any]) -> dict[str, object]:
    sponsor = _find_faction(counter["sponsor"])
    rival = _find_faction(counter["rival"])
    anomaly_tags = _infer_anomaly_tags(counter["anomaly"])
    sector_tags = SECTOR_TAGS.get(counter["sector"], [])
    return {
        "mission_id": str(uuid.uuid4()),
        "title": f"{counter['sector']} // {counter['objective']}",
        "sector": counter["sector"],
        "threat": counter["threat"],
        "objective": counter["objective"],
        "difficulty": counter["difficulty"],
        "anomaly": counter["anomaly"],
        "brief": (
            f"{squad_name} has been diverted into an urgent counter-operation at {counter['sector']}. "
            f"{counter['summary']}"
        ),
        "response": random.sample(RESPONSES, 3),
        "sponsor": sponsor["name"],
        "rival": rival["name"],
        "directive": counter["directive"],
        "pressure_tags": [f"Counter-mission triggered by event: {counter['event_name']}"],
        "artifact": random.choice(ARTIFACTS),
        "stability_cost": counter["stability_cost"],
        "reward_intel": counter["reward_intel"],
        "mission_type": "counter",
        "mission_label": MISSION_TYPES["counter"]["label"],
        "stages": _default_stages(),
        "sector_tags": sector_tags,
        "anomaly_tags": anomaly_tags,
        "event_name": counter["event_name"],
    }


def build_mission(squad_name: str) -> dict[str, object]:
    counter = _active_counter_mission()
    if counter:
        return _build_counter_mission(squad_name, counter)

    world = st.session_state.get("world_state", _default_world_state())
    sponsor = _choose_sponsor()
    rival = _find_faction(sponsor["rival"])
    sector = _sector_for_faction(sponsor)
    control = st.session_state.get("sector_control", {}).get(sector, {})
    threat = _threat_for_faction(sponsor)
    anomaly = random.choice(ANOMALIES)
    anomaly_tags = _infer_anomaly_tags(anomaly)
    sector_tags = SECTOR_TAGS.get(sector, [])
    response = random.sample(RESPONSES, 3)
    artifact = random.choice(ARTIFACTS)
    pressure_bias = sponsor["pressure"] + rival["pressure"] // 2
    if pressure_bias >= 85:
        difficulty = random.choice(["Severe", "Terminal"])
    elif pressure_bias >= 55:
        difficulty = random.choice(["Elevated", "Severe"])
    else:
        difficulty = random.choice(["Low", "Elevated", "Severe"])

    weights: list[tuple[str, int]] = [("standard", 4)]
    if sponsor["interest"] == "Knowledge capture":
        weights.append(("survey", 6))
        weights.append(("containment", 2))
        weights.append(("parley", 2))
    elif sponsor["interest"] == "Artifact extraction":
        weights.append(("extraction", 7))
        weights.append(("survey", 1))
        weights.append(("parley", 1))
    else:
        weights.append(("containment", 6))
        weights.append(("extraction", 2))
        weights.append(("parley", 1))

    # Gate deep-dive missions to later campaign.
    if world.get("cycle", 1) >= 10 and world.get("intel", 0) >= 40:
        weights.append(("deep_dive", 2))

    mission_type_pool: list[str] = []
    for name, w in weights:
        mission_type_pool.extend([name] * max(1, w))
    mission_type = random.choice(mission_type_pool)
    config = MISSION_TYPES.get(mission_type, MISSION_TYPES["standard"])

    objective_pool = list(config["objectives"]) + sponsor["preferred_objectives"]
    objective = random.choice(objective_pool)

    title = f"{sector} // {objective}"
    brief = (
        f"{squad_name} is entering {sector} under unstable field conditions. "
        f"Primary threat is {threat.lower()}. Field reports indicate that {anomaly.lower()} "
        f"{sponsor['name']} is watching this sector for {sponsor['interest'].lower()}. "
        f"{rival['name']} is expected to contest the operation."
    )
    faction_directive = (
        f"{sponsor['name']} directive: {sponsor['goal']} Rival pressure from {rival['name']} "
        f"is currently rated at {rival['pressure']}."
    )
    tags = _mission_pressure_tags(sponsor)
    if control:
        tags.append(f"Sector control: {control['controller']} · Heat {control['heat']} · Stability {control['stability']}")

    stability_low, stability_high = config["stability_cost_range"]
    intel_low, intel_high = config["reward_intel_range"]

    return {
        "mission_id": str(uuid.uuid4()),
        "title": title,
        "sector": sector,
        "threat": threat,
        "objective": objective,
        "difficulty": difficulty,
        "anomaly": anomaly,
        "brief": brief,
        "response": response,
        "sponsor": sponsor["name"],
        "rival": rival["name"],
        "directive": faction_directive,
        "pressure_tags": tags,
        "artifact": artifact,
        "stability_cost": random.randint(int(stability_low), int(stability_high)),
        "reward_intel": random.randint(int(intel_low), int(intel_high)) + world["cycle"] + int(config.get("intel_bonus", 0)),
        "mission_type": mission_type,
        "mission_label": config.get("label", mission_type),
        "stages": _default_stages(),
        "sector_tags": sector_tags,
        "anomaly_tags": anomaly_tags,
    }


def remember_current_mission(memory, mission: dict[str, object]) -> str:
    text = (
        f"{mission['title']}. Sector: {mission['sector']}. Threat: {mission['threat']}. "
        f"Objective: {mission['objective']}. Anomaly: {mission['anomaly']}. "
        f"Sponsor: {mission['sponsor']}. Artifact target: {mission['artifact']['name']}. "
        f"Rival: {mission['rival']}. Directive: {mission['directive']} "
        f"Brief: {mission['brief']}"
    )
    payload = {
        "type": "mission_brief",
        "sector": mission["sector"],
        "sector_tags": list(mission.get("sector_tags", [])),
        "threat": mission["threat"],
        "objective": mission["objective"],
        "difficulty": mission["difficulty"],
        "sponsor": mission["sponsor"],
        "rival": mission["rival"],
        "mission_type": mission.get("mission_type", "standard"),
        "anomaly_tags": list(mission.get("anomaly_tags", [])),
        "stages": list(mission.get("stages", [])),
        "artifact_name": mission["artifact"]["name"],
        "squad_name": st.session_state["squad_name"],
        "operator_name": st.session_state["operator_name"],
    }
    return memory.save(text=text, payload=payload)


def _adjust_faction(name: str, success: bool) -> None:
    for faction in st.session_state["factions"]:
        if faction["name"] != name:
            continue
        faction["pressure"] = max(0, min(100, faction["pressure"] + (-6 if success else 7)))
        faction["leverage"] = max(0, min(100, faction["leverage"] + (5 if success else -4)))
        faction["last_outcome"] = "Success" if success else "Compromised"
        if faction["pressure"] >= 70:
            faction["stance"] = "Aggressive"
        elif faction["pressure"] >= 45:
            faction["stance"] = "Wary"
        elif faction["pressure"] <= 20:
            faction["stance"] = "Aligned"
        else:
            faction["stance"] = "Transactional"
        break


def _ensure_campaign_arc(faction: dict[str, Any]) -> None:
    """Stable multi-cycle intent line for retrieval + UI (long-arc play)."""
    if faction.get("campaign_arc"):
        return
    name = faction.get("name", "Unknown")
    interest = faction.get("interest", "")
    if interest == "Artifact extraction":
        faction["campaign_arc"] = (
            f"{name}: map two high-signal sectors, bank leverage, then one major extraction push—not every cycle."
        )
    elif interest == "Knowledge capture":
        faction["campaign_arc"] = (
            f"{name}: stabilize archive-adjacent reads before contesting hot relic lanes."
        )
    else:
        faction["campaign_arc"] = (
            f"{name}: lock two transit choke sectors before chaining sabotage; deny first, spike second."
        )


def _build_faction_memory_query(faction: dict[str, Any], world: dict[str, Any]) -> str:
    _ensure_campaign_arc(faction)
    arc = str(faction.get("campaign_arc", ""))
    tc = int(world.get("threat_clock", 0))
    threat_note = "low threat" if tc < 35 else "mid threat" if tc < 65 else "high threat"
    recent = " ".join(faction.get("recent_memory", [])[-6:])
    return (
        f"{faction['name']} {faction['interest']} {faction['goal']} "
        f"Arc: {arc} Cycle {int(world.get('cycle', 0))} ({threat_note}). {recent}"
    )


def _record_faction_memory(name: str, text: str) -> None:
    for faction in st.session_state["factions"]:
        if faction["name"] == name:
            faction["recent_memory"] = (faction.get("recent_memory", []) + [text])[-8:]
            break


def _shift_rival_pressure(name: str, success: bool) -> None:
    for faction in st.session_state["factions"]:
        if faction["name"] != name:
            continue
        faction["pressure"] = max(0, min(100, faction["pressure"] + (4 if success else -3)))
        faction["leverage"] = max(0, min(100, faction["leverage"] + (2 if success else -2)))
        if faction["pressure"] >= 70:
            faction["stance"] = "Aggressive"
        elif faction["pressure"] >= 45:
            faction["stance"] = "Wary"
        elif faction["pressure"] <= 20:
            faction["stance"] = "Aligned"
        else:
            faction["stance"] = "Transactional"
        break


def _resolve_mission_for_actor(
    *,
    memory,
    mission: dict[str, object],
    actor_name: str,
    committed_supplies: int,
    history_type: str,
    advance_cycle: bool,
    forced_outcome: str | None = None,
    outcome_bonus_intel: int = 0,
    artifact_bonus: int = 0,
    stage_results_override: list[dict[str, Any]] | None = None,
    deltas_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    world = st.session_state["world_state"]
    sector_state = st.session_state["sector_control"][mission["sector"]]
    config = MISSION_TYPES.get(str(mission.get("mission_type", "standard")), MISSION_TYPES["standard"])

    committed = max(0, min(2, int(committed_supplies), int(world.get("supplies", 0))))

    difficulty_weight = {
        "Low": 0.85,
        "Elevated": 0.7,
        "Severe": 0.52,
        "Terminal": 0.35,
    }
    base_success = difficulty_weight[mission["difficulty"]]
    base_success += min(world["intel"], 30) / 100.0
    base_success += min(len(st.session_state["inventory"]), 4) * 0.03
    base_success -= min(max(sector_state.get("heat", 0), 0), 100) / 200.0
    base_success += committed * 0.05
    base_success = max(0.08, min(base_success, 0.95))

    stage_results: list[dict[str, Any]] = []
    if forced_outcome in {"Success", "Compromised"}:
        staged_success = forced_outcome == "Success"
        if isinstance(stage_results_override, list) and stage_results_override:
            stage_results = list(stage_results_override)
        else:
            stage_results = [{"stage": "Encounter", "success": staged_success, "chance": round(base_success, 3)}]
    else:
        stages = list(mission.get("stages") or _default_stages())
        stage_mods: dict[str, float] = dict(config.get("stage_mods", {}))
        staged_success = True
        for stage in stages:
            name = str(stage.get("name", "Stage"))
            mod = float(stage_mods.get(name, -0.02))
            chance = max(0.05, min(0.97, base_success + mod))
            ok = random.random() < chance
            stage_results.append({"stage": name, "success": ok, "chance": round(chance, 3)})
            if not ok:
                staged_success = False
                break

    if advance_cycle:
        world["cycle"] += 1
    # Base mission cost always applies (campaign pacing).
    world["supplies"] = max(0, world["supplies"] - 1 - committed)
    world["sector_stability"] = max(0, world["sector_stability"] - mission["stability_cost"])
    sector_state["heat"] = min(100, sector_state["heat"] + int(config.get("heat_delta", 8)))
    sector_state["stability"] = max(0, sector_state["stability"] - mission["stability_cost"])

    # If an encounter provides explicit deltas, apply them as the authoritative result layer.
    if isinstance(deltas_override, dict):
        intel_delta = int(deltas_override.get("intel_delta", 0) or 0)
        supplies_delta = int(deltas_override.get("supplies_delta", 0) or 0)
        heat_delta = int(deltas_override.get("heat_delta", 0) or 0)
        stability_delta = int(deltas_override.get("stability_delta", 0) or 0)
        threat_delta = int(deltas_override.get("threat_delta", 0) or 0)

        world["intel"] = max(0, int(world.get("intel", 0)) + intel_delta)
        world["supplies"] = max(0, int(world.get("supplies", 0)) + supplies_delta)
        world["sector_stability"] = max(0, min(100, int(world.get("sector_stability", 0)) + stability_delta))
        world["threat_clock"] = max(0, min(100, int(world.get("threat_clock", 0)) + threat_delta))
        sector_state["heat"] = max(0, min(100, int(sector_state.get("heat", 0)) + heat_delta))
        sector_state["stability"] = max(0, min(100, int(sector_state.get("stability", 0)) + stability_delta))

    if committed:
        append_log(f"Committed {committed} extra supplies to improve expedition odds.")

    def _mark_sector(sector: str, kind: str, detail: str = "") -> None:
        st.session_state["sector_activity"][sector] = {
            "cycle": int(world.get("cycle", 0)),
            "kind": kind,
            "detail": detail,
        }

    # Stage-triggered consequences: inject distinct events that persist beyond the roll.
    # If an encounter provided stage results, this logic becomes deterministic and tied to that stage list.
    failure_stage = next((item["stage"] for item in stage_results if not item.get("success")), None)
    if failure_stage == "Approach":
        world["supplies"] = max(0, world["supplies"] - 1)
        _push_event(
            {
                "cycle": world["cycle"],
                "name": "Staging Slip",
                "severity": "Low",
                "sector": mission["sector"],
                "source_faction": actor_name,
                "target_faction": mission["sponsor"],
                "summary": "The squad lost time and equipment before contact. One extra supply was burned stabilizing the approach.",
                "why_now": "The entry geometry shifted and forced a rapid re-route.",
                "player_impact": "Supplies are tighter for the next cycle; consider Survey or Resupply behavior.",
            }
        )
    elif failure_stage == "Contact":
        world["intel"] = max(0, world["intel"] - 3)
        _push_event(
            {
                "cycle": world["cycle"],
                "name": "Injury Report",
                "severity": "Elevated",
                "sector": mission["sector"],
                "source_faction": actor_name,
                "target_faction": mission["sponsor"],
                "summary": "First contact produced an injury and scrambled field notes. Some intel was lost during extraction.",
                "why_now": "The anomaly rules were misread during the contact window.",
                "player_impact": "Intel gain is partially negated; consider running a Survey mission next to rebuild understanding.",
            }
        )
    elif failure_stage == "Objective":
        sector_state["heat"] = min(100, sector_state.get("heat", 0) + 6)
        _push_event(
            {
                "cycle": world["cycle"],
                "name": "Gear Loss",
                "severity": "Severe",
                "sector": mission["sector"],
                "source_faction": st.session_state["squad_name"],
                "target_faction": mission["rival"],
                "summary": "The objective zone consumed critical kit. Sector heat spiked as the squad fought to recover and disengage.",
                "why_now": "Overlapping anomaly rules forced a retreat without stabilizing the objective site.",
                "player_impact": "Higher heat increases risk on future expeditions in this sector and attracts faction attention.",
            }
        )
    elif failure_stage == "Extraction":
        _push_event(
            {
                "cycle": world["cycle"],
                "name": "Pursuit Echo",
                "severity": "Elevated",
                "sector": mission["sector"],
                "source_faction": mission["rival"],
                "target_faction": st.session_state["squad_name"],
                "summary": "A rival-aligned pursuit pattern shadowed the squad out of the sector, escalating political pressure.",
                "why_now": "The exit corridor was compromised and exposed the squad’s route.",
                "player_impact": "Rival pressure is likely to rise and a counter-mission is more likely to appear soon.",
            }
        )

    artifact_found = 0
    encounter_used = isinstance(deltas_override, dict)
    if staged_success:
        _mark_sector(str(mission["sector"]), "player_success" if history_type == "player_expedition" else "ai_success")
        # If an encounter was used, we already applied deltas above; add baseline reward and bonuses here.
        world["intel"] += int(mission.get("reward_intel", 0)) + int(outcome_bonus_intel)
        sector_state["controller"] = mission["sponsor"]
        sector_state["interest"][mission["sponsor"]] += 8
        sector_state["interest"][mission["rival"]] = max(0, sector_state["interest"][mission["rival"]] - 3)
        if encounter_used:
            # Deterministic: encounter decides whether an artifact was secured.
            if int(artifact_bonus) > 0:
                st.session_state["inventory"].append(
                    {
                        **mission["artifact"],
                        "source_sector": mission["sector"],
                        "cycle_found": world["cycle"],
                    }
                )
                artifact_found = 1
        else:
            artifact_chance = float(config.get("artifact_chance", 0.55))
            if random.random() < artifact_chance:
                st.session_state["inventory"].append(
                    {
                        **mission["artifact"],
                        "source_sector": mission["sector"],
                        "cycle_found": world["cycle"],
                    }
                )
                artifact_found = 1
        _adjust_faction(mission["sponsor"], success=True)
        _shift_rival_pressure(mission["rival"], success=True)
        outcome = "Success"
        summary = (
            f"{actor_name} stabilized {mission['sector']} and secured new insight "
            f"under pressure from {mission['sponsor']} while blunting {mission['rival']}."
        )
    else:
        _mark_sector(str(mission["sector"]), "player_fail" if history_type == "player_expedition" else "ai_fail")
        sector_state["controller"] = mission["rival"]
        sector_state["interest"][mission["rival"]] += 7
        _adjust_faction(mission["sponsor"], success=False)
        _shift_rival_pressure(mission["rival"], success=False)
        outcome = "Compromised"
        summary = (
            f"The expedition into {mission['sector']} fractured early. {mission['sponsor']} gained leverage "
            f"while {mission['rival']} repositioned and the ruin escalated."
        )

    sponsor_memory = (
        f"Cycle {world['cycle']}: {outcome} at {mission['sector']} pursuing {mission['objective']}."
    )
    rival_memory = (
        f"Cycle {world['cycle']}: {mission['sponsor']} operation in {mission['sector']} ended in {outcome.lower()}."
    )
    _record_faction_memory(mission["sponsor"], sponsor_memory)
    _record_faction_memory(mission["rival"], rival_memory)

    history_item = {
        "cycle": world["cycle"],
        "outcome": outcome,
        "type": history_type,
        "actor": actor_name,
        "mission_id": mission.get("mission_id", ""),
        "sector": mission["sector"],
        "objective": mission["objective"],
        "sponsor": mission["sponsor"],
        "rival": mission["rival"],
        "mission_type": mission.get("mission_type", "standard"),
        "sector_tags": list(mission.get("sector_tags", [])),
        "anomaly_tags": list(mission.get("anomaly_tags", [])),
        "stage_results": stage_results,
        "committed_supplies": committed,
        "reward_intel": int(mission.get("reward_intel", 0)),
        "artifact_found": int(artifact_found),
        "summary": summary,
    }
    st.session_state["history"].append(history_item)
    if mission.get("mission_type") == "counter" and st.session_state["counter_missions"]:
        st.session_state["counter_missions"].pop(0)

    memory_text = (
        f"Cycle {history_item['cycle']} outcome {outcome}. Sector {mission['sector']}. Objective "
        f"{mission['objective']}. Sponsor {mission['sponsor']}. Summary: {summary}"
    )
    memory.save(
        text=memory_text,
        payload={
            "type": "mission_outcome",
            "cycle": history_item["cycle"],
            "outcome": outcome,
            "sector": mission["sector"],
            "sponsor": mission["sponsor"],
            "mission_type": history_item["mission_type"],
            "sector_tags": history_item["sector_tags"],
            "anomaly_tags": history_item["anomaly_tags"],
            "stage_results": stage_results,
            "actor": actor_name,
            "history_type": history_type,
        },
    )

    append_log(summary)
    record_snapshot("mission_resolution")
    evaluate_endgame()
    persist_state()
    return history_item


def resolve_current_mission(memory) -> dict[str, Any]:
    snapshot_for_undo("resolve_expedition")
    mission = st.session_state["current_mission"]
    committed = int(st.session_state.get("mission_commit_supplies", 0) or 0)
    encounter = st.session_state.get("pending_encounter")
    forced_outcome = None
    bonus_intel = 0
    artifact_bonus = 0
    stage_results_override = None
    deltas = None
    if isinstance(encounter, dict) and encounter.get("mission_id") == mission.get("mission_id"):
        forced_outcome = "Success" if bool(encounter.get("success")) else "Compromised"
        # v2 encounter contract preferred
        deltas = encounter.get("deltas")
        if isinstance(deltas, dict):
            bonus_intel = int(deltas.get("intel_delta", 0) or 0)
        else:
            bonus_intel = int(encounter.get("bonus_intel", 0) or 0)
        artifact_bonus = int(encounter.get("artifact_bonus", 0) or 0)
        stage_results_override = encounter.get("stage_results")
    history_item = _resolve_mission_for_actor(
        memory=memory,
        mission=mission,
        actor_name=st.session_state["squad_name"],
        committed_supplies=committed,
        history_type="player_expedition",
        advance_cycle=True,
        forced_outcome=forced_outcome,
        outcome_bonus_intel=bonus_intel,
        artifact_bonus=artifact_bonus,
        stage_results_override=stage_results_override,
        deltas_override=deltas,
    )
    st.session_state["current_mission"] = build_mission(st.session_state["squad_name"])
    st.session_state["last_resolution"] = history_item
    st.session_state["mission_commit_supplies"] = 0
    st.session_state["pending_encounter"] = None
    st.session_state.pop("_topology_fig_cache", None)
    st.session_state.pop("_topology_globe_fig_cache", None)
    return history_item


def _max_commit_for_mission(*, mission_type: str, supplies: int) -> int:
    cap = 0
    if mission_type in {"extraction", "containment", "standard", "counter"}:
        cap = 2
    elif mission_type == "survey":
        cap = 1
    return max(0, min(int(cap), int(supplies)))


def ensure_autonomous_stack_defaults() -> dict[str, Any]:
    """Default fully autonomous stack: AI player turns + faction ML/LLM + embedded rival learning."""
    defaults = {
        "enabled": True,
        "player_expeditions_per_tick": 1,
        "faction_rounds_per_tick": 1,
        "use_ollama_player_commit": True,
    }
    cur = st.session_state.get("autonomous_stack")
    if not isinstance(cur, dict):
        st.session_state["autonomous_stack"] = dict(defaults)
        return st.session_state["autonomous_stack"]
    merged = {**defaults, **cur}
    st.session_state["autonomous_stack"] = merged
    return merged


def choose_player_autopilot_commit(world: dict[str, Any], mission: dict[str, object]) -> int:
    """Heuristic commit policy for player autopilot (keeps campaigns moving without hard-spiking supply burn)."""
    supplies = int(world.get("supplies", 0) or 0)
    if supplies <= 1:
        return 0
    mission_type = str(mission.get("mission_type", "standard"))
    max_commit = _max_commit_for_mission(mission_type=mission_type, supplies=supplies)
    if max_commit <= 0:
        return 0
    difficulty = str(mission.get("difficulty", "Elevated"))
    threat = int(world.get("threat_clock", 0) or 0)
    stability = int(world.get("sector_stability", 0) or 0)

    # Hard missions: spend to keep the run alive.
    if difficulty in {"Terminal", "Severe"}:
        return max_commit
    if stability <= 25 and max_commit >= 1:
        return 1
    if threat >= 70 and max_commit >= 1:
        return 1
    if difficulty == "Elevated" and max_commit >= 1 and supplies >= 4:
        return 1
    return 0


def autoplay_player_turn(
    memory,
    decision_engine: OllamaDecisionEngine | None = None,
) -> dict[str, Any]:
    """Run one fully automated player expedition: heuristic commit, optional Ollama assist, then resolve."""
    ensure_autonomous_stack_defaults()
    mission = st.session_state["current_mission"]
    world = st.session_state["world_state"]
    mission_type = str(mission.get("mission_type", "standard"))
    supplies = int(world.get("supplies", 0) or 0)
    max_commit = _max_commit_for_mission(mission_type=mission_type, supplies=supplies)
    commit = choose_player_autopilot_commit(world, mission)
    stack = st.session_state.get("autonomous_stack", {})
    if (
        decision_engine is not None
        and stack.get("use_ollama_player_commit", True)
        and decision_engine.is_available()
        and max_commit > 0
    ):
        llm_c, _st = propose_expedition_commit_supplies(
            decision_engine, mission, world, max_commit=max_commit
        )
        if llm_c is not None:
            commit = llm_c
    st.session_state["mission_commit_supplies"] = commit
    # Autopilot ignores encounter mini-games; it resolves directly.
    st.session_state["pending_encounter"] = None
    return resolve_current_mission(memory)


def run_autonomous_stack_tick(
    memory,
    decision_engine: OllamaDecisionEngine,
) -> dict[str, Any]:
    """
    One autonomous tick: optional AI player resolution(s), then faction round(s).
    Rival ML units still advance inside ``simulate_agent_round`` (``simulate_ai_competitors``).
    """
    stack = ensure_autonomous_stack_defaults()
    out: dict[str, Any] = {"player": [], "factions": []}
    if not stack.get("enabled", True):
        return out
    if st.session_state["endgame"]["ended"]:
        return out

    pe = max(0, min(3, int(stack.get("player_expeditions_per_tick", 1))))
    for _ in range(pe):
        if st.session_state["endgame"]["ended"]:
            break
        out["player"].append(autoplay_player_turn(memory, decision_engine=decision_engine))

    fr = max(1, min(8, int(stack.get("faction_rounds_per_tick", 1))))
    if not st.session_state["endgame"]["ended"]:
        out["factions"] = simulate_agent_round(memory, decision_engine, rounds=fr)
    return out


def ensure_rival_agents_for_swarm() -> None:
    """Ensure rival expedition units exist so each faction round advances competitor learning."""
    init_rival_learning()
    cfg = st.session_state.setdefault("ai_config", {"num_agents": 3})
    n = max(1, min(6, int(cfg.get("num_agents", 3) or 3)))
    cfg["num_agents"] = n
    if len(st.session_state.get("ai_agents", [])) != n:
        configure_ai_agents(n)


def apply_ml_swarm_presets(*, max_throughput: bool = True, mas_bandit_only: bool = False) -> None:
    """
    Tune the autonomous stack for dense end-to-end ticks: player autoplay + many faction cycles.
    If mas_bandit_only, every faction MAS slot is set to bandit (local learning, no per-faction LLM).
    """
    stack = ensure_autonomous_stack_defaults()
    stack["enabled"] = True
    if max_throughput:
        stack["player_expeditions_per_tick"] = 3
        stack["faction_rounds_per_tick"] = 8
    st.session_state["autonomous_stack"] = stack
    ensure_rival_agents_for_swarm()
    if mas_bandit_only:
        init_mas()
        for slot in st.session_state["mas"].get("agents", {}).values():
            if isinstance(slot, dict):
                slot["policy"] = "bandit"


def run_ml_swarm_until_endgame(
    memory,
    decision_engine: OllamaDecisionEngine,
    *,
    max_stack_ticks: int = 2000,
    max_throughput: bool = True,
    mas_bandit_only: bool = False,
    restore_stack_after: bool = True,
    restore_mas_after: bool = True,
) -> dict[str, Any]:
    """
    Run autonomous stack ticks until an endgame fires or max_stack_ticks is reached.
    Player expeditions resolve without encounter UI; factions negotiate + act each inner round.

    Optionally restores prior ``autonomous_stack`` / ``mas`` so the dashboard returns to user presets.
    """
    stop_autoplay()
    evaluate_endgame()
    if st.session_state["endgame"]["ended"]:
        return {
            "ticks": 0,
            "player_resolutions": 0,
            "faction_rows": 0,
            "ended": True,
            "result": st.session_state["endgame"].get("result"),
            "hit_cap": False,
        }

    prev_stack = copy.deepcopy(ensure_autonomous_stack_defaults()) if restore_stack_after else None
    prev_mas = copy.deepcopy(st.session_state.get("mas")) if (mas_bandit_only and restore_mas_after) else None
    total_p = total_f = 0
    ticks = 0
    hit_cap = False
    try:
        apply_ml_swarm_presets(max_throughput=max_throughput, mas_bandit_only=mas_bandit_only)
        cap = max(1, int(max_stack_ticks))
        while ticks < cap:
            if st.session_state["endgame"]["ended"]:
                break
            tick = run_autonomous_stack_tick(memory, decision_engine)
            total_p += len(tick.get("player", []))
            total_f += len(tick.get("factions", []))
            ticks += 1
            evaluate_endgame()
            if st.session_state["endgame"]["ended"]:
                break
        hit_cap = ticks >= cap and not st.session_state["endgame"]["ended"]
    finally:
        if prev_stack is not None:
            st.session_state["autonomous_stack"] = prev_stack
        if prev_mas is not None:
            st.session_state["mas"] = prev_mas
            init_mas()

    result = st.session_state["endgame"].get("result") if st.session_state["endgame"]["ended"] else None
    _tail = "Cap hit — still Active." if hit_cap else f"Outcome: {result or 'n/a'}."
    append_log(
        f"ML swarm finished after {ticks} stack tick(s): "
        f"{total_p} player resolution(s), {total_f} faction row(s). {_tail}"
    )
    return {
        "ticks": ticks,
        "player_resolutions": total_p,
        "faction_rows": total_f,
        "ended": bool(st.session_state["endgame"]["ended"]),
        "result": result,
        "hit_cap": hit_cap,
    }


def _rewire_allowed_targets() -> list[str]:
    g = current_sector_graph()
    out: list[str] = []
    for a in SECTORS:
        for b in g.get(a, []):
            if a < b:
                out.append(f"cut|{a}|{b}")
    for a in SECTORS:
        for b in SECTORS:
            if a >= b:
                continue
            if b in g.get(a, []):
                continue
            out.append(f"bridge|{a}|{b}")
    return sorted(out)


def _choose_action(faction: dict[str, Any], world: dict[str, Any]) -> tuple[str, str]:
    action_scores = {
        "extract": 1.0,
        "research": 1.0,
        "sabotage": 1.0,
        "fortify": 1.0,
        "resupply": 1.0,
        "rewire": 0.45,
    }

    if faction["interest"] == "Artifact extraction":
        action_scores["extract"] += 2.2
        action_scores["sabotage"] += 0.6
    if faction["interest"] == "Knowledge capture":
        action_scores["research"] += 2.4
        action_scores["fortify"] += 0.5
    if faction["interest"] == "Territorial denial":
        action_scores["sabotage"] += 2.5
        action_scores["fortify"] += 0.8
        action_scores["rewire"] += 1.9

    if world["supplies"] <= 2:
        action_scores["resupply"] += 1.8
    if world["sector_stability"] <= 30:
        action_scores["fortify"] += 1.5
        action_scores["research"] += 0.7
    if faction["pressure"] >= 65:
        action_scores["sabotage"] += 1.2
    if faction["pressure"] >= 58:
        action_scores["rewire"] += 1.4
    if faction["leverage"] <= 20:
        action_scores["extract"] += 0.5
        action_scores["research"] += 0.5
    if faction["leverage"] >= 52:
        action_scores["rewire"] += 1.0

    action = max(action_scores, key=action_scores.get)
    if action in ("extract", "fortify"):
        target = random.choice(faction["preferred_sectors"])
    elif action == "research":
        target = random.choice(["Archive Chamber Theta", "Thermal Lens Cathedral", "Vault Spine 3"])
    elif action == "resupply":
        target = "Expedition Hub"
    elif action == "rewire":
        opts = _rewire_allowed_targets()
        if not opts:
            return "resupply", "Expedition Hub"
        target = random.choice(opts)
    else:
        target = faction["rival"]
    return action, target


def _allowed_targets(action: str, faction: dict[str, Any]) -> list[str]:
    if action == "sabotage":
        return [other["name"] for other in st.session_state["factions"] if other["name"] != faction["name"]]
    if action == "resupply":
        return ["Expedition Hub"]
    if action == "rewire":
        return _rewire_allowed_targets()
    return list(SECTORS)


def _allowed_partners(faction: dict[str, Any]) -> list[str]:
    return [other["name"] for other in st.session_state["factions"] if other["name"] != faction["name"]]


def _choose_negotiation_move(faction: dict[str, Any], partner: dict[str, Any]) -> str:
    scores = {
        "threaten": 1.0,
        "bargain": 1.0,
        "align": 1.0,
        "deceive": 1.0,
    }
    if faction["pressure"] >= 60:
        scores["threaten"] += 1.6
    if faction["leverage"] <= 25:
        scores["bargain"] += 1.2
    if partner["name"] != faction["rival"]:
        scores["align"] += 1.4
    if faction["interest"] == "Knowledge capture":
        scores["deceive"] += 0.8
    return max(scores, key=scores.get)


def _build_negotiation_prompt(
    faction: dict[str, Any],
    partner: dict[str, Any],
    world: dict[str, Any],
    heuristic_move: str,
) -> str:
    return f"""
You are a faction negotiator for the game Threadfall.
Reply with valid JSON only.

Choose exactly one move from:
threaten, bargain, align, deceive

Source faction:
- name: {faction['name']}
- interest: {faction['interest']}
- goal: {faction['goal']}
- stance: {faction['stance']}
- pressure: {faction['pressure']}
- leverage: {faction['leverage']}
- rival: {faction['rival']}

Negotiation partner:
- name: {partner['name']}
- interest: {partner['interest']}
- stance: {partner['stance']}
- pressure: {partner['pressure']}
- leverage: {partner['leverage']}

World state:
- cycle: {world['cycle']}
- intel: {world['intel']}
- supplies: {world['supplies']}
- threat_clock: {world['threat_clock']}
- sector_stability: {world['sector_stability']}

Heuristic fallback:
- move: {heuristic_move}

Return JSON with this shape:
{{"move":"bargain","terms":"short proposal","confidence":0.72}}

Rules:
- terms must be under 25 words
- confidence must be between 0 and 1
- no markdown
""".strip()


def _validated_llm_negotiation(
    decision_engine: OllamaDecisionEngine,
    faction: dict[str, Any],
    partner: dict[str, Any],
    world: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    heuristic_move = _choose_negotiation_move(faction, partner)
    fallback = {
        "move": heuristic_move,
        "terms": f"{faction['name']} defaulted to a cautious {heuristic_move} posture.",
        "confidence": 0.53,
        "source": "heuristic",
    }
    prompt = _build_negotiation_prompt(faction, partner, world, heuristic_move)
    schema = {
        "type": "object",
        "properties": {
            "move": {"type": "string", "enum": ["threaten", "bargain", "align", "deceive"]},
            "terms": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["move", "terms", "confidence"],
    }
    response, status = decision_engine.generate_json(prompt, schema=schema)
    if response is None:
        fallback["terms"] = f"{faction['name']} used heuristic negotiation. Reason: {status}."
        return fallback, status

    move = str(response.get("move", "")).strip()
    if move not in {"threaten", "bargain", "align", "deceive"}:
        fallback["terms"] = f"{faction['name']} rejected invalid negotiation move '{move}'."
        return fallback, "invalid negotiation move"

    terms = str(response.get("terms", "")).strip() or fallback["terms"]
    words = terms.split()
    if len(words) > 25:
        terms = " ".join(words[:25]).strip() + "..."
    return {
        "move": move,
        "terms": terms,
        "confidence": max(0.0, min(1.0, float(response.get("confidence", 0.5)))),
        "source": decision_engine.model,
    }, "ok"


def _apply_negotiation(
    faction: dict[str, Any],
    partner: dict[str, Any],
    move: str,
) -> dict[str, int]:
    effects = {
        "source_pressure": 0,
        "source_leverage": 0,
        "partner_pressure": 0,
        "partner_leverage": 0,
        "supplies_delta": 0,
        "intel_delta": 0,
    }
    if move == "threaten":
        effects["source_leverage"] += 2
        effects["partner_pressure"] += 4
        effects["source_pressure"] += 1
    elif move == "bargain":
        effects["source_pressure"] -= 2
        effects["partner_pressure"] -= 1
        effects["source_leverage"] += 1
        effects["supplies_delta"] += 1
    elif move == "align":
        effects["source_pressure"] -= 3
        effects["partner_pressure"] -= 3
        effects["source_leverage"] += 1
        effects["partner_leverage"] += 1
        effects["intel_delta"] += 1
    else:
        effects["source_leverage"] += 3
        effects["partner_pressure"] += 2
        effects["source_pressure"] += 1

    faction["pressure"] = max(0, min(100, faction["pressure"] + effects["source_pressure"]))
    faction["leverage"] = max(0, min(100, faction["leverage"] + effects["source_leverage"]))
    partner["pressure"] = max(0, min(100, partner["pressure"] + effects["partner_pressure"]))
    partner["leverage"] = max(0, min(100, partner["leverage"] + effects["partner_leverage"]))
    st.session_state["world_state"]["supplies"] = max(
        0, st.session_state["world_state"]["supplies"] + effects["supplies_delta"]
    )
    st.session_state["world_state"]["intel"] = max(
        0, st.session_state["world_state"]["intel"] + effects["intel_delta"]
    )
    return effects


def _log_negotiation(item: dict[str, Any]) -> None:
    st.session_state["negotiations"] = (st.session_state["negotiations"] + [item])[-40:]


def _run_negotiation_phase(memory, decision_engine: OllamaDecisionEngine, faction: dict[str, Any]) -> dict[str, Any]:
    world = st.session_state["world_state"]
    partner_name = faction["rival"] if random.random() < 0.7 else random.choice(_allowed_partners(faction))
    partner = _find_faction(partner_name)
    negotiation, status = _validated_llm_negotiation(
        decision_engine=decision_engine,
        faction=faction,
        partner=partner,
        world=world,
    )
    effects = _apply_negotiation(faction, partner, negotiation["move"])
    summary = (
        f"{faction['name']} used {negotiation['move']} with {partner['name']}. "
        f"{negotiation['terms']}"
    )
    item = {
        "cycle": world["cycle"],
        "source": faction["name"],
        "partner": partner["name"],
        "move": negotiation["move"],
        "terms": negotiation["terms"],
        "confidence": round(float(negotiation["confidence"]), 2),
        "source_model": negotiation["source"],
        "status": status,
        "effects": effects,
    }
    _log_negotiation(item)
    _record_faction_memory(faction["name"], f"Cycle {world['cycle']}: negotiated with {partner['name']} via {negotiation['move']}.")
    _record_faction_memory(partner["name"], f"Cycle {world['cycle']}: received {negotiation['move']} from {faction['name']}.")
    memory.save(
        text=f"Negotiation. Cycle {world['cycle']}. {summary}",
        payload={
            "type": "negotiation",
            "cycle": world["cycle"],
            "source": faction["name"],
            "partner": partner["name"],
            "move": negotiation["move"],
            "source_model": negotiation["source"],
        },
    )
    return item


def _sector_counts() -> dict[str, int]:
    counts = {"Neutral": 0}
    for faction in FACTIONS:
        counts[faction] = 0
    for info in st.session_state["sector_control"].values():
        controller = info["controller"]
        counts[controller] = counts.get(controller, 0) + 1
    return counts


def _campaign_detail_lines() -> list[str]:
    counts = _sector_counts()
    top_faction = max((f for f in FACTIONS), key=lambda name: counts.get(name, 0))
    return [
        f"Cycle reached: {st.session_state['world_state']['cycle']}",
        f"Recovered artifacts: {len(st.session_state['inventory'])}",
        f"Top controlling faction: {top_faction} ({counts.get(top_faction, 0)} sectors)",
        f"Threat clock: {st.session_state['world_state']['threat_clock']}",
        f"Sector stability: {st.session_state['world_state']['sector_stability']}",
    ]


def record_snapshot(label: str) -> None:
    world = st.session_state["world_state"]
    counts = _sector_counts()
    factions = {f["name"]: f for f in st.session_state["factions"]}
    snapshot = {
        "cycle": world["cycle"],
        "label": label,
        "intel": world["intel"],
        "supplies": world["supplies"],
        "threat_clock": world["threat_clock"],
        "sector_stability": world["sector_stability"],
        "neutral_control": counts.get("Neutral", 0),
    }
    for name in FACTIONS:
        snapshot[f"{name}_pressure"] = factions[name]["pressure"]
        snapshot[f"{name}_leverage"] = factions[name]["leverage"]
        snapshot[f"{name}_control"] = counts.get(name, 0)

    existing = st.session_state.get("snapshots", [])
    if existing and existing[-1]["cycle"] == snapshot["cycle"] and existing[-1]["label"] == label:
        existing[-1] = snapshot
        st.session_state["snapshots"] = existing[-120:]
        return

    st.session_state["snapshots"] = (existing + [snapshot])[-120:]


def _finalize_endgame(result: str, winner: str, summary: str) -> None:
    st.session_state["endgame"] = {
        "ended": True,
        "result": result,
        "winner": winner,
        "trigger_cycle": st.session_state["world_state"]["cycle"],
        "summary": summary,
        "details": _campaign_detail_lines(),
    }
    stop_autoplay()
    append_log(f"Campaign ended: {result}. {summary}")
    persist_state()


def evaluate_endgame() -> dict[str, Any]:
    current = st.session_state["endgame"]
    if current["ended"]:
        return current

    world = st.session_state["world_state"]
    for agent in st.session_state.get("ai_agents", []):
        if int(agent.get("intel", 0)) >= 200 and int(agent.get("artifacts", 0)) >= 8:
            _finalize_endgame(
                result="Rival Victory",
                winner=str(agent.get("name", "Rival Unit")),
                summary=(
                    f"{agent.get('name', 'A rival unit')} reached the breakthrough thresholds first and locked in control of the ruin."
                ),
            )
            return st.session_state["endgame"]

    counts = _sector_counts()
    sector_leader = max((f for f in FACTIONS), key=lambda name: counts.get(name, 0))
    leader_faction = _find_faction(sector_leader)

    if world["intel"] >= 200 and len(st.session_state["inventory"]) >= 8:
        _finalize_endgame(
            result="Player Victory",
            winner=st.session_state["squad_name"],
            summary=(
                f"{st.session_state['squad_name']} assembled enough knowledge and relic control "
                f"to outmaneuver every faction in the ruin."
            ),
        )
    elif world["threat_clock"] >= 100 or world["sector_stability"] <= 0:
        _finalize_endgame(
            result="Player Defeat",
            winner="The Ruin",
            summary="The ruin entered terminal escalation before the squad could stabilize it.",
        )
    elif counts.get(sector_leader, 0) >= 3 and leader_faction["leverage"] >= 60:
        _finalize_endgame(
            result="Faction Victory",
            winner=sector_leader,
            summary=(
                f"{sector_leader} converted regional control and leverage into full operational dominance."
            ),
        )
    elif world["supplies"] <= 0 and len(st.session_state["counter_missions"]) >= 3:
        _finalize_endgame(
            result="Player Defeat",
            winner="Rival Factions",
            summary="The expedition hub ran dry while rival factions forced too many simultaneous crises.",
        )
    elif world["cycle"] >= 80:
        _finalize_endgame(
            result="Stalemate",
            winner=sector_leader if counts.get(sector_leader, 0) > 0 else "None",
            summary="The campaign hardened into a long war of containment with no decisive breakthrough.",
        )

    return st.session_state["endgame"]


def _build_decision_prompt(
    faction: dict[str, Any],
    world: dict[str, Any],
    related_memories,
    heuristic_action: str,
    heuristic_target: str,
) -> str:
    memory_lines = []
    for item in related_memories[:2]:
        memory_lines.append(f"- score={item.score:.3f} text={item.text[:220]}")
    memory_block = "\n".join(memory_lines) if memory_lines else "- no relevant memory retrieved"

    sectors = []
    for sector, info in st.session_state["sector_control"].items():
        sectors.append(
            f"- {sector}: controller={info['controller']}, heat={info['heat']}, stability={info['stability']}"
        )
    rewire_opts = _allowed_targets("rewire", faction)
    rewire_hint = ", ".join(rewire_opts[:24]) + (" …" if len(rewire_opts) > 24 else "")
    allowed_lines = [
        f"- extract: {', '.join(_allowed_targets('extract', faction))}",
        f"- research: {', '.join(_allowed_targets('research', faction))}",
        f"- sabotage: {', '.join(_allowed_targets('sabotage', faction))}",
        f"- fortify: {', '.join(_allowed_targets('fortify', faction))}",
        f"- resupply: {', '.join(_allowed_targets('resupply', faction))}",
        f"- rewire (topology): {rewire_hint or 'none'}",
    ]

    return f"""
You are a faction decision engine for the game Threadfall.
Reply with valid JSON only.

Choose exactly one action from:
extract, research, sabotage, fortify, resupply, rewire

Faction state:
- name: {faction['name']}
- interest: {faction['interest']}
- goal: {faction['goal']}
- stance: {faction['stance']}
- pressure: {faction['pressure']}
- leverage: {faction['leverage']}
- rival: {faction['rival']}
- recent_memory: {' | '.join(faction.get('recent_memory', [])) or 'none'}

World state:
- cycle: {world['cycle']}
- intel: {world['intel']}
- supplies: {world['supplies']}
- threat_clock: {world['threat_clock']}
- sector_stability: {world['sector_stability']}

Sector control:
{chr(10).join(sectors)}

Retrieved memory:
{memory_block}

Allowed targets by action:
{chr(10).join(allowed_lines)}

Heuristic fallback suggestion:
- action: {heuristic_action}
- target: {heuristic_target}

Return JSON with this shape:
{{"action":"extract","target":"Vault Spine 3","reasoning":"short explanation","confidence":0.78}}

Rules:
- target must match one of the allowed targets for the chosen action
- for rewire, target must be exactly one string like cut|Sector A|Sector B or bridge|Sector A|Sector B (see allowed list)
- keep reasoning under 35 words
- confidence must be a number between 0 and 1
- no markdown
""".strip()


def _validated_llm_decision(
    decision_engine: OllamaDecisionEngine,
    faction: dict[str, Any],
    world: dict[str, Any],
    related_memories,
) -> tuple[DecisionProposal, str]:
    heuristic_action, heuristic_target = _choose_action(faction, world)
    heuristic = DecisionProposal(
        action=heuristic_action,
        target=heuristic_target,
        reasoning=(
            f"{faction['name']} followed the deterministic fallback because local model advice was unavailable."
        ),
        confidence=0.55,
        source="heuristic",
    )

    prompt = _build_decision_prompt(
        faction=faction,
        world=world,
        related_memories=related_memories,
        heuristic_action=heuristic_action,
        heuristic_target=heuristic_target,
    )
    allowed_targets = sorted(
        set(
            _allowed_targets("extract", faction)
            + _allowed_targets("research", faction)
            + _allowed_targets("sabotage", faction)
            + _allowed_targets("fortify", faction)
            + _allowed_targets("resupply", faction)
            + _allowed_targets("rewire", faction)
        )
    )
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["extract", "research", "sabotage", "fortify", "resupply", "rewire"],
            },
            "target": {"type": "string", "enum": allowed_targets},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["action", "target", "reasoning", "confidence"],
    }
    proposal, status = decision_engine.propose_decision(prompt, schema=schema)
    if proposal is None:
        heuristic.reasoning = (
            f"{faction['name']} used heuristic fallback. Local decision model status: {status}."
        )
        return heuristic, status

    allowed_actions = {"extract", "research", "sabotage", "fortify", "resupply", "rewire"}
    if proposal.action not in allowed_actions:
        heuristic.reasoning = (
            f"{faction['name']} rejected invalid model action '{proposal.action}' and used heuristic fallback."
        )
        return heuristic, "invalid action"

    if proposal.target not in _allowed_targets(proposal.action, faction):
        heuristic.reasoning = (
            f"{faction['name']} rejected invalid model target '{proposal.target}' and used heuristic fallback."
        )
        return heuristic, "invalid target"

    words = proposal.reasoning.split()
    if len(words) > 35:
        proposal.reasoning = " ".join(words[:35]).strip() + "..."

    return proposal, "ok"


def _agent_reasoning(faction: dict[str, Any], action: str, target: str, memories: list[Any]) -> str:
    memory_hint = "No high-confidence memory match."
    if memories:
        payload = memories[0].payload
        sector = payload.get("sector") or payload.get("sponsor") or "unknown pattern"
        memory_hint = f"Recent retrieved pattern pointed toward {sector}."
    if action == "extract":
        return (
            f"{faction['name']} is exploiting a relic window at {target}. "
            f"{memory_hint} Goal alignment favors immediate recovery."
        )
    if action == "research":
        return (
            f"{faction['name']} is prioritizing anomaly understanding in {target}. "
            f"{memory_hint} Better knowledge should reduce future uncertainty."
        )
    if action == "sabotage":
        return (
            f"{faction['name']} is moving against {target}. "
            f"{memory_hint} Pressure and rivalry now outweigh passive positioning."
        )
    if action == "fortify":
        return (
            f"{faction['name']} is reinforcing control around {target}. "
            f"{memory_hint} Sector fragility makes defense more valuable than expansion."
        )
    if action == "rewire":
        return (
            f"{faction['name']} is retuning corridor topology ({target}) to open a new angle of attack. "
            f"{memory_hint} Graph distance is leverage when the old routes are predictable."
        )
    return (
        f"{faction['name']} is rebuilding supplies through {target}. "
        f"{memory_hint} Resource recovery is needed before the next major push."
    )


def _apply_agent_action(faction: dict[str, Any], action: str, target: str) -> dict[str, int]:
    world = st.session_state["world_state"]
    sector_control = st.session_state["sector_control"]
    effects = {
        "intel_delta": 0,
        "supplies_delta": 0,
        "threat_delta": 0,
        "stability_delta": 0,
        "pressure_delta": 0,
        "leverage_delta": 0,
    }

    if action == "extract":
        effects["intel_delta"] += random.randint(2, 5)
        effects["supplies_delta"] -= 1
        effects["threat_delta"] += 1
        effects["leverage_delta"] += 4
        sector = target if target in sector_control else random.choice(faction["preferred_sectors"])
        sector_control[sector]["interest"][faction["name"]] += 6
        sector_control[sector]["heat"] = min(100, sector_control[sector]["heat"] + 6)
        _mark_sector_activity(sector, "extract", faction["name"])
        for nbr in _neighbors(sector):
            if nbr not in sector_control:
                continue
            sector_control[nbr]["heat"] = min(100, sector_control[nbr]["heat"] + 3)
            sector_control[nbr]["interest"][faction["name"]] += 2
            _mark_sector_activity(nbr, "extract_spill", faction["name"])
    elif action == "research":
        effects["intel_delta"] += random.randint(3, 6)
        effects["stability_delta"] += 2
        effects["leverage_delta"] += 2
        sector = target if target in sector_control else random.choice(faction["preferred_sectors"])
        sector_control[sector]["interest"][faction["name"]] += 5
        sector_control[sector]["stability"] = min(100, sector_control[sector]["stability"] + 3)
        _mark_sector_activity(sector, "research", faction["name"])
        for nbr in _neighbors(sector):
            if nbr not in sector_control:
                continue
            sector_control[nbr]["stability"] = min(100, sector_control[nbr]["stability"] + 1)
            _mark_sector_activity(nbr, "research_spill", faction["name"])
    elif action == "sabotage":
        effects["threat_delta"] += random.randint(2, 5)
        effects["stability_delta"] -= 2
        effects["pressure_delta"] += 4
        effects["leverage_delta"] += 3
        for rival in st.session_state["factions"]:
            if rival["name"] == target:
                rival["pressure"] = max(0, min(100, rival["pressure"] + 5))
                rival["leverage"] = max(0, min(100, rival["leverage"] - 3))
                break
        rival_name = target
        anchor = _strongest_interest_sector(rival_name) or random.choice(list(sector_control.keys()))
        contested_sector = _pick_nearest_sector(faction["preferred_sectors"], anchor)
        sector_control[contested_sector]["heat"] = min(100, sector_control[contested_sector]["heat"] + 8)
        sector_control[contested_sector]["controller"] = faction["name"]
        _mark_sector_activity(contested_sector, "sabotage", faction["name"])
        for nbr in _neighbors(contested_sector):
            if nbr not in sector_control:
                continue
            sector_control[nbr]["heat"] = min(100, sector_control[nbr]["heat"] + 4)
            sector_control[nbr]["stability"] = max(0, sector_control[nbr]["stability"] - 2)
            _mark_sector_activity(nbr, "sabotage_spill", faction["name"])
    elif action == "fortify":
        effects["stability_delta"] += random.randint(2, 4)
        effects["threat_delta"] -= 1
        effects["pressure_delta"] -= 2
        sector = target if target in sector_control else random.choice(faction["preferred_sectors"])
        sector_control[sector]["controller"] = faction["name"]
        sector_control[sector]["stability"] = min(100, sector_control[sector]["stability"] + 5)
        sector_control[sector]["interest"][faction["name"]] += 4
        _mark_sector_activity(sector, "fortify", faction["name"])
        for nbr in _neighbors(sector):
            if nbr not in sector_control:
                continue
            if sector_control[nbr]["controller"] == faction["name"]:
                sector_control[nbr]["stability"] = min(100, sector_control[nbr]["stability"] + 2)
                _mark_sector_activity(nbr, "fortify_spill", faction["name"])
    elif action == "rewire":
        parts = target.split("|", 2)
        if len(parts) == 3 and parts[0] in ("cut", "bridge"):
            mode, a, b = parts[0], parts[1], parts[2]
            if a in SECTORS and b in SECTORS:
                a, b = sorted((a, b))
                g = st.session_state.setdefault("sector_graph", _normalize_sector_graph(None))
                if mode == "cut" and b in g.get(a, []):
                    g[a].remove(b)
                    g[b].remove(a)
                    effects["threat_delta"] += 2
                    effects["leverage_delta"] -= 2
                    effects["stability_delta"] -= 1
                    sector_control[a]["heat"] = min(100, sector_control[a]["heat"] + 4)
                    sector_control[b]["heat"] = min(100, sector_control[b]["heat"] + 4)
                    _mark_sector_activity(a, "rewire_cut", faction["name"])
                    _mark_sector_activity(b, "rewire_cut", faction["name"])
                elif mode == "bridge" and b not in g.get(a, []):
                    g[a].append(b)
                    g[b].append(a)
                    effects["threat_delta"] += 1
                    effects["leverage_delta"] -= 3
                    effects["intel_delta"] += 2
                    sector_control[a]["interest"][faction["name"]] += 2
                    sector_control[b]["interest"][faction["name"]] += 2
                    sector_control[a]["heat"] = min(100, sector_control[a]["heat"] + 2)
                    sector_control[b]["heat"] = min(100, sector_control[b]["heat"] + 2)
                    _mark_sector_activity(a, "rewire_bridge", faction["name"])
                    _mark_sector_activity(b, "rewire_bridge", faction["name"])
    else:
        effects["supplies_delta"] += random.randint(1, 3)
        effects["pressure_delta"] -= 1

    world["intel"] = max(0, world["intel"] + effects["intel_delta"])
    world["supplies"] = max(0, world["supplies"] + effects["supplies_delta"])
    world["threat_clock"] = max(0, min(100, world["threat_clock"] + effects["threat_delta"]))
    world["sector_stability"] = max(0, min(100, world["sector_stability"] + effects["stability_delta"]))

    faction["pressure"] = max(0, min(100, faction["pressure"] + effects["pressure_delta"]))
    faction["leverage"] = max(0, min(100, faction["leverage"] + effects["leverage_delta"]))
    if faction["pressure"] >= 70:
        faction["stance"] = "Aggressive"
    elif faction["pressure"] >= 45:
        faction["stance"] = "Wary"
    elif faction["pressure"] <= 20:
        faction["stance"] = "Aligned"
    else:
        faction["stance"] = "Transactional"

    return effects


def _log_activity(activity: dict[str, Any]) -> None:
    st.session_state["activities"] = (st.session_state["activities"] + [activity])[-60:]


def _push_event(event: dict[str, Any]) -> None:
    st.session_state["events"] = (st.session_state["events"] + [event])[-20:]
    sector = event.get("sector")
    if sector:
        st.session_state.get("sector_activity", {})[sector] = {
            "cycle": int(event.get("cycle", st.session_state.get("world_state", {}).get("cycle", 0))),
            "kind": "event",
            "detail": event.get("name", ""),
        }


def _queue_counter_mission(event: dict[str, Any]) -> None:
    counter = {
        "event_name": event["name"],
        "sponsor": event["target_faction"],
        "rival": event["source_faction"],
        "sector": event["sector"],
        "objective": event["counter_objective"],
        "difficulty": event["difficulty"],
        "threat": event["threat"],
        "anomaly": event["anomaly"],
        "summary": event["summary"],
        "directive": event["counter_directive"],
        "stability_cost": event["stability_cost"],
        "reward_intel": event["reward_intel"],
    }
    st.session_state["counter_missions"] = (st.session_state["counter_missions"] + [counter])[-8:]


def _maybe_spawn_event(faction: dict[str, Any], action: str, target: str) -> None:
    if action not in {"sabotage", "extract", "fortify"}:
        return

    world = st.session_state["world_state"]
    cycle = int(world.get("cycle", 0))
    threat = int(world.get("threat_clock", 0))

    room = st.session_state.setdefault("_crisis_room", {"last_event_cycle": -99, "last_source": None})
    last_c = int(room.get("last_event_cycle", -99))
    # (3) Global breathing room: avoid stacking crises cycle-on-cycle.
    if cycle - last_c < 3:
        return

    base_p = 0.26
    if threat >= 78:
        base_p *= 0.52
    elif threat >= 60:
        base_p *= 0.72

    if random.random() > base_p:
        return

    if room.get("last_source") == faction["name"] and cycle - last_c < 8:
        if random.random() > 0.36:
            return

    priors = st.session_state.get("_faction_prior_aggressive", {})
    prev = priors.get(faction["name"])
    if prev == action and action in ("sabotage", "extract"):
        if random.random() > 0.44:
            return

    if target in SECTORS:
        sector = target
        target_faction = _find_faction(faction["rival"])["name"]
    elif action == "sabotage":
        rival_name = target
        anchor = _strongest_interest_sector(rival_name) or random.choice(list(st.session_state["sector_control"].keys()))
        sector = _pick_nearest_sector(faction["preferred_sectors"], anchor)
        target_faction = rival_name
    else:
        sector = random.choice(faction["preferred_sectors"])
        target_faction = target

    names = {
        "sabotage": "Blackout Cascade",
        "extract": "Relic Surge",
        "fortify": "Control Lockdown",
    }
    event = {
        "cycle": st.session_state["world_state"]["cycle"],
        "name": names[action],
        "source_faction": faction["name"],
        "target_faction": target_faction,
        "sector": sector,
        "severity": random.choice(["Elevated", "Severe", "Critical"]),
        "difficulty": random.choice(["Elevated", "Severe", "Terminal"]),
        "threat": _threat_for_faction(faction),
        "anomaly": random.choice(ANOMALIES),
        "summary": (
            f"{faction['name']} triggered {names[action]} in {sector}, forcing {target_faction} to respond "
            f"or lose position."
        ),
        "counter_objective": random.choice(
            ["Restore extraction relay", "Map anomaly corridor", "Recover unstable artifact"]
        ),
        "counter_directive": (
            f"Counter {faction['name']} influence in {sector} before the event hardens into permanent control."
        ),
        "stability_cost": random.randint(4, 8),
        "reward_intel": random.randint(6, 11),
        "player_impact": (
            f"If ignored, {target_faction} may lose position in {sector} and the next player mission is more likely "
            f"to become a counter-operation."
        ),
        "why_now": (
            f"{faction['name']} had enough pressure and leverage to force a sudden shift in {sector}."
        ),
    }
    _push_event(event)
    _queue_counter_mission(event)
    room["last_event_cycle"] = cycle
    room["last_source"] = faction["name"]
    append_log(f"New event: {event['name']} in {sector}. Counter-mission added.")


def simulate_agent_round(memory, decision_engine: OllamaDecisionEngine, rounds: int = 1) -> list[dict[str, Any]]:
    init_mas()
    snapshot_for_undo(f"agent_round_x{max(1, rounds)}")
    results: list[dict[str, Any]] = []
    for _ in range(rounds):
        if st.session_state["endgame"]["ended"]:
            break
        world = st.session_state["world_state"]
        faction_list = ordered_factions(list(st.session_state["factions"]))
        for faction in faction_list:
            if st.session_state["endgame"]["ended"]:
                break
            negotiation = _run_negotiation_phase(memory, decision_engine, faction)
            proposal, decision_status, related, obs_key, mas_policy = decide_faction_turn(
                decision_engine=decision_engine,
                memory=memory,
                faction=faction,
                world=world,
            )
            action = proposal.action
            target = proposal.target
            reasoning = proposal.reasoning
            if proposal.source in ("heuristic", "mas_llm_fallback"):
                reasoning = _agent_reasoning(faction, action, target, related)
                reasoning = f"{reasoning} Fallback reason: {decision_status}."
            effects = _apply_agent_action(faction, action, target)
            update_bandit_after_action(
                faction_name=faction["name"],
                obs_key=obs_key,
                action=action,
                effects=effects,
            )

            activity = {
                "cycle": world["cycle"],
                "faction": faction["name"],
                "action": action,
                "target": target,
                "reasoning": reasoning,
                "effects": effects,
                "confidence": round(proposal.confidence, 2),
                "decision_source": proposal.source,
                "decision_status": decision_status,
                "memory_refs": [item.id for item in related],
                "negotiation_move": negotiation["move"],
                "negotiation_partner": negotiation["partner"],
                "mas_policy": mas_policy,
                "mas_obs": obs_key,
            }
            _log_activity(activity)
            arc = str(faction.get("campaign_arc", ""))[:140]
            _record_faction_memory(
                faction["name"],
                f"Cycle {world['cycle']}: {action} @ {target}. Arc: {arc}",
            )
            _maybe_spawn_event(faction, action, target)
            pri = st.session_state.setdefault("_faction_prior_aggressive", {})
            if action in ("sabotage", "rewire", "extract", "fortify"):
                pri[faction["name"]] = action
            else:
                pri[faction["name"]] = None
            memory.save(
                text=(
                    f"Agent activity. Cycle {world['cycle']}. Faction {faction['name']} chose {action} "
                    f"against {target}. Reasoning: {reasoning}"
                ),
                payload={
                    "type": "agent_activity",
                    "cycle": world["cycle"],
                    "faction": faction["name"],
                    "action": action,
                    "target": target,
                    "decision_source": proposal.source,
                    "campaign_arc": faction.get("campaign_arc", ""),
                    "mas_policy": mas_policy,
                    "mas_obs": obs_key,
                },
            )
            results.append(activity)

        advance_turn_order(num_factions=len(st.session_state["factions"]))
        world["cycle"] += 1
        if world["supplies"] == 0:
            append_log("Supplies hit zero. The expedition hub is under severe stress.")
        if world["sector_stability"] <= 15:
            append_log("Sector stability is collapsing. Agent activity is now amplifying risk.")
        record_snapshot("agent_round")
        simulate_ai_competitors(memory, per_tick=1)
        evaluate_endgame()

    persist_state()
    if results:
        # Force globe / 2D topology to rebuild on next render (Plotly + fingerprint edge cases).
        st.session_state.pop("_topology_fig_cache", None)
        st.session_state.pop("_topology_globe_fig_cache", None)
    return results


def game_status() -> dict[str, str]:
    endgame = st.session_state["endgame"]
    if endgame["ended"]:
        return {
            "state": endgame["result"],
            "detail": endgame["summary"],
        }
    world = st.session_state["world_state"]
    if world["threat_clock"] >= 100 or world["sector_stability"] <= 0:
        return {"state": "Crisis", "detail": "The ruin has entered an unrecoverable escalation spiral."}
    if world["intel"] >= 100 and len(st.session_state["inventory"]) >= 4:
        return {"state": "Breakthrough", "detail": "The squad has enough knowledge and artifacts to dominate the ruin."}
    return {"state": "Active", "detail": "The campaign is ongoing and contested by multiple active factions."}


def autoplay_state() -> dict[str, Any]:
    return {
        "enabled": st.session_state.get("autoplay_enabled", False),
        "remaining_ticks": st.session_state.get("autoplay_remaining_ticks", 0),
        "rounds_per_tick": st.session_state.get("autoplay_rounds_per_tick", 1),
        "delay_seconds": st.session_state.get("autoplay_delay_seconds", 1.0),
    }


def configure_autoplay(enabled: bool, ticks: int, rounds_per_tick: int, delay_seconds: float) -> None:
    st.session_state["autoplay_enabled"] = enabled
    st.session_state["autoplay_remaining_ticks"] = max(0, ticks)
    st.session_state["autoplay_rounds_per_tick"] = max(1, rounds_per_tick)
    st.session_state["autoplay_delay_seconds"] = max(0.2, float(delay_seconds))


def stop_autoplay() -> None:
    st.session_state["autoplay_enabled"] = False
    st.session_state["autoplay_remaining_ticks"] = 0


def reset_campaign() -> None:
    fresh = _empty_persistent_state()
    st.session_state["world_state"] = fresh["world_state"]
    st.session_state["sector_control"] = fresh["sector_control"]
    st.session_state["sector_graph"] = fresh["sector_graph"]
    st.session_state["factions"] = fresh["factions"]
    st.session_state["inventory"] = fresh["inventory"]
    st.session_state["history"] = fresh["history"]
    st.session_state["activities"] = fresh["activities"]
    st.session_state["events"] = fresh["events"]
    st.session_state["counter_missions"] = fresh["counter_missions"]
    st.session_state["negotiations"] = fresh["negotiations"]
    st.session_state["snapshots"] = fresh["snapshots"]
    st.session_state["ai_config"] = fresh["ai_config"]
    configure_ai_agents(int(st.session_state.get("ai_config", {}).get("num_agents", 0)))
    st.session_state["endgame"] = fresh["endgame"]
    st.session_state["mas"] = json.loads(json.dumps(fresh["mas"]))
    st.session_state["autonomous_stack"] = json.loads(json.dumps(fresh["autonomous_stack"]))
    st.session_state["current_mission"] = build_mission(st.session_state["squad_name"])
    st.session_state.pop("_crisis_room", None)
    st.session_state.pop("_faction_prior_aggressive", None)
    st.session_state.pop("_topology_fig_cache", None)
    st.session_state.pop("_topology_globe_fig_cache", None)
    record_snapshot("campaign_reset")
    append_log("Campaign state reset to a fresh expedition cycle.")
    persist_state()


def append_log(message: str) -> None:
    st.session_state["log"].append(message)
