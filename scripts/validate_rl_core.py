#!/usr/bin/env python3
from __future__ import annotations

"""
Sanity checks for the pure-state RL core.

This is not a unit test framework; it’s a quick validation tool you can run locally.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    from threadfall.rl_core import build_mission, initial_state, resolve_mission, reward_from_info

    for ep in range(int(args.episodes)):
        state = initial_state(seed=int(args.seed) + ep)
        total = 0.0
        for t in range(int(args.steps)):
            mission = build_mission(state, squad_name="Validator", mission_type="standard")
            state, info = resolve_mission(state, mission=mission, committed_supplies=0, advance_cycle=True)
            total += reward_from_info(info)
            if state["world"]["supplies"] <= 0:
                break
        w = state["world"]
        print(
            f"episode={ep} total_reward={total:.1f} cycle={w['cycle']} intel={w['intel']} "
            f"artifacts={state.get('inventory_artifacts', 0)} threat={w['threat_clock']} supplies={w['supplies']}"
        )


if __name__ == "__main__":
    main()

