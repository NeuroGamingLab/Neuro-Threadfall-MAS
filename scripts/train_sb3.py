#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Train a Threadfall rival policy with Stable-Baselines3 (scaffolding).")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--out", type=str, default="artifacts/sb3-policy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit(
            "Missing Stable-Baselines3. Install: pip install -r requirements-rl.txt"
        ) from exc

    from rl.threadfall_env import EnvConfig, ThreadfallRivalEnv

    env = ThreadfallRivalEnv(EnvConfig(seed=int(args.seed)))
    model = PPO("MlpPolicy", env, verbose=1, seed=int(args.seed))
    model.learn(total_timesteps=int(args.steps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved policy to {out}.zip")


if __name__ == "__main__":
    main()

