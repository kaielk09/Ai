#!/usr/bin/env python3
"""Train the existing Purple AI V1.0 on a programming corpus."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "training" / "train_from_scratch.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue training the same Purple AI V1.0 checkpoint on programming data."
    )
    parser.add_argument("--data", default=str(ROOT / "training" / "programming_data.txt"))
    parser.add_argument("--steps", type=int, default=20000,
                        help="Total target training steps, not additional steps.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--embed", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()

    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        raise SystemExit(
            f"Dataset not found: {data}\n"
            "Run prepare_dataset.py first."
        )
    if not TRAIN_SCRIPT.is_file():
        raise SystemExit(f"Existing Purple AI trainer not found: {TRAIN_SCRIPT}")
    if args.steps < 1 or args.batch_size < 1 or args.block_size < 32:
        raise SystemExit("steps/batch-size must be positive and block-size >= 32.")
    if args.embed % args.heads:
        raise SystemExit("embed must be divisible by heads.")

    env = os.environ.copy()
    env.update({
        "PURPLE_DATA": str(data),
        "PURPLE_STEPS": str(args.steps),
        "PURPLE_BATCH_SIZE": str(args.batch_size),
        "PURPLE_BLOCK_SIZE": str(args.block_size),
        "PURPLE_LR": str(args.learning_rate),
        "PURPLE_EMBED": str(args.embed),
        "PURPLE_HEADS": str(args.heads),
        "PURPLE_LAYERS": str(args.layers),
        "PURPLE_CHECKPOINT_DIR": str(Path(args.checkpoint_dir).expanduser().resolve()),
    })

    print("PURPLE AI V1.0 — TRENING PROGRAMOWANIA")
    print("Model: ten sam PurpleGPT, bez tworzenia drugiego modelu")
    print(f"Dane: {data}")
    print(f"Checkpoint: {env['PURPLE_CHECKPOINT_DIR']}")
    print(f"Docelowa liczba kroków: {args.steps}")
    print("\nUWAGA: zrób kopię checkpointu przed treningiem.", flush=True)

    subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
