"""Generate a reproducible synthetic inverse-kinematics dataset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import StewartPlatformIK, default_platform_config


POSE_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw")
EXTENSION_COLUMNS = tuple(f"e{i}" for i in range(1, 7))
NOISY_EXTENSION_COLUMNS = tuple(f"e{i}_noisy" for i in range(1, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stewart-platform IK samples for FK training."
    )
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/v1_clean"))
    parser.add_argument("--noise-std-mm", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempt-multiplier", type=int, default=60)
    return parser.parse_args()


def sample_pose(rng: np.random.Generator, ik: StewartPlatformIK) -> np.ndarray:
    config = ik.config
    roll_deg, pitch_deg, yaw_deg = config.rotation_limit_deg
    return np.array(
        [
            rng.uniform(-config.xy_range_m, config.xy_range_m),
            rng.uniform(-config.xy_range_m, config.xy_range_m),
            rng.uniform(*config.z_range_m),
            rng.uniform(-np.deg2rad(roll_deg), np.deg2rad(roll_deg)),
            rng.uniform(-np.deg2rad(pitch_deg), np.deg2rad(pitch_deg)),
            rng.uniform(-np.deg2rad(yaw_deg), np.deg2rad(yaw_deg)),
        ],
        dtype=np.float64,
    )


def generate_rows(
    *,
    samples: int,
    noise_std_mm: float,
    seed: int,
    max_attempt_multiplier: int,
) -> tuple[list[dict[str, float]], int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if noise_std_mm < 0.0:
        raise ValueError("noise_std_mm cannot be negative")
    if max_attempt_multiplier <= 0:
        raise ValueError("max_attempt_multiplier must be positive")

    rng = np.random.default_rng(seed)
    ik = StewartPlatformIK()
    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = samples * max_attempt_multiplier

    while len(rows) < samples and attempts < max_attempts:
        attempts += 1
        pose = sample_pose(rng, ik)
        extensions = ik.solve(pose[:3], pose[3:])
        if not ik.extensions_are_valid(extensions):
            continue

        row = dict(zip(EXTENSION_COLUMNS + POSE_COLUMNS, (*extensions, *pose)))
        if noise_std_mm > 0.0:
            noise_m = rng.normal(0.0, noise_std_mm / 1000.0, size=6)
            noisy_extensions = np.clip(
                extensions + noise_m,
                0.0,
                ik.config.stroke_length_m,
            )
            row.update(dict(zip(NOISY_EXTENSION_COLUMNS, noisy_extensions)))
        rows.append(row)

    if len(rows) != samples:
        raise RuntimeError(
            f"generated only {len(rows)} valid samples after {attempts} attempts"
        )
    return rows, attempts


def write_split(path: Path, rows: list[dict[str, float]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.train_fraction <= 0.0 or args.val_fraction <= 0.0:
        raise ValueError("split fractions must be positive")
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")

    rows, attempts = generate_rows(
        samples=args.samples,
        noise_std_mm=args.noise_std_mm,
        seed=args.seed,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)

    train_end = int(args.train_fraction * len(rows))
    val_end = train_end + int(args.val_fraction * len(rows))
    splits = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = EXTENSION_COLUMNS + POSE_COLUMNS
    if args.noise_std_mm > 0.0:
        fieldnames += NOISY_EXTENSION_COLUMNS

    for split_name, split_rows in splits.items():
        write_split(args.output_dir / f"{split_name}.csv", split_rows, fieldnames)

    config = default_platform_config()
    metadata = {
        "samples": args.samples,
        "attempts": attempts,
        "seed": args.seed,
        "noise_std_mm": args.noise_std_mm,
        "splits": {name: len(split_rows) for name, split_rows in splits.items()},
        "rotation_order": "Rx(roll) @ Ry(pitch) @ Rz(yaw)",
        "stroke_length_m": config.stroke_length_m,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
