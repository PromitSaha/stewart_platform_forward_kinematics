# Stewart Platform Forward Kinematics

This project builds a forward-kinematics predictor for a six-actuator Stewart
platform. The first milestone is a tested inverse-kinematics implementation and
a reproducible synthetic dataset generator.

Given a platform pose:

```text
x, y, z, roll, pitch, yaw
```

inverse kinematics computes six actuator extensions:

```text
e1, e2, e3, e4, e5, e6
```

The generated pairs will later be used to train an SVR forward-kinematics
baseline and an RL-based residual refinement model.

## Current Scope

- Centralized platform geometry and actuator limits.
- Inverse kinematics with validated inputs.
- Configurable synthetic dataset generation.
- Unit tests for the inverse-kinematics foundation.

SVR and RL code will be added after the geometry and generated dataset are
validated. Future trained model outputs should live under
`model/svr/artifacts/` and `model/rl/artifacts/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest inverse_kinematics/tests
```

The included tests can also run without `pytest`:

```bash
python -m unittest discover -s inverse_kinematics/tests -v
```

## Generate a Dataset

```bash
python -m inverse_kinematics.generate_dataset --samples 50000 --output-dir data/v1_clean
```

Useful options:

```bash
python -m inverse_kinematics.generate_dataset \
  --samples 50000 \
  --output-dir data/v1_clean \
  --noise-std-mm 0.5 \
  --seed 42
```

The generator stores clean actuator extensions and, when noise is enabled,
noisy extensions clipped to the physical actuator range. It writes
`train.csv`, `val.csv`, `test.csv`, and `metadata.json`.

## Rotation Convention

Rotations are provided as `[roll, pitch, yaw]` in radians. This project uses:

```text
Rx(roll) @ Ry(pitch) @ Rz(yaw)
```
