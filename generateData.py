import os
import sys
import math
import numpy as np
import pandas as pd
from inverseKinematics import inv_kinematics

def rpy_deg_to_rad(rpy_deg):
    return np.deg2rad(np.array(rpy_deg, dtype=float))

def sample_pose(rng, cfg):
    x = rng.uniform(-cfg["xy_range_m"], cfg["xy_range_m"])
    y = rng.uniform(-cfg["xy_range_m"], cfg["xy_range_m"])
    z = rng.uniform(cfg["z_min_m"], cfg["z_max_m"])  # relative to home_pos inside IK

    roll  = math.radians(rng.uniform(-cfg["roll_deg"],  cfg["roll_deg"]))
    pitch = math.radians(rng.uniform(-cfg["pitch_deg"], cfg["pitch_deg"]))
    yaw   = math.radians(rng.uniform(-cfg["yaw_deg"],   cfg["yaw_deg"]))

    return np.array([x, y, z], dtype=float), np.array([roll, pitch, yaw], dtype=float)

def is_valid_extension(ext, min_ext, max_ext):
    ext = np.asarray(ext, dtype=float).reshape(-1)
    return ext.shape[0] == 6 and np.all((ext >= min_ext) & (ext <= max_ext))

def main():
    OUT_DIR = "stewart_fk_dataset_50k"
    N_VALID = 50_000

    # Actuator limits
    MIN_EXT = 0.0
    MAX_EXT = 0.202

    # Pose sampling ranges
    cfg = {
        "xy_range_m": 0.45,
        "z_min_m":   0.0,
        "z_max_m":    0.202,
        "roll_deg":    1.0,
        "pitch_deg":   1.2,
        "yaw_deg":    1.5,
    }

    # Optional: add measurement noise to extensions (helps robustness)
    ADD_NOISE = True
    NOISE_STD_MM = 0.5  # standard deviation in mm
    noise_std_m = NOISE_STD_MM / 1000.0

    # Splits
    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.15  # test = remainder

    # Random seed for repeatability
    SEED = 42
    rng = np.random.default_rng(SEED)

    # ----------------- INIT -----------------
    os.makedirs(OUT_DIR, exist_ok=True)

    ik = inv_kinematics()  # your geometry + solve() :contentReference[oaicite:5]{index=5}

    rows = []
    valid = 0
    attempts = 0

    # Safety stop (if your ranges are too aggressive, this prevents infinite looping)
    MAX_ATTEMPTS = N_VALID * 60

    # NOTE: Your IK prints every call. If you want it silent, edit inverseKinematics.py and comment the prints.
    while valid < N_VALID and attempts < MAX_ATTEMPTS:
        attempts += 1

        trans, rot = sample_pose(rng, cfg)

        try:
            ext = ik.solve(trans, rot)  # returns extension (m) :contentReference[oaicite:6]{index=6}
        except Exception:
            continue

        if not is_valid_extension(ext, MIN_EXT, MAX_EXT):
            continue

        ext = np.asarray(ext, dtype=float)

        row = {
            "e1": ext[0], "e2": ext[1], "e3": ext[2],
            "e4": ext[3], "e5": ext[4], "e6": ext[5],
            "x": trans[0], "y": trans[1], "z": trans[2],
            "roll": rot[0], "pitch": rot[1], "yaw": rot[2],
        }

        # if ADD_NOISE:
        #     noise = rng.normal(0.0, noise_std_m, size=6)
        #     row.update({
        #         "e1_noisy": ext[0] + noise[0],
        #         "e2_noisy": ext[1] + noise[1],
        #         "e3_noisy": ext[2] + noise[2],
        #         "e4_noisy": ext[3] + noise[3],
        #         "e5_noisy": ext[4] + noise[4],
        #         "e6_noisy": ext[5] + noise[5],
        #     })

        rows.append(row)
        valid += 1

        if valid % 5000 == 0:
            print(f"Valid samples: {valid}/{N_VALID} (attempts: {attempts})")

    if valid < N_VALID:
        print(f"WARNING: Only generated {valid} valid samples after {attempts} attempts.")
        print("Tip: Reduce rotation ranges or z range, or increase MAX_ATTEMPTS.")
    else:
        print(f"Done: generated {valid} valid samples in {attempts} attempts.")

    df = pd.DataFrame(rows)

    # Shuffle + split
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()

    # Save
    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path = os.path.join(OUT_DIR, "val.csv")
    test_path = os.path.join(OUT_DIR, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("\nSaved:")
    print(" ", train_path)
    print(" ", val_path)
    print(" ", test_path)
    print("\nColumns:", list(df.columns))

if __name__ == "__main__":
    main()
