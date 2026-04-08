import os
import numpy as np
import pandas as pd
from inverseKinematics import inv_kinematics

folderName = "data/v1_clean"


def rpy_deg_to_rad(rpy_deg):
    """Convert [roll,pitch,yaw] in degrees to radians."""
    return np.deg2rad(np.array(rpy_deg, dtype=float))


def sample_pose(rng, cfg):
    """
    Sample a random pose.
    Returns:
      trans: [x,y,z] in meters
      rot_rad: [roll,pitch,yaw] in radians  ✅ (everything switched to radians)
    """
    x = rng.uniform(-cfg["xy_range_m"], cfg["xy_range_m"])
    y = rng.uniform(-cfg["xy_range_m"], cfg["xy_range_m"])
    z = rng.uniform(cfg["z_min_m"], cfg["z_max_m"])  # relative to home_pos inside IK

    # Sample angles using degree bounds (human-friendly), then convert to radians
    roll_deg = rng.uniform(-cfg["roll_deg"], cfg["roll_deg"])
    pitch_deg = rng.uniform(-cfg["pitch_deg"], cfg["pitch_deg"])
    yaw_deg = rng.uniform(-cfg["yaw_deg"], cfg["yaw_deg"])

    rot_rad = rpy_deg_to_rad([roll_deg, pitch_deg, yaw_deg])

    return np.array([x, y, z], dtype=float), rot_rad


def is_valid_extension(ext, min_ext, max_ext):
    ext = np.asarray(ext, dtype=float).reshape(-1)
    return ext.shape[0] == 6 and np.all((ext >= min_ext) & (ext <= max_ext))


def _make_ik_solver():
    """
    Create IK solver. If your inv_kinematics supports verbose flag, disable it for speed.
    """
    try:
        return inv_kinematics(verbose=False)
    except TypeError:
        # older signature: inv_kinematics()
        return inv_kinematics()


def main():
    OUT_DIR = folderName
    N_VALID = 50_000

    # Actuator limits (meters)
    MIN_EXT = 0.0
    MAX_EXT = 0.202

    # Pose sampling ranges
    # NOTE: roll/pitch/yaw are specified in degrees here, but are converted to radians before IK + saving
    cfg = {
        "xy_range_m": 0.45,
        "z_min_m": 0.0,
        "z_max_m": 0.202,
        "roll_deg": 58,
        "pitch_deg": 70,
        "yaw_deg": 86,
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

    ik = _make_ik_solver()  # geometry + solve()

    rows = []
    valid = 0
    attempts = 0

    # Safety stop (prevents infinite loop if ranges are too aggressive)
    MAX_ATTEMPTS = N_VALID * 60

    while valid < N_VALID and attempts < MAX_ATTEMPTS:
        attempts += 1

        trans, rot_rad = sample_pose(rng, cfg)

        try:
            # ✅ IK receives radians now
            ext = ik.solve(trans, rot_rad)  # returns extension (m)
        except Exception:
            continue

        if not is_valid_extension(ext, MIN_EXT, MAX_EXT):
            continue

        ext = np.asarray(ext, dtype=float)

        row = {
            "e1": ext[0], "e2": ext[1], "e3": ext[2],
            "e4": ext[3], "e5": ext[4], "e6": ext[5],
            "x": trans[0], "y": trans[1], "z": trans[2],
            # ✅ store radians in CSV
            "roll": rot_rad[0], "pitch": rot_rad[1], "yaw": rot_rad[2],
        }

        if ADD_NOISE:
            noise = rng.normal(0.0, noise_std_m, size=6)
            row.update({
                "e1_noisy": ext[0] + noise[0],
                "e2_noisy": ext[1] + noise[1],
                "e3_noisy": ext[2] + noise[2],
                "e4_noisy": ext[3] + noise[3],
                "e5_noisy": ext[4] + noise[4],
                "e6_noisy": ext[5] + noise[5],
            })

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