import os
import math
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Helpers
# -----------------------------
def to_mm(x_m: np.ndarray) -> np.ndarray:
    return x_m * 1000.0

def rad_to_deg(x_rad: np.ndarray) -> np.ndarray:
    return x_rad * (180.0 / math.pi)

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))
    return float(mae), float(rmse), max_err

def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)

def ensure_cols(df: pd.DataFrame, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SVR FK baseline (6 outputs) for Stewart platform.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset folder containing train.csv/val.csv/test.csv")
    parser.add_argument("--use_noisy", action="store_true",
                        help="Use e*_noisy columns as inputs (default: clean e1..e6)")
    parser.add_argument("--run_name", type=str, default="svr_run",
                        help="Name for saving artifacts (default: svr_run)")

    # SVR hyperparams (start reasonable; later you can tune)
    parser.add_argument("--C", type=float, default=50.0)
    parser.add_argument("--gamma", type=str, default="scale")   # "scale" or float string
    parser.add_argument("--epsilon", type=float, default=0.05)

    # speed controls
    parser.add_argument("--train_limit", type=int, default=0,
                        help="If >0, randomly subsample this many training rows (useful for faster SVR)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    repo_root = Path(__file__).resolve().parents[2]  # models/svr/train_svr.py -> repo root
    data_dir = Path(args.data_dir).expanduser().resolve()
    artifacts_dir = repo_root / "models" / "svr" / "artifacts" / args.run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    INPUTS_CLEAN = ["e1", "e2", "e3", "e4", "e5", "e6"]
    INPUTS_NOISY = ["e1_noisy", "e2_noisy", "e3_noisy", "e4_noisy", "e5_noisy", "e6_noisy"]
    OUTPUTS = ["x", "y", "z", "roll", "pitch", "yaw"]

    input_cols = INPUTS_NOISY if args.use_noisy else INPUTS_CLEAN

    # Load data
    df_train = load_split(data_dir, "train")
    df_val   = load_split(data_dir, "val")
    df_test  = load_split(data_dir, "test")

    ensure_cols(df_train, input_cols + OUTPUTS, "train")
    ensure_cols(df_val,   input_cols + OUTPUTS, "val")
    ensure_cols(df_test,  input_cols + OUTPUTS, "test")

    # Optional subsampling for SVR speed
    if args.train_limit and args.train_limit > 0 and args.train_limit < len(df_train):
        df_train = df_train.sample(n=args.train_limit, random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Subsampled train to {len(df_train)} rows for speed.")

    X_train = df_train[input_cols].to_numpy(dtype=float)
    X_val   = df_val[input_cols].to_numpy(dtype=float)
    X_test  = df_test[input_cols].to_numpy(dtype=float)

    # Parse gamma if numeric string
    gamma_value = args.gamma
    try:
        gamma_value = float(args.gamma)
    except ValueError:
        gamma_value = args.gamma  # "scale" or "auto"

    svr_params = dict(kernel="rbf", C=args.C, gamma=gamma_value, epsilon=args.epsilon)

    results = []
    print("\n=== Training SVR FK Baseline ===")
    print(f"Data dir: {data_dir}")
    print(f"Inputs:   {input_cols}")
    print(f"Outputs:  {OUTPUTS}")
    print(f"SVR:      {svr_params}")
    print(f"Artifacts:{artifacts_dir}\n")

    for out in OUTPUTS:
        model_path = artifacts_dir / f"svr_{out}.joblib"
        if model_path.exists():
            print(f"[SKIP] {out} already trained -> {model_path.name}")
            continue

        y_train = df_train[out].to_numpy(dtype=float)
        y_val   = df_val[out].to_numpy(dtype=float)
        y_test  = df_test[out].to_numpy(dtype=float)

        # Scale X and y (SVR is very sensitive to scaling)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        Xtr_s = x_scaler.fit_transform(X_train)
        ytr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        svr = SVR(**svr_params)
        print(f"[TRAIN] {out} ...")
        svr.fit(Xtr_s, ytr_s)

        # Predict on val/test
        Xval_s = x_scaler.transform(X_val)
        Xte_s  = x_scaler.transform(X_test)

        yval_pred_s = svr.predict(Xval_s)
        ytest_pred_s = svr.predict(Xte_s)

        yval_pred = y_scaler.inverse_transform(yval_pred_s.reshape(-1, 1)).ravel()
        ytest_pred = y_scaler.inverse_transform(ytest_pred_s.reshape(-1, 1)).ravel()

        # Metrics in meaningful units
        if out in ["x", "y", "z"]:
            mae_val, rmse_val, max_val = metrics(to_mm(y_val), to_mm(yval_pred))
            mae_te,  rmse_te,  max_te  = metrics(to_mm(y_test), to_mm(ytest_pred))
            unit = "mm"
        else:
            mae_val, rmse_val, max_val = metrics(rad_to_deg(y_val), rad_to_deg(yval_pred))
            mae_te,  rmse_te,  max_te  = metrics(rad_to_deg(y_test), rad_to_deg(ytest_pred))
            unit = "deg"

        # Save bundle (model + scalers + metadata)
        bundle = {
            "svr": svr,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "input_cols": input_cols,
            "output": out,
            "svr_params": svr_params,
            "train_rows": len(df_train),
            "seed": args.seed,
        }
        joblib.dump(bundle, model_path)
        print(f"[SAVE] {model_path.name}")

        results.append([out, unit, mae_val, rmse_val, max_val, mae_te, rmse_te, max_te])

    # If results are empty because everything was skipped, still show existing summary if present
    res_csv = artifacts_dir / "svr_results_summary.csv"

    if results:
        header = ["output","unit","VAL_MAE","VAL_RMSE","VAL_MAX","TEST_MAE","TEST_RMSE","TEST_MAX"]
        res_df = pd.DataFrame(results, columns=header)
        res_df.to_csv(res_csv, index=False)

        print("\n=== SVR FK Results (this run) ===")
        print("{:<8} {:<4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*header))
        for r in results:
            print("{:<8} {:<4} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(*r))

        print(f"\n[SAVED] Metrics summary -> {res_csv}")
    else:
        if res_csv.exists():
            print(f"\n[INFO] No new models trained. Existing summary found: {res_csv}")
            print(pd.read_csv(res_csv))
        else:
            print("\n[INFO] No new models trained and no summary CSV exists yet.")

    print("\nDone.")


if __name__ == "__main__":
    main()
