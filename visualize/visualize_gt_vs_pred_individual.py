import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Config
# =========================
DATA_PATH = Path("data/v1_clean/test.csv")
MODEL_DIR = Path("models/svr/artifacts/svr_run")

OUTPUTS = [
    ("x", "m"),
    ("y", "m"),
    ("z", "m"),
    ("roll", "rad"),
    ("pitch", "rad"),
    ("yaw", "rad"),
]

MAX_POINTS = 3000

# Save figures?
SAVE_FIG = True
OUT_DIR = Path("analysis/gt_vs_pred_separate")  # folder for 6 images
DPI = 300

# =========================
# Helpers
# =========================
def predict_from_bundle(bundle: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    svr = bundle["svr"]
    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    input_cols = bundle["input_cols"]
    out_col = bundle["output"]

    X = df[input_cols].values
    Xs = x_scaler.transform(X)

    y_pred_scaled = svr.predict(Xs).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()

    y_true = df[out_col].values
    return y_true, y_pred


def subsample(y_true: np.ndarray, y_pred: np.ndarray, max_points: int | None, seed: int = 0):
    if max_points is None or len(y_true) <= max_points:
        return y_true, y_pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=max_points, replace=False)
    return y_true[idx], y_pred[idx]


# =========================
# Main
# =========================
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test CSV not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if SAVE_FIG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    for out_name, unit in OUTPUTS:
        model_path = MODEL_DIR / f"svr_{out_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        bundle = joblib.load(model_path)

        y_true, y_pred = predict_from_bundle(bundle, df)
        y_true, y_pred = subsample(y_true, y_pred, MAX_POINTS, seed=0)

        # ---- Separate figure per output ----
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        ax.scatter(y_true, y_pred, s=10, alpha=0.5)

        # Ideal y=x line
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)

        
        ax.set_xlabel(f"Ground Truth ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.grid(True)

        plt.tight_layout()

        if SAVE_FIG:
            out_file = OUT_DIR / f"gt_vs_pred_{out_name}.png"
            plt.savefig(out_file, dpi=DPI)
            print(f"[SAVED] {out_file.resolve()}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
