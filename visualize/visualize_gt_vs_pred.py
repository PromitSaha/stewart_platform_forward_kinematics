import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Config
# =========================
DATA_PATH = Path("data/v1_clean/test.csv")
MODEL_DIR = Path("models/svr/artifacts/svr_clean_v1")

# Your outputs (must match filenames: svr_x.joblib, svr_y.joblib, ...)
OUTPUTS = [
    ("x", "mm"),
    ("y", "mm"),
    ("z", "mm"),
    ("roll", "deg"),
    ("pitch", "deg"),
    ("yaw", "deg"),
]

# How many points to plot (scatter can get dense). Set None to plot all.
MAX_POINTS = 3000

# Save figure?
SAVE_FIG = True
OUT_FIG = Path("analysis/gt_vs_pred_svr_clean_v1.png")

# =========================
# Helpers
# =========================
def predict_from_bundle(bundle: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    bundle keys (from your print):
      'svr', 'x_scaler', 'y_scaler', 'input_cols', 'output', ...
    Returns:
      y_true, y_pred in original (unscaled) units
    """
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
    # Load test data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Test CSV not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Prepare figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (out_name, unit) in zip(axes, OUTPUTS):
        model_path = MODEL_DIR / f"svr_{out_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        bundle = joblib.load(model_path)

        # Predict (correctly handles scaling)
        y_true, y_pred = predict_from_bundle(bundle, df)
        y_true, y_pred = subsample(y_true, y_pred, MAX_POINTS, seed=0)

        # Scatter: GT vs Pred
        ax.scatter(y_true, y_pred, s=8, alpha=0.5)

        # Ideal y=x line
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)

        ax.set_title(out_name)
        ax.set_xlabel(f"Ground Truth ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.grid(True)

    fig.suptitle("SVR Forward Kinematics: Ground Truth vs Predicted (Test Set)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save and/or show
    if SAVE_FIG:
        OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_FIG, dpi=300)
        print(f"[SAVED] {OUT_FIG.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
