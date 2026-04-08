import os
import sys
import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.svr_rl_env import SVRResidualEnv


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    input_cols = ["e1", "e2", "e3", "e4", "e5", "e6"]
    output_cols = ["x", "y", "z", "roll", "pitch", "yaw"]

    X = df[input_cols].values.astype(np.float32)
    Y = df[output_cols].values.astype(np.float32)
    return X, Y


def load_svr_models():
    svr_path = os.path.join(PROJECT_ROOT, "models", "svr", "artifacts", "svr_run")

    return [
        joblib.load(os.path.join(svr_path, "svr_x.joblib")),
        joblib.load(os.path.join(svr_path, "svr_y.joblib")),
        joblib.load(os.path.join(svr_path, "svr_z.joblib")),
        joblib.load(os.path.join(svr_path, "svr_roll.joblib")),
        joblib.load(os.path.join(svr_path, "svr_pitch.joblib")),
        joblib.load(os.path.join(svr_path, "svr_yaw.joblib")),
    ]


def mae(a, b):
    return np.mean(np.abs(a - b), axis=0)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def max_err(a, b):
    return np.max(np.abs(a - b), axis=0)


def main():
    test_csv = os.path.join(PROJECT_ROOT, "data", "v1_clean", "test.csv")

    X_test, Y_test = load_data(test_csv)
    svr_models = load_svr_models()

    model_path = os.path.join(PROJECT_ROOT, "models", "rl", "ppo_svr_residual")
    rl_model = PPO.load(model_path)

    # normalization stats (use train set ideally)
    X_train, Y_train = load_data(os.path.join(PROJECT_ROOT, "data", "v1_clean", "train.csv"))
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)

    env = SVRResidualEnv(
        X=X_test,
        Y=Y_test,
        svr_models=svr_models,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        training=False,
    )

    y_true_all = []
    y_svr_all = []
    y_rl_all = []

    for i in range(len(X_test)):
        obs, info = env.reset(options={"index": i})

        done = False
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        y_true_all.append(info["y_true"])
        y_svr_all.append(info["y_svr"])
        y_rl_all.append(info["y_est"])

    y_true_all = np.array(y_true_all)
    y_svr_all = np.array(y_svr_all)
    y_rl_all = np.array(y_rl_all)

    from visualize.compare_svr_vs_rl import plot_all
    plot_all(y_true_all, y_svr_all, y_rl_all)

    labels = ["x", "y", "z", "roll", "pitch", "yaw"]

    print("\n=== SVR ONLY ===")
    print("MAE :", dict(zip(labels, mae(y_true_all, y_svr_all))))
    print("RMSE:", dict(zip(labels, rmse(y_true_all, y_svr_all))))
    print("MAX :", dict(zip(labels, max_err(y_true_all, y_svr_all))))

    print("\n=== SVR + RL ===")
    print("MAE :", dict(zip(labels, mae(y_true_all, y_rl_all))))
    print("RMSE:", dict(zip(labels, rmse(y_true_all, y_rl_all))))
    print("MAX :", dict(zip(labels, max_err(y_true_all, y_rl_all))))


if __name__ == "__main__":
    main()