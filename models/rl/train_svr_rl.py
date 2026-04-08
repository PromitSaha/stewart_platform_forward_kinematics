import os
import sys
import joblib
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Allow imports from project root when running:
# python models/rl/train_svr_rl.py
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


def main():
    train_csv = os.path.join(PROJECT_ROOT, "data", "v1_clean", "train.csv")

    X_train, Y_train = load_data(train_csv)
    svr_models = load_svr_models()

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)

    def make_env():
        return SVRResidualEnv(
            X=X_train,
            Y=Y_train,
            svr_models=svr_models,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            action_scale=np.array([0.002, 0.002, 0.002, 0.004, 0.004, 0.004], dtype=np.float32),
            alpha_e=1.0,
            beta_e=1.0,
            max_steps=10,
            pos_tol=0.005,
            ori_tol=0.01,
            training=True,
        )

    env = make_vec_env(make_env, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=os.path.join(PROJECT_ROOT, "models", "rl", "tb_logs"),
    )

    model.learn(total_timesteps=100_000)

    save_path = os.path.join(PROJECT_ROOT, "models", "rl", "ppo_svr_residual")
    model.save(save_path)

    print(f"\nSaved trained model to: {save_path}.zip")


if __name__ == "__main__":
    main()