import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SVRResidualEnv(gym.Env):
    """
    Multi-step residual refinement environment for Stewart Platform FK.

    State:
        [ normalized actuator inputs (6),
          normalized current pose estimate (6),
          normalized residual error (true - estimate) (6) ]
        Total = 18

    Action:
        Small correction to current pose estimate (6)

    Transition:
        y_est_{t+1} = y_est_t + delta_y_t

    Reward:
        r_{t+1} = -(alpha_e * e_pos + beta_e * e_ori)

    where:
        e_pos = || p_est - p_true ||_2
        e_ori = || w_est - w_true ||_2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X,
        Y,
        svr_models,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        action_scale=None,
        alpha_e=1.0,
        beta_e=1.0,
        max_steps=10,
        pos_tol=0.005,
        ori_tol=0.01,
        training=True,
    ):
        super().__init__()

        self.X = np.asarray(X, dtype=np.float32)
        self.Y = np.asarray(Y, dtype=np.float32)
        self.svr_models = svr_models
        self.training = training

        if self.X.ndim != 2 or self.X.shape[1] != 6:
            raise ValueError("X must have shape (N, 6)")
        if self.Y.ndim != 2 or self.Y.shape[1] != 6:
            raise ValueError("Y must have shape (N, 6)")
        if not isinstance(self.svr_models, (list, tuple)) or len(self.svr_models) != 6:
            raise ValueError("svr_models must be a list/tuple of 6 trained SVR models")

        self.x_mean = np.zeros(6, dtype=np.float32) if x_mean is None else np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.ones(6, dtype=np.float32) if x_std is None else np.asarray(x_std, dtype=np.float32)
        self.y_mean = np.zeros(6, dtype=np.float32) if y_mean is None else np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.ones(6, dtype=np.float32) if y_std is None else np.asarray(y_std, dtype=np.float32)

        self.x_std = np.where(self.x_std < 1e-8, 1.0, self.x_std)
        self.y_std = np.where(self.y_std < 1e-8, 1.0, self.y_std)

        if action_scale is None:
            # normalized correction per step
            self.action_scale = np.array([0.002, 0.002, 0.002, 0.004, 0.004, 0.004], dtype=np.float32)
        else:
            self.action_scale = np.asarray(action_scale, dtype=np.float32)
            if self.action_scale.shape != (6,):
                raise ValueError("action_scale must have shape (6,)")

        self.alpha_e = float(alpha_e)
        self.beta_e = float(beta_e)
        self.max_steps = int(max_steps)
        self.pos_tol = float(pos_tol)
        self.ori_tol = float(ori_tol)

        # Observation = [x_norm(6), y_est_norm(6), residual_norm(6)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        # Action = normalized residual correction
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.idx = None
        self.current_step = 0
        self.current_x = None
        self.current_y_true = None
        self.current_y_svr = None
        self.current_y_est = None

    def _normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def _normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def _denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def _predict_svr(self, x):
        """
        Predict 6 outputs using 6 saved SVR bundles.
        Each bundle is a dict with keys like:
            'svr', 'x_scaler', 'y_scaler', ...
        """
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        preds = []

        for bundle in self.svr_models:
            if isinstance(bundle, dict):
                svr = bundle["svr"]
                x_scaler = bundle.get("x_scaler", None)
                y_scaler = bundle.get("y_scaler", None)

                x_in = x
                if x_scaler is not None:
                    x_in = x_scaler.transform(x_in)

                y_pred = svr.predict(x_in).reshape(-1, 1)

                if y_scaler is not None:
                    y_pred = y_scaler.inverse_transform(y_pred)

                preds.append(float(y_pred.ravel()[0]))
            else:
                # fallback if a raw model is stored directly
                y_pred = bundle.predict(x)[0]
                preds.append(float(y_pred))

        return np.asarray(preds, dtype=np.float32)

    def _pose_errors(self, y_est, y_true):
        pos_error = np.linalg.norm(y_est[:3] - y_true[:3])
        ori_error = np.linalg.norm(y_est[3:] - y_true[3:])
        return float(pos_error), float(ori_error)

    def _make_obs(self):
        x_norm = self._normalize_x(self.current_x)
        y_est_norm = self._normalize_y(self.current_y_est)
        residual_norm = self._normalize_y(self.current_y_true - self.current_y_est)
        obs = np.concatenate([x_norm, y_est_norm, residual_norm]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.training:
            self.idx = int(self.np_random.integers(0, len(self.X)))
        else:
            if options is not None and "index" in options:
                self.idx = int(options["index"])
            else:
                self.idx = 0

        self.current_step = 0
        self.current_x = self.X[self.idx].copy()
        self.current_y_true = self.Y[self.idx].copy()
        self.current_y_svr = self._predict_svr(self.current_x)
        self.current_y_est = self.current_y_svr.copy()

        obs = self._make_obs()

        info = {
            "index": self.idx,
            "step": self.current_step,
            "y_true": self.current_y_true.copy(),
            "y_svr": self.current_y_svr.copy(),
            "y_est": self.current_y_est.copy(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # error BEFORE applying action
        prev_pos_error, prev_ori_error = self._pose_errors(self.current_y_est, self.current_y_true)
        prev_total_error = self.alpha_e * prev_pos_error + self.beta_e * prev_ori_error

        # apply correction in normalized pose space
        y_est_norm = self._normalize_y(self.current_y_est)
        delta_y_norm = action * self.action_scale
        new_y_est_norm = y_est_norm + delta_y_norm
        self.current_y_est = self._denormalize_y(new_y_est_norm)

        self.current_step += 1

        # error AFTER applying action
        pos_error, ori_error = self._pose_errors(self.current_y_est, self.current_y_true)
        new_total_error = self.alpha_e * pos_error + self.beta_e * ori_error

        # action penalty
        lambda_action = 0.05
        action_penalty = lambda_action * np.linalg.norm(delta_y_norm)

        # reward = improvement - penalty
        reward = (prev_total_error - new_total_error) - action_penalty

        terminated = (pos_error < self.pos_tol and ori_error < self.ori_tol)
        truncated = self.current_step >= self.max_steps

        obs = self._make_obs()

        info = {
            "index": self.idx,
            "step": self.current_step,
            "y_true": self.current_y_true.copy(),
            "y_svr": self.current_y_svr.copy(),
            "y_est": self.current_y_est.copy(),
            "pos_error": pos_error,
            "ori_error": ori_error,
            "prev_total_error": prev_total_error,
            "new_total_error": new_total_error,
            "action_penalty": action_penalty,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info