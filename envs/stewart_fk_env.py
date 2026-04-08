import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StewartFKEnv(gym.Env):
    """
    Option A: RL as iterative FK solver.

    Target: actuator extensions l* (6x)
    State: current pose estimate p_hat = [x,y,z,roll,pitch,yaw]  (angles in radians)
    Action: delta pose (6x) applied to p_hat

    Observation: [e (6), de (6), p_hat_norm (6)] => 18D
      e  = IK(p_hat) - l*
      de = e - e_prev

    Reward:
      -alpha * ||e_norm||_2 + beta*(prev_rms - new_rms) - gamma*||dp||^2 + success_bonus (on success)

    You must provide an IK solver object with: solve(trans_xyz, rot_rpy_rad) -> (6,)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ik_solver,
        # workspace bounds (meters/radians): shape (6,2) for [x,y,z,roll,pitch,yaw]
        pose_bounds,
        # normalization scales
        length_scale=0.20,      # meters (typical extension scale)
        # action scaling
        action_step_max=(0.002, 0.002, 0.002, 0.003, 0.003, 0.003),  # meters/radians per step
        # termination
        eps_len_rms=0.001,      # meters RMS length error
        max_steps=60,
        # reward weights
        alpha=1.0,
        beta=0.5,
        gamma=0.05,
        success_bonus=5.0,
        # reset
        init_mode="center_noise",  # "center_noise" or "near_gt"
        init_noise=(0.01, 0.01, 0.01, 0.05, 0.05, 0.05),  # meters/radians
        seed=None,
    ):
        super().__init__()
        self.ik = ik_solver

        self.pose_bounds = np.asarray(pose_bounds, dtype=np.float32)
        if self.pose_bounds.shape != (6, 2):
            raise ValueError("pose_bounds must be shape (6,2) for [x,y,z,roll,pitch,yaw]")

        self.length_scale = float(length_scale)

        self.action_step_max = np.asarray(action_step_max, dtype=np.float32)
        if self.action_step_max.shape != (6,):
            raise ValueError("action_step_max must be length 6")

        self.eps_len_rms = float(eps_len_rms)
        self.max_steps = int(max_steps)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.success_bonus = float(success_bonus)

        self.init_mode = str(init_mode)
        self.init_noise = np.asarray(init_noise, dtype=np.float32)
        if self.init_noise.shape != (6,):
            raise ValueError("init_noise must be length 6")

        self.rng = np.random.default_rng(seed)

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Loose observation bounds (we normalize internally)
        obs_high = np.ones(18, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Internal state
        self.p_gt = None
        self.l_star = None
        self.p_hat = None
        self.e_prev = None
        self.step_count = 0

    # -------- helpers --------
    def _sample_pose_uniform(self) -> np.ndarray:
        lo = self.pose_bounds[:, 0]
        hi = self.pose_bounds[:, 1]
        return self.rng.uniform(lo, hi).astype(np.float32)

    def _clip_pose(self, pose: np.ndarray) -> np.ndarray:
        lo = self.pose_bounds[:, 0]
        hi = self.pose_bounds[:, 1]
        return np.clip(pose, lo, hi).astype(np.float32)

    def _ik_ext(self, pose: np.ndarray) -> np.ndarray:
        trans = pose[:3].astype(np.float32)
        rot = pose[3:].astype(np.float32)  # roll,pitch,yaw (rad)
        ext = self.ik.solve(trans, rot)
        ext = np.asarray(ext, dtype=np.float32).reshape(6,)
        return ext

    def _length_residual(self, pose: np.ndarray) -> np.ndarray:
        return (self._ik_ext(pose) - self.l_star).astype(np.float32)

    def _pose_norm(self, pose: np.ndarray) -> np.ndarray:
        # normalize pose to [-1,1] using bounds
        lo = self.pose_bounds[:, 0]
        hi = self.pose_bounds[:, 1]
        rng = np.where((hi - lo) == 0, 1.0, (hi - lo))
        x01 = (pose - lo) / rng          # [0,1]
        return (x01 * 2.0 - 1.0).astype(np.float32)

    def _make_obs(self, e: np.ndarray, de: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        e_n = e / self.length_scale
        de_n = de / self.length_scale
        p_n = self._pose_norm(p_hat)
        return np.concatenate([e_n, de_n, p_n], axis=0).astype(np.float32)

    # -------- gym API --------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # sample GT pose and target extensions
        self.p_gt = self._sample_pose_uniform()
        self.l_star = self._ik_ext(self.p_gt)

        # initial guess
        if self.init_mode == "near_gt":
            noise = self.rng.normal(0.0, self.init_noise).astype(np.float32)
            self.p_hat = self._clip_pose(self.p_gt + noise)
        else:
            center = (self.pose_bounds[:, 0] + self.pose_bounds[:, 1]) / 2.0
            noise = self.rng.normal(0.0, self.init_noise).astype(np.float32)
            self.p_hat = self._clip_pose(center + noise)

        e0 = self._length_residual(self.p_hat)
        self.e_prev = e0.copy()

        obs = self._make_obs(e0, np.zeros_like(e0), self.p_hat)

        info = {
            "p_gt": self.p_gt.copy(),
            "l_star": self.l_star.copy(),
            "len_rms": float(np.sqrt(np.mean(e0**2))),
        }
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(6,)
        dp = action * self.action_step_max

        p_next = self._clip_pose(self.p_hat + dp)

        e_next = self._length_residual(p_next)
        de = e_next - self.e_prev

        rms_next = float(np.sqrt(np.mean(e_next**2)))
        rms_prev = float(np.sqrt(np.mean(self.e_prev**2)))

        # normalized error norm (dense)
        e_next_n = e_next / self.length_scale
        err_norm = float(np.linalg.norm(e_next_n, ord=2))

        progress = (rms_prev - rms_next)

        reward = (
            -self.alpha * err_norm
            + self.beta * progress
            - self.gamma * float(np.sum(dp**2))
        )

        terminated = False
        if rms_next < self.eps_len_rms:
            terminated = True
            reward += self.success_bonus

        truncated = self.step_count >= self.max_steps

        self.p_hat = p_next
        self.e_prev = e_next

        obs = self._make_obs(e_next, de, self.p_hat)

        info = {
            "len_rms": rms_next,
            "progress": progress,
            "p_hat": self.p_hat.copy(),
        }
        return obs, reward, terminated, truncated, info