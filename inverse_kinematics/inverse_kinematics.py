"""Inverse kinematics for a six-actuator Stewart platform."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import PlatformConfig, default_platform_config


FloatArray = NDArray[np.float64]


class StewartPlatformIK:
    """Convert a platform pose into six actuator extensions.

    Translation is relative to the configured home position and expressed in
    meters. Rotation is ``[roll, pitch, yaw]`` in radians. The multiplication
    order is ``Rx(roll) @ Ry(pitch) @ Rz(yaw)``.
    """

    def __init__(
        self,
        config: PlatformConfig | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        self.config = config or default_platform_config()
        self.verbose = verbose

    @staticmethod
    def _rotation_x(theta: float) -> FloatArray:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_y(theta: float) -> FloatArray:
        return np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_z(theta: float) -> FloatArray:
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def rotation_matrix(self, rotation_rpy_rad: ArrayLike) -> FloatArray:
        roll, pitch, yaw = self._vector(rotation_rpy_rad, "rotation_rpy_rad")
        return (
            self._rotation_x(roll)
            @ self._rotation_y(pitch)
            @ self._rotation_z(yaw)
        )

    def solve(self, translation_m: ArrayLike, rotation_rpy_rad: ArrayLike) -> FloatArray:
        translation = self._vector(translation_m, "translation_m")
        rotation = self._vector(rotation_rpy_rad, "rotation_rpy_rad")

        platform_center = translation + self.config.home_position
        rotated_platform_joints = self.rotation_matrix(rotation) @ self.config.platform_joints
        leg_vectors = (
            platform_center[:, np.newaxis]
            + rotated_platform_joints
            - self.config.base_joints
        )
        leg_lengths = np.linalg.norm(leg_vectors, axis=0)
        extensions = leg_lengths - self.config.rest_length_m

        if self.verbose:
            print("Raw leg lengths (m):", leg_lengths)
            print("Actuator extensions (m):", extensions)

        return extensions

    def extensions_are_valid(self, extensions_m: ArrayLike) -> bool:
        extensions = np.asarray(extensions_m, dtype=np.float64)
        return bool(
            extensions.shape == (6,)
            and np.isfinite(extensions).all()
            and (extensions >= 0.0).all()
            and (extensions <= self.config.stroke_length_m).all()
        )

    @staticmethod
    def _vector(values: ArrayLike, name: str) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain only finite values")
        return vector
