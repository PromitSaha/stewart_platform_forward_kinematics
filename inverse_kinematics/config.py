"""Configuration for the Stewart-platform geometry and workspace."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _base_joints() -> FloatArray:
    return np.array(
        [
            [0.0440, -0.1642, -0.1642, 0.0440, 0.1202, 0.1202],
            [0.1642, 0.0440, -0.0440, -0.1642, -0.1202, 0.1202],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _platform_joints() -> FloatArray:
    return np.array(
        [
            [-0.0391, -0.0878, -0.0878, -0.0391, 0.1269, 0.1269],
            [0.1240, 0.0959, -0.0959, -0.1240, -0.0281, 0.0281],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PlatformConfig:
    """Physical geometry and limits, expressed in meters and radians."""

    base_joints: FloatArray = field(default_factory=_base_joints)
    platform_joints: FloatArray = field(default_factory=_platform_joints)
    home_position: FloatArray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.5628], dtype=np.float64)
    )
    rest_length_m: float = 0.57
    stroke_length_m: float = 0.202
    xy_range_m: float = 0.45
    z_range_m: tuple[float, float] = (0.0, 0.202)
    rotation_limit_deg: tuple[float, float, float] = (58.0, 70.0, 86.0)

    def __post_init__(self) -> None:
        if self.base_joints.shape != (3, 6):
            raise ValueError("base_joints must have shape (3, 6)")
        if self.platform_joints.shape != (3, 6):
            raise ValueError("platform_joints must have shape (3, 6)")
        if self.home_position.shape != (3,):
            raise ValueError("home_position must have shape (3,)")
        if self.rest_length_m <= 0.0:
            raise ValueError("rest_length_m must be positive")
        if self.stroke_length_m <= 0.0:
            raise ValueError("stroke_length_m must be positive")


def default_platform_config() -> PlatformConfig:
    return PlatformConfig()
