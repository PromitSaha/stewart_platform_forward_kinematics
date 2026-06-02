"""Stewart-platform inverse kinematics."""

from .config import PlatformConfig, default_platform_config
from .inverse_kinematics import StewartPlatformIK

__all__ = ["PlatformConfig", "StewartPlatformIK", "default_platform_config"]
