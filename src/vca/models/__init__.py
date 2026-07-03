"""Trajectory models: the ``TrajectoryModel`` interface and implementations."""

from vca.models.base import SimulationResult, TrajectoryModel
from vca.models.baseline import MarginalResamplingModel

__all__ = ["TrajectoryModel", "SimulationResult", "MarginalResamplingModel"]
