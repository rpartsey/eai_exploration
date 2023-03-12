from dataclasses import dataclass
from typing import Optional

from habitat.config.default_structured_configs import (
    MeasurementConfig,
    TopDownMapMeasurementConfig as TopDownMapMeasurementConfigBase
)
from hydra.core.config_store import ConfigStore


cs = ConfigStore.instance()


@dataclass
class TopDownMapV2MeasurementConfig(TopDownMapMeasurementConfigBase):
    type: str = "TopDownMap"
    meters_per_pixel: Optional[float] = None


cs.store(
    package="habitat.task.measurements.top_down_map",
    group="habitat/task/measurements",
    name="top_down_map",
    node=TopDownMapV2MeasurementConfig,
)


@dataclass
class ExplorationVisitedLocationsRewardMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationVisitedLocationsReward"
    alpha: float = 1.0


cs.store(
    package="habitat.task.measurements.exploration_vlr",
    group="habitat/task/measurements",
    name="exploration_vlr",
    node=ExplorationVisitedLocationsRewardMeasurementConfig,
)


@dataclass
class SceneCoverageMeasurementConfig(MeasurementConfig):
    type: str = "SceneCoverage"


cs.store(
    package="habitat.task.measurements.scene_coverage",
    group="habitat/task/measurements",
    name="scene_coverage",
    node=SceneCoverageMeasurementConfig,
)


@dataclass
class ExplorationSuccessMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationSuccess"
    success_threshold: float = 0.7


cs.store(
    package="habitat.task.measurements.exploration_success",
    group="habitat/task/measurements",
    name="exploration_success",
    node=ExplorationSuccessMeasurementConfig,
)
