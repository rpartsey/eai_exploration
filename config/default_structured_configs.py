from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from habitat.config.default_structured_configs import (
    MeasurementConfig,
    TopDownMapMeasurementConfig as TopDownMapMeasurementConfigBase
)
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


@dataclass
class ExplorationSuccessMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationSuccess"
    success_threshold: float = 0.7


@dataclass
class TopDownMapV2MeasurementConfig(TopDownMapMeasurementConfigBase):
    type: str = "TopDownMap"
    meters_per_pixel: Optional[float] = None


@dataclass
class ExplorationVisitedLocationsRewardMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationVisitedLocationsReward"
    alpha: float = 1.0


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.top_down_map",
    group="habitat/task/measurements",
    name="top_down_map",
    node=TopDownMapV2MeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.exploration_vlr",
    group="habitat/task/measurements",
    name="exploration_vlr",
    node=ExplorationVisitedLocationsRewardMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.exploration_success",
    group="habitat/task/measurements",
    name="exploration_success",
    node=ExplorationSuccessMeasurementConfig,
)
