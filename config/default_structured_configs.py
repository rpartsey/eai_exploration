from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from habitat.config.default_structured_configs import (
    MeasurementConfig,
    TopDownMapMeasurementConfig as TopDownMapMeasurementConfigBase
)
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


@dataclass
class TopDownMapV2MeasurementConfig(TopDownMapMeasurementConfigBase):
    type: str = "TopDownMap"
    meters_per_pixel: Optional[float] = None


@dataclass
class ExplorationVisitedLocationsRewardMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationVisitedLocationsReward"
    map_config: TopDownMapV2MeasurementConfig = TopDownMapV2MeasurementConfig()


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.top_down_map_v2",
    group="habitat/task/measurements",
    name="top_down_map_v2",
    node=TopDownMapV2MeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.exploration_vlr",
    group="habitat/task/measurements",
    name="exploration_vlr",
    node=ExplorationVisitedLocationsRewardMeasurementConfig,
)
