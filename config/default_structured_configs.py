from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from habitat.config.default_structured_configs import (
    MeasurementConfig,
    TopDownMapMeasurementConfig as TopDownMapMeasurementConfigBase
)
from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING


@dataclass
class TopDownMapMeasurementConfig2(TopDownMapMeasurementConfigBase):
    meters_per_pixel: Optional[float] = None


@dataclass
class ExplorationVisitedLocationsRewardMeasurementConfig(MeasurementConfig):
    type: str = "ExplorationVisitedLocationsReward"
    map_config: TopDownMapMeasurementConfig2 = TopDownMapMeasurementConfig2()


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.top_down_map",
    group="habitat/task/measurements",
    name="top_down_map",
    node=TopDownMapMeasurementConfig2,
)
cs.store(
    package="habitat.task.measurements.exploration_visited_locations_reward",
    group="habitat/task/measurements",
    name="exploration_visited_locations_reward",
    node=ExplorationVisitedLocationsRewardMeasurementConfig,
)
