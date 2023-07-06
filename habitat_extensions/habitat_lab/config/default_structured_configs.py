from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING
from habitat.config.default_structured_configs import PointGoalSensorConfig


@dataclass
class StepCountSensorConfig(PointGoalSensorConfig):
    """
    Indicates the step counter at the current step.
    """

    type: str = "StepCountSensor"

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.step_count_sensor",
    group="habitat/task/lab_sensors",
    name="step_count_sensor",
    node=StepCountSensorConfig,
)