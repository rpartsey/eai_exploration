from typing import Any, Optional

import numpy as np
from habitat import registry, Measure
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import maps
from omegaconf import DictConfig

from habitat_extensions.habitat_lab.tasks.nav.measures import TopDownMap


@registry.register_measure
class ExplorationVisitedLocationsReward(Measure):
    r"""The measure calculates a reward based on the number of locations visited.

    Let |Visited_t| denote the number of location visited at time t,
    then the agent maximizes |Visited_T |. The reward is rt = 0.25(|Visited_t| − |Visited_t−1|).
    See DD-PPO paper: https://arxiv.org/pdf/1911.00357.pdf.
    """

    cls_uuid: str = "exploration_vlr"

    def __init__(
        self, sim: HabitatSim, config: DictConfig, *args: Any, **kwargs: Any
    ):
        self._sim: HabitatSim = sim
        self._alpha: float = config.alpha
        self._n_prev_visited_locations = 0.0
        self._topdown_map_measure: Optional[TopDownMap] = None
        self._meters_per_pixel_squared: Optional[float] = None
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [TopDownMap.cls_uuid])
        self._topdown_map_measure = task.measurements.measures[TopDownMap.cls_uuid]
        self._meters_per_pixel_squared = (
            self._topdown_map_measure.meters_per_pixel * self._topdown_map_measure.meters_per_pixel
        )
        self._n_prev_visited_locations = 0
        self._metric = 0

    def update_metric(self, episode, task, *args: Any, **kwargs):
        topdown_map_metric = self._topdown_map_measure.get_metric()

        n_curr_visited_locations = topdown_map_metric["fog_of_war_mask"].sum(dtype=np.float)
        n_curr_visited_locations = np.maximum(n_curr_visited_locations, self._n_prev_visited_locations)
        visited_locations_area = (
                n_curr_visited_locations - self._n_prev_visited_locations
        ) * self._meters_per_pixel_squared

        self._metric = self._alpha * visited_locations_area
        # print(f"{self._metric}, {n_curr_visited_locations}, {self._n_prev_visited_locations}")

        self._n_prev_visited_locations = n_curr_visited_locations


@registry.register_measure
class ExplorationSuccess(Measure):
    r"""Whether the agent succeeded at its task.

    This measure depends on TopDownMap measure.
    """

    cls_uuid: str = "exploration_success"

    def __init__(
        self, sim: HabitatSim, config: DictConfig, *args: Any, **kwargs: Any
    ):
        self._sim: HabitatSim = sim
        self._success_threshold: float = config.success_threshold
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [TopDownMap.cls_uuid]
        )
        self.update_metric(episode, task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs):
        topdown_map_measure = task.measurements.measures[
            TopDownMap.cls_uuid
        ].get_metric()
        n_curr_visited_locations = topdown_map_measure["fog_of_war_mask"].sum()
        n_free_locations = (topdown_map_measure["map"] == maps.MAP_VALID_POINT).sum()

        success = (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and n_curr_visited_locations / n_free_locations > self._success_threshold
        )

        self._metric = 1.0 if success else 0
