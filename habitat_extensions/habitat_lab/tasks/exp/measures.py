from typing import Any, Optional

import numpy as np
from habitat import registry, Measure
from habitat.utils.visualizations import maps

from habitat_extensions.habitat_lab.tasks.nav.measures import TopDownMap


@registry.register_measure
class ExplorationVisitedLocationsReward(Measure):
    r"""The measure calculates a reward based on the number of locations visited.

    Let |Visited_t| denote the number of location visited at time t,
    then the agent maximizes |Visited_T |. The reward is rt = alpha * (|Visited_t| − |Visited_t−1|).
    See DD-PPO paper: https://arxiv.org/pdf/1911.00357.pdf.
    """

    cls_uuid: str = "exploration_vlr"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        measure_config = kwargs["config"]
        self._alpha: float = measure_config.alpha
        self._n_prev_visited_locations = 0.0
        self._topdown_map_measure: Optional[TopDownMap] = None
        self._meters_per_pixel_squared: Optional[float] = None

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

        n_curr_visited_locations = topdown_map_metric["fog_of_war_mask"].sum(dtype=float)
        n_curr_visited_locations = np.maximum(n_curr_visited_locations, self._n_prev_visited_locations)
        visited_locations_area_m = (
                n_curr_visited_locations - self._n_prev_visited_locations
        ) * self._meters_per_pixel_squared

        self._metric = self._alpha * visited_locations_area_m
        self._n_prev_visited_locations = n_curr_visited_locations


@registry.register_measure
class SceneCoverage(Measure):
    r"""Measures explored scene area.

    Returns the value between 0 and 1.
    """

    cls_uuid: str = "scene_coverage"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._topdown_map_measure: Optional[TopDownMap] = None
        self._scene_area: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [TopDownMap.cls_uuid])
        self._topdown_map_measure = task.measurements.measures[TopDownMap.cls_uuid]
        topdown_map_metric = self._topdown_map_measure.get_metric()
        self._scene_area = (topdown_map_metric["map"] == maps.MAP_VALID_POINT).sum()
        self.update_metric(episode, task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs):
        topdown_map_metric = self._topdown_map_measure.get_metric()
        self._metric = topdown_map_metric["fog_of_war_mask"].sum() / self._scene_area


@registry.register_measure
class ExplorationSuccess(Measure):
    r"""Measures whether the explored >= success_threshold of scene area (0 <= success_threshold <=1).

    This measure depends on SceneCoverage and equals to 1
    if episode has ended and SceneCoverage.get_metric() >= success_threshold.
    """

    cls_uuid: str = "exploration_success"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        measure_config = kwargs["config"]
        self._success_threshold: float = measure_config.success_threshold
        self._scene_coverage_measure: Optional[SceneCoverage] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [SceneCoverage.cls_uuid])
        self._scene_coverage_measure = task.measurements.measures[SceneCoverage.cls_uuid]
        self.update_metric(episode, task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs):
        scene_coverage_metric = self._scene_coverage_measure.get_metric()

        success = (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and scene_coverage_metric > self._success_threshold
        )

        self._metric = 1.0 if success else 0
