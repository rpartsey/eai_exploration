from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.registry import registry
from habitat.tasks.nav.nav import TopDownMap as TopDownMapBase
from habitat.utils.visualizations import fog_of_war, maps
from habitat.core.simulator import (
    AgentState,
)


@registry.register_measure
class TopDownMap(TopDownMapBase):
    cls_uuid = "top_down_map"

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config, *args, **kwargs)
        self._meters_per_pixel = config.meters_per_pixel or maps.calculate_meters_per_pixel(
            self._map_resolution, sim=self._sim
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
            meters_per_pixel=self._meters_per_pixel
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def update_fog_of_war_mask(self, agent_position, angle):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                angle,
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / self._meters_per_pixel,
            )

    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        # if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
        #     color = 10 + min(
        #         self._step_count * 245 // self._config.max_episode_steps, 245
        #     )
        #
        #     thickness = self.line_thickness
        #     if self._previous_xy_location[agent_index] is not None:
        #         cv2.line(
        #             self._top_down_map,
        #             self._previous_xy_location[agent_index],
        #             (a_y, a_x),
        #             color,
        #             thickness=thickness,
        #         )
        angle = TopDownMap.get_polar_angle(agent_state)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y
