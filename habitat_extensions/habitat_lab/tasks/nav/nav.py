from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from habitat.core.registry import registry
from habitat.tasks.nav.nav import TopDownMap as TopDownMapBase
from habitat.utils.visualizations import fog_of_war, maps


@registry.register_measure
class TopDownMap(TopDownMapBase):
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
