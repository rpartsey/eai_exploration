from typing import Any, Optional, Dict
from omegaconf import DictConfig

from habitat import EmbodiedTask, Measure
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.registry import registry
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Simulator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
# from habitat.tasks.nav.nav import TopDownMap
from habitat_extensions.habitat_lab.tasks.nav.nav import TopDownMap


@registry.register_measure
class ExplorationVisitedLocationsReward(Measure):
    """
    The measure calculates a reward based on the number of locations visited.
    Let |Visited_t| denote the number of location visited at time t,
    then the agent maximizes |Visited_T |. The reward is rt = 0.25(|Visited_t| − |Visited_t−1|).
    See DD-PPO paper: https://arxiv.org/pdf/1911.00357.pdf.
    """

    cls_uuid: str = "exploration_vlr"

    def __init__(
        self, sim: HabitatSim, config: DictConfig, *args: Any, **kwargs: Any
    ):
        self._sim: HabitatSim = sim
        self._config: DictConfig = config
        self._topdown_map_measure: TopDownMap = TopDownMap(sim, config.map_config, *args, **kwargs)
        self._n_prev_visited_locations = 0
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._topdown_map_measure.reset_metric(episode)
        self._update_metric()

    def update_metric(self, *args: Any, **kwargs):
        self._topdown_map_measure.update_metric(*args, **kwargs)
        self._update_metric()

    def _update_metric(self):
        curr_state = self._topdown_map_measure.get_metric()
        n_curr_visited_locations = curr_state["fog_of_war_mask"].sum()
        self._metric = {
            "reward": n_curr_visited_locations - self._n_prev_visited_locations,
            "top_down_map": self._topdown_map_measure.get_metric()
        }
        self._n_prev_visited_locations = n_curr_visited_locations


@registry.register_task(name="Exp-v0")
class ExplorationTask(EmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [
                    float(k) for k in episode.start_rotation
                ]
                agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
