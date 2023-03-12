from typing import Any, Optional
from omegaconf import DictConfig

from habitat import EmbodiedTask
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.registry import registry
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Simulator


@registry.register_task(name="Exp-v0")
class ExplorationTask(EmbodiedTask):
    def __init__(
        self,
        config: DictConfig,
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
