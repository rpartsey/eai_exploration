import uuid
from typing import List, Optional

import numpy as np
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from habitat import Dataset, read_write
from habitat.core.dataset import BaseEpisode
from habitat.core.registry import registry


class ExplorationEpisode(BaseEpisode):
    def __init__(
            self,
            scene_id: str,
            start_position: List[float],
            start_rotation: List[float]
    ):
        self.scene_id = scene_id
        self.episode_id = str(uuid.uuid4())
        self.start_position = start_position
        self.start_rotation = start_rotation

    @classmethod
    def new_episode(cls, sim: HabitatSim, scene_id: str):
        if not sim.habitat_config.scene == scene_id:
            with read_write(sim.habitat_config):
                sim.habitat_config.scene = scene_id
            sim.reconfigure(sim.habitat_config)

        start_position = cls.sample_start_position(sim)
        start_rotation = cls.sample_start_rotation()

        return cls(scene_id, start_position, start_rotation)

    @staticmethod
    def sample_start_position(sim: HabitatSim, island_radius_limit: float = 1.5):
        start_position = sim.sample_navigable_point()
        while sim.island_radius(start_position) < island_radius_limit:
            start_position = sim.sample_navigable_point()

        return start_position

    @staticmethod
    def sample_start_rotation():
        angle = np.random.uniform(0, 2 * np.pi)
        start_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]

        return start_rotation


@registry.register_dataset(name="ExplorationStaticDataset")
class ExplorationStaticDataset(Dataset):
    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        r"""Creates dataset from :p:`json_str`.

        :param json_str: JSON string containing episodes information.
        :param scenes_dir: directory containing graphical assets relevant
            for episodes present in :p:`json_str`.

        Directory containing relevant graphical assets of scenes is passed
        through :p:`scenes_dir`.
        """
        raise NotImplementedError


@registry.register_dataset(name="ExplorationStaticDataset")
class ExplorationDynamicDataset(Dataset):
    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        r"""Creates dataset from :p:`json_str`.

        :param json_str: JSON string containing episodes information.
        :param scenes_dir: directory containing graphical assets relevant
            for episodes present in :p:`json_str`.

        Directory containing relevant graphical assets of scenes is passed
        through :p:`scenes_dir`.
        """
        raise NotImplementedError
