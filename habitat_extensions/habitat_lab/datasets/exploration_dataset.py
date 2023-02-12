import gzip
import json
import os
import uuid
from typing import List, Optional

import numpy as np
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from habitat import Dataset, read_write
from habitat.core.dataset import Episode, ALL_SCENES_MASK
from habitat.core.registry import registry
from omegaconf import DictConfig


CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


class ExplorationEpisode(Episode):
    # def __init__(
    #         self,
    #         episode_id: str,
    #         scene_id: str,
    #         start_position: List[float],
    #         start_rotation: List[float]
    # ):
    #     self.episode_id = episode_id
    #     self.scene_id = scene_id
    #     self.start_position = start_position
    #     self.start_rotation = start_rotation

    @classmethod
    def new_episode(cls, sim: HabitatSim, scene_id: str):
        if not sim.habitat_config.scene == scene_id:
            with read_write(sim.habitat_config):
                sim.habitat_config.scene = scene_id
            sim.reconfigure(sim.habitat_config)

        episode_id = str(uuid.uuid4())
        start_position = cls.sample_start_position(sim)
        start_rotation = cls.sample_start_rotation()

        return cls(
            episode_id=episode_id,
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=start_rotation
        )

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
    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
        self.episodes = []

        if config is None:
            return

        dataset_file_path = config.data_path.format(split=config.split)
        with gzip.open(dataset_file_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        dataset_dir = os.path.dirname(dataset_file_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )
            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(f.read(), scenes_dir=config.scenes_dir)
        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        for episode in deserialized["episodes"]:
            episode = ExplorationEpisode(**episode)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            self.episodes.append(episode)

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes


@registry.register_dataset(name="ExplorationDynamicDataset")
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
