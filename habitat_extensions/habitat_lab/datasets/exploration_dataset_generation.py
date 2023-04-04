import argparse
import gzip
import json
import os
from typing import List, Dict, Callable
from glob import iglob
import shutil


class PointNavToExplorationDatasetConverter:
    def __init__(
            self,
            pointnav_dataset_path: str,
            exploration_dataset_path: str,
            splits: List[str]
    ):
        self._pointnav_dataset_path = pointnav_dataset_path
        self._exploration_dataset_path = exploration_dataset_path
        self._splits = splits

    @staticmethod
    def _get_exploration_episode_fields(pointnav_episode: Dict):
        return {
            "episode_id": pointnav_episode["episode_id"],
            "scene_id": pointnav_episode["scene_id"],
            "start_position": pointnav_episode["start_position"],
            "start_rotation": pointnav_episode["start_rotation"],
        }

    @staticmethod
    def _pointnav_episodes_to_exploration(
            pointnav_episodes: List[Dict],
            convert_function: Callable[[Dict], Dict]
    ):
        exploration_episodes = [convert_function(episode) for episode in pointnav_episodes]

        return exploration_episodes

    @staticmethod
    def _load_episode_dataset(data_file_path: str):
        with gzip.open(data_file_path, "rt") as fp:
            data = json.load(fp)

        return data

    @staticmethod
    def _save_episode_dataset(episodes: List[Dict], save_path: str):
        assert not os.path.exists(save_path), f"Path {save_path} already exists."
        with gzip.open(save_path, "wt") as f:
            json.dump({"episodes": episodes}, f)

    @staticmethod
    def _create_dirs(dirs_path: str, exist_ok: bool = False):
        os.makedirs(dirs_path, exist_ok=exist_ok)

    def convert(self):
        for split in self._splits:
            dataset_file_path = f"{self._pointnav_dataset_path}/{split}/{split}.json.gz"
            data = self._load_episode_dataset(dataset_file_path)
            pointnav_episodes = data["episodes"]

            if len(pointnav_episodes) != 0:
                exploration_episodes = self._pointnav_episodes_to_exploration(
                    pointnav_episodes=pointnav_episodes,
                    convert_function=self._get_exploration_episode_fields
                )
                dest_data_file_path = f"{self._exploration_dataset_path}/{split}/{split}.json.gz"
                self._create_dirs(
                    dirs_path=os.path.dirname(dest_data_file_path),
                    exist_ok=True
                )
                self._save_episode_dataset(
                    episodes=exploration_episodes,
                    save_path=dest_data_file_path
                )

            else:
                self._create_dirs(
                    dirs_path=f"{self._exploration_dataset_path}/{split}",
                    exist_ok=True
                )
                shutil.copyfile(
                    f"{self._pointnav_dataset_path}/{split}/{split}.json.gz",
                    f"{self._exploration_dataset_path}/{split}/{split}.json.gz"
                )

                for scene_dataset_file_path in iglob(f"{self._pointnav_dataset_path}/{split}/content/*.json.gz"):
                    data = self._load_episode_dataset(scene_dataset_file_path)
                    pointnav_episodes = data["episodes"]

                    exploration_episodes = self._pointnav_episodes_to_exploration(
                        pointnav_episodes=pointnav_episodes,
                        convert_function=self._get_exploration_episode_fields
                    )

                    scene_dataset_file_name = os.path.basename(scene_dataset_file_path)
                    dest_data_file_path = f"{self._exploration_dataset_path}/{split}/content/{scene_dataset_file_name}"
                    self._create_dirs(
                        dirs_path=os.path.dirname(dest_data_file_path),
                        exist_ok=True
                    )
                    self._save_episode_dataset(
                        episodes=exploration_episodes,
                        save_path=dest_data_file_path
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnav_dataset_path", type=str, required=True)
    parser.add_argument("--exploration_dataset_path", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", required=True, help="List of dataset splits to covert.")

    args = parser.parse_args()

    PointNavToExplorationDatasetConverter(
        pointnav_dataset_path=args.pointnav_dataset_path,
        exploration_dataset_path=args.exploration_dataset_path,
        splits=args.splits,
    ).convert()
