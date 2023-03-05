import os
from typing import TYPE_CHECKING, Union, cast, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy import int64

import habitat
import habitat.gym
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationEpisode, TopDownMap
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image, overlay_frame,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat_sim.utils import viz_utils as vut

# registrations
import config.default_structured_configs  # noqa structured configs
import habitat_extensions.habitat_lab.tasks.exp.exp # noqa ExplorationVisitedLocationsReward
import habitat_extensions.habitat_lab.tasks.nav.nav # noqa TopDownMap
import habitat_extensions.habitat_lab.datasets.exploration_dataset # noqa register Exploration datasets

from config import get_config

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim



# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

output_path = "output/examples/spf_topdown_map"
if not os.path.exists(output_path):
    os.makedirs(output_path)


class RandomAgent(habitat.Agent):
    def __init__(self):
        self._possible_actions = [
            HabitatSimActions.move_forward,
            HabitatSimActions.turn_left,
            HabitatSimActions.turn_right,
            HabitatSimActions.stop
        ]

    def reset(self) -> None:
        pass

    def act(self, observations: "Observations") -> Dict[str, int64]:
        action = np.random.choice(self._possible_actions)
        return action


class RandomNoStopAgent(RandomAgent):
    def __init__(self):
        super().__init__()
        self._possible_actions = [
            HabitatSimActions.move_forward,
            HabitatSimActions.turn_left,
            HabitatSimActions.turn_right,
        ]


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


def example_get_topdown_map():
    # Create habitat config
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Load the first episode
        env.reset()
        # Generate topdown map
        top_down_map = maps.get_topdown_map_from_sim(
            cast("HabitatSim", env.sim), map_resolution=1024
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        # By default, `get_topdown_map_from_sim` returns image
        # containing 0 if occupied, 1 if unoccupied, and 2 if border
        # The line below recolors returned image so that
        # occupied regions are colored in [255, 255, 255],
        # unoccupied in [128, 128, 128] and border is [0, 0, 0]
        top_down_map = recolor_map[top_down_map]
        plt.imshow(top_down_map)
        plt.title("top_down_map.png")
        plt.show()


def example_top_down_map_measure():
    # Create habitat config
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        # Create video of agent navigating in the first episode
        num_episodes = 1
        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            # Create video from images and save to disk
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()
            # Display video
            vut.display_video(f"{output_path}/{video_name}.mp4")


def example_exploration_vlr():
    # Create habitat config
    config = habitat.get_config(
        config_path="config/benchmark/exploration_hm3d_10pct_depth.yaml"
    )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(env=env, goal_radius=0.2)
        # Create video of agent navigating in the first episode
        num_episodes = 1
        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            info = flatten_dict(info)
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, extract_scalars_from_info(info))
            # Add fame to vis_frames
            vis_frames = [frame]

            step = 1
            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                frame = overlay_frame(frame, extract_scalars_from_info(info))
                vis_frames.append(frame)

                if step > 70:
                    fog_of_war_mask = info["top_down_map.fog_of_war_mask"]
                    plt.imshow(fog_of_war_mask)
                    plt.show()

                step += 1

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            # Create video from images and save to disk
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()
            # Display video
            vut.display_video(f"{output_path}/{video_name}.mp4")


def example_exploration_vlr_2():
    config = get_config(
        # config_path="config/exploration_hm3d_10pct_depth.yaml"
        config_path="config/benchmark/exploration_hm3d_10pct_depth_1scene_1episode.yaml"
    )
    with habitat.gym.make_gym_from_config(config) as env:
        agent = RandomNoStopAgent()
        num_episodes = 1
        for _ in range(num_episodes):
            observations, reward, done, info = env.reset(), 0, False, {}
            agent.reset()

            # Get metrics
            # info = env.get_info()
            # info = flatten_dict(info)
            # frame = observations_to_image(observations, info)

            # frame = overlay_frame(frame, extract_scalars_from_info(info))
            # vis_frames = [frame]
            vis_frames = []

            step = 1
            while not done:
                action = agent.act(observations)
                if action is None:
                    break

                observations, reward, done, info = env.step(action)
                frame = observations_to_image(observations, info)

                frame = overlay_frame(frame, extract_scalars_from_info(info))
                vis_frames.append(frame)

                # if step > 70:
                #     fog_of_war_mask = info["top_down_map.fog_of_war_mask"]
                #     plt.imshow(fog_of_war_mask)
                #     plt.show()

                step += 1

            current_episode = env.current_episode()
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()
            vut.display_video(f"{output_path}/{video_name}.mp4")


if __name__ == "__main__":
    # example_get_topdown_map()
    # example_top_down_map_measure()
    example_exploration_vlr_2()


# PPOTrainer
# GymHabitatEnv
