from typing import Dict, List, Any, Tuple

import time
import numpy as np
import pygame as pg
import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from habitat_sim.logging import logger


class InteractivePyGameViewer:
    def __init__(
            self,
            sim_settings: Dict[str, Any],
            window_size: Tuple[int, int],
            target_fps=60.0
    ) -> None:
        self.sim_settings: Dict[str, Any] = sim_settings
        self.agent_id: int = self.sim_settings["default_agent"]
        self.sensor_name: str = "color_sensor"
        self.sim = None

        self.key_to_action = {
            pg.K_UP: "look_up",
            pg.K_DOWN: "look_down",
            pg.K_LEFT: "turn_left",
            pg.K_RIGHT: "turn_right",
            pg.K_a: "move_left",
            pg.K_d: "move_right",
            pg.K_s: "move_backward",
            pg.K_w: "move_forward",
            pg.K_x: "move_down",
            pg.K_z: "move_up",
        }

        pg.init()
        self.target_fps = target_fps
        window_height, window_width = window_size
        self.screen = pg.display.set_mode((window_width, window_height))
        self.reconfigure_sim()
        self.print_help_text()

    def move_and_look(self, pressed_keys) -> None:
        agent = self.sim.agents[self.agent_id]
        action_queue: List[str] = [
            act for key, act in self.key_to_action.items()
            if pressed_keys[key]
        ]
        [agent.act(x) for x in action_queue]

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        move_amount, look_amount = 0.07, 2

        action_list = self.key_to_action.values()
        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        for action in action_list:
            actuation_spec_amt = move_amount if "move" in action else look_amount
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def get_obs_image(self):
        vis_sensor = self.sim._Simulator__sensors[self.agent_id][self.sensor_name]
        vis_sensor.draw_observation()
        sensor_obs = vis_sensor.get_observation()
        obs_image = pg.surfarray.make_surface(np.rot90(np.fliplr(sensor_obs[:, :, :3])))

        return obs_image

    def reconfigure_sim(self) -> None:
        self.cfg = make_cfg(self.sim_settings)
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.sim_settings["stage_requires_lighting"]:
            logger.info("Setting synthetic lighting override for stage.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        if self.sim is None:
            self.sim = habitat_sim.Simulator(self.cfg)
        else:  # edge case
            if self.sim.config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id:
                # we need to force a reset, so change the internal config scene name
                self.sim.config.sim_cfg.scene_id = "NONE"
            self.sim.reconfigure(self.cfg)

        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

    def exec(self):
        prev_time = time.time()
        quit_viewer = False
        while not quit_viewer:
            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT:
                    quit_viewer = True
                    break
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_ESCAPE:
                        quit_viewer = True
                        break
                    elif e.key == pg.K_h:
                        self.print_help_text()

            pressed_keys = pg.key.get_pressed()
            self.move_and_look(pressed_keys)

            obs_image = self.get_obs_image()
            self.screen.blit(obs_image, (0, 0))
            pg.display.update()

            # slow down the viewer to self.target_fps
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(1.0 / self.target_fps - diff, 0)
            time.sleep(delay)
            prev_time = curr_time

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info("""
        =====================================================
        Key Commands:
        -------------
            esc:        Exit the application.
            'h':        Display this help message.

        Agent Controls:
            'wasd':     Move the agent's body forward/backward and left/right.
            'zx':       Move the agent's body up/down.
            arrow keys: Turn the agent's body left/right and camera look up/down.

            Utilities:
            'r':        Reset the simulator with the most recently loaded scene.
        =====================================================
        """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="./data/test_assets/scenes/simple_room.glb",
        type=str,
        help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',
    )
    parser.add_argument(
        "--dataset",
        default="./data/objects/ycb/ycb.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help='dataset configuration file to use (default: "./data/objects/ycb/ycb.scene_dataset_config.json")',
    )
    parser.add_argument(
        "--window_height",
        default=480,
        type=int,
        help="Simulation window height.",
    )
    parser.add_argument(
        "--window_width",
        default=640,
        type=int,
        help="Simulation window width.",
    )
    parser.add_argument(
        "--disable_physics",
        action="store_true",
        help="disable physics simulation (default: False)",
    )
    parser.add_argument(
        "--stage_requires_lighting",
        action="store_true",
        help="Override configured lighting to use synthetic lighting for the stage.",
    )

    args = parser.parse_args()

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["stage_requires_lighting"] = args.stage_requires_lighting

    InteractivePyGameViewer(
        sim_settings,
        window_size=(args.window_height, args.window_width)
    ).exec()
