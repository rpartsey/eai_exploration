import sys
from typing import Dict, List, Any, Tuple, Union, Sequence

import os
import argparse
import time
import numpy as np
import pygame as pg
import habitat_sim
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from od.cubercnn import util, vis
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from habitat_sim.logging import logger

from od.demo import setup
from od.demo import build_model


class InteractiveOdPyGameViewer:
    def __init__(
            self,
            sim_settings: Dict[str, Any],
            window_size: Tuple[int, int],
            od_settings: Union[Dict[str, Any], argparse.Namespace],
            vis_sensor_name="color_sensor",
            target_fps=60.0,
    ) -> None:
        self._sim_settings: Dict[str, Any] = sim_settings
        self._agent_id: int = self._sim_settings["default_agent"]
        self._vis_sensor_name: str = vis_sensor_name
        self._sim = None

        self._od_settings = od_settings
        self._od_config = None
        self._od_model = None
        self._setup_od()

        self._key_to_action = {
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
        pg.display.set_caption("Interactive Habitat Viewer")
        self._target_fps = target_fps
        window_height, window_width = window_size
        self._screen = pg.display.set_mode((window_width, window_height))
        self._reconfigure_sim()
        self._print_help_text()

    def _default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        move_amount, look_amount = 0.07, 2

        action_list = self._key_to_action.values()
        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        for action in action_list:
            actuation_spec_amt = move_amount if "move" in action else look_amount
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self._agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def _reconfigure_sim(self) -> None:
        self.cfg = make_cfg(self._sim_settings)
        self.cfg.agents[self._agent_id] = self._default_agent_config()

        if self._sim_settings["stage_requires_lighting"]:
            logger.info("Setting synthetic lighting override for stage.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        if self._sim is None:
            self._sim = habitat_sim.Simulator(self.cfg)
        else:  # edge case
            if self._sim.config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id:
                # we need to force a reset, so change the internal config scene name
                self._sim.config.sim_cfg.scene_id = "NONE"
            self._sim.reconfigure(self.cfg)

        # set _sim_settings scene name as actual loaded scene
        self._sim_settings["scene"] = self._sim.curr_scene_name

    def _get_sim_obs(self) -> np.ndarray:
        vis_sensor = self._sim._Simulator__sensors[self._agent_id][self._vis_sensor_name]
        vis_sensor.draw_observation()
        sim_obs = vis_sensor.get_observation()[:, :, :3]

        return sim_obs

    def _move_and_look(self, pressed_keys: Sequence) -> None:
        agent = self._sim.agents[self._agent_id]
        action_queue: List[str] = [
            act for key, act in self._key_to_action.items()
            if pressed_keys[key]
        ]
        [agent.act(x) for x in action_queue]

    def _setup_od(self) -> None:
        self._od_config = setup(self._od_settings)
        self._od_model = build_model(self._od_config)

        logger.info("Model:\n{}".format(self._od_model))
        DetectionCheckpointer(self._od_model, save_dir=self._od_config.OUTPUT_DIR).resume_or_load(
            self._od_config.MODEL.WEIGHTS, resume=True
        )
        self._od_model.eval()

    def _od_forward(self, obs_image) -> np.ndarray:
        thres = self._od_settings.threshold

        output_dir = self._od_config.OUTPUT_DIR
        min_size = self._od_config.INPUT.MIN_SIZE_TEST
        max_size = self._od_config.INPUT.MAX_SIZE_TEST
        augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        util.mkdir_if_missing(output_dir)

        category_path = os.path.join(util.file_parts(self._od_settings.config_file)[0], 'category_meta.json')

        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)
        cats = metadata['thing_classes']

        with torch.no_grad():
            image_shape = obs_image.shape[:2]  # h, w

            h, w = image_shape
            f_ndc = 4
            f = f_ndc * h / 2

            K = np.array([
                [f, 0.0, w / 2],
                [0.0, f, h / 2],
                [0.0, 0.0, 1.0]
            ])

            aug_input = T.AugInput(obs_image)
            _ = augmentations(aug_input)
            image = aug_input.image

            batched = [{
                'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(torch.device("cpu")),
                'height': image_shape[0], 'width': image_shape[1], 'K': K
            }]

            dets = self._od_model(batched)[0]['instances']
            n_det = len(dets)

            meshes = []
            meshes_text = []

            if n_det > 0:
                for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                        dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
                        dets.pred_pose, dets.scores, dets.pred_classes
                )):
                    if score < thres:
                        continue

                    cat = cats[cat_idx]

                    bbox3D = center_cam.tolist() + dimensions.tolist()
                    meshes_text.append('{} {:.2f}'.format(cat, score))
                    color = [c / 255.0 for c in util.get_color(idx)]
                    box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                    meshes.append(box_mesh)

            print('File: {} dets'.format(len(meshes)))

            if len(meshes) > 0:
                im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(
                    obs_image,
                    K,
                    meshes,
                    text=meshes_text,
                    scale=image.shape[0],
                    blend_weight=0.5,
                    blend_weight_overlay=0.85
                )
                return im_drawn_rgb
            else:
                return obs_image

    def _draw(self, obs: np.ndarray) -> None:
        vis_surf = self._obs_as_surf(obs)
        self._screen.blit(vis_surf, (0, 0))
        pg.display.update()

    def exec(self) -> None:
        sim_obs = self._get_sim_obs()
        self._draw(sim_obs)
        od_done = False

        prev_time = time.time()
        while True:
            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT:
                    self._quit()
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_ESCAPE:
                        self._quit()
                    elif e.key == pg.K_h:
                        self._print_help_text()
                    elif e.key == pg.K_q:
                        if not od_done:
                            self._draw(self._od_forward(sim_obs))
                            od_done = True
                        else:
                            self._draw(sim_obs)
                            od_done = False

            keys_pressed = pg.key.get_pressed()
            if any(keys_pressed[k] for k in self._key_to_action):
                self._move_and_look(keys_pressed)
                sim_obs = self._get_sim_obs()
                self._draw(sim_obs)
                od_done = False

            # slow down the viewer to self._target_fps
            curr_time = time.time()
            diff = curr_time - prev_time
            delay = max(1.0 / self._target_fps - diff, 0)
            time.sleep(delay)
            prev_time = curr_time

    @staticmethod
    def _obs_as_surf(obs: np.ndarray):
        return pg.surfarray.make_surface(np.rot90(np.fliplr(obs)))

    @staticmethod
    def _quit() -> None:
        pg.quit()
        sys.exit()

    @staticmethod
    def _print_help_text() -> None:
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
    parser = argparse.ArgumentParser()

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

    # Setting up _sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["stage_requires_lighting"] = args.stage_requires_lighting

    default_od_settings = {
        "config_file": "cubercnn://indoor/cubercnn_DLA34_FPN.yaml",
        "threshold": 0.25,
        "num_gpus": 0,
        "num_machines": 1,
        "machine_rank": 0,
        "opts": [
            "MODEL.WEIGHTS", "cubercnn://indoor/cubercnn_DLA34_FPN.pth",
            "OUTPUT_DIR", "output/demo/indoor_dla34"
        ]
    }
    InteractiveOdPyGameViewer(
        sim_settings,
        window_size=(args.window_height, args.window_width),
        od_settings=argparse.Namespace(**default_od_settings)
    ).exec()
