# eai_rearrangement

## Exploration Task

### Requitements

**Packages**

Core:
- habitat-sim (==0.3.0, from this [commit](https://github.com/facebookresearch/habitat-sim/commit/dfb388e29e5e1f25da4b576305e85bdc0be140b8))
- habitat-lab (==0.3.0, install in editable mode from [this](https://github.com/rpartsey/habitat-lab/tree/eai_exploration) branch)
- (and their dependencies)

Third party:
- eai-vc (install from [this](https://github.com/facebookresearch/eai-vc/commit/76fe35e87b1937168f1ec4b236e863451883eaf3) commit) `cd eai-cv && pip install -e ./vc_models`

**Datasets**
- HM3D [scenes dataset](https://aihabitat.org/datasets/hm3d/). Note, it can be also downloaded using habitat-sim [datasets_download.py](https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py) utility: `python -m habitat_sim.utils.datasets_download --uids hm3d --data-path path/to/data/`
- HM3D PointGoal Navigation	[task dataset](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)


### Experiments
1\. Navigate to eai_exploration repository root.

2\. Convert PointNav task dataset to Exploration task dataset:
```bash
python habitat_extensions/habitat_lab/datasets/exploration_dataset_generation.py \
--pointnav_dataset_path path/to/data/datasets/pointnav/hm3d/v1 \
--exploration_dataset_path path/to/data/datasets/exploration/hm3d/v1 \
--splits train_10_percent
```
or if you are using Visual Studio Code:
```json
{
    "type": "python",
    "request": "launch",
    "program": "habitat_extensions/habitat_lab/datasets/exploration_dataset_generation.py",
    "args": [
        "--pointnav_dataset_path", "data/datasets/pointnav/hm3d/v1",
        "--exploration_dataset_path", "data/datasets/exploration/hm3d/v1",
        "--splits", "train_10_percent"
    ]
}
```

3\. Run depth agent exploration DD-PPO training:
```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -u run.py \
--config-name=ddppo_exploration.yaml \
benchmark=exploration_hm3d_10pct_1scene_1episode \
habitat_baselines.torch_gpu_id=0 \
habitat_baselines.num_environments=1 \
habitat_baselines.evaluate=False
+habitat/simulator/agents@habitat.simulator.agents.main_agent=depth_agent
```

Expected output:
```bash
2023-04-28 11:44:06,486 Initializing dataset ExplorationStaticDataset
2023-04-28 11:44:07,829 Initializing dataset ExplorationStaticDataset
2023-04-28 11:44:07,949 initializing sim Sim-v0
2023-04-28 11:44:08,064 Initializing task Exp-v0
2023-04-28 11:44:09,441 agent number of parameters: 12389253
2023-04-28 11:44:26,018 update: 10      fps: 80.387
2023-04-28 11:44:26,018 update: 10      env-time: 5.363s        pth-time: 10.434s       frames: 1280
2023-04-28 11:44:26,018 Average window size: 10  exploration_success: 0.000  exploration_vlr: 1.339  reward: 6.465  scene_coverage: 0.082
```
or if you are using Visual Studio Code:
```json
{
    "type": "python",
    "request": "launch",
    "program": "run.py",
    "env": {
        "MAGNUM_LOG": "quiet",
        "HABITAT_SIM_LOG": "quiet",
    },
    "args": [
        "--config-name", "ddppo_exploration.yaml",
        "benchmark=exploration_hm3d_10pct_1scene_1episode",
        "habitat_baselines.torch_gpu_id=0",
        "habitat_baselines.num_environments=1",
        "habitat_baselines.evaluate=False",
        "+habitat/simulator/agents@habitat.simulator.agents.main_agent=depth_agent",
    ]
}
```
Note, if you are using Mac you may have problems with spawning processes by VectorEnv. In this case, you can use HABITAT_ENV_DEBUG=1 environment variable to run training using ThreadedVectorEnv.

**CLI overrides**

Agent type can be changed by specifying any of available agents configs (depth_agent, rgb_agent, rgbd_agent):
```bash
+habitat/simulator/agents@habitat.simulator.agents.main_agent=depth_agent
```

To controll sensor resolution add:
```bash
habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=256 \
habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=256
```

Lab sensors can be added by specifying any of available sensors configs. For example, to add GPS + Compass add:
```
+habitat/task/lab_sensors@habitat.task.lab_sensors.pointgoal_with_gps_compass_sensor=pointgoal_with_gps_compass_sensor
```

To use VC1NetPolicy (with VC-1 as visual encoder) add:
```bash
habitat_baselines.rl.policy.main_agent.name=VC1NetPolicy
```
