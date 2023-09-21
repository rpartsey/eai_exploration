# eai_rearrangement

## Exploration Task

### Requitements

Packages:
- habitat-sim
- habitat-lab (install from [this](https://github.com/rpartsey/habitat-lab/tree/rpartsey/train_with_non_scalar_metrics) branch or do the same changes as in [this](https://github.com/facebookresearch/habitat-lab/commit/ff2fb15a89ba80172ec8d16c510f0c782f0d6703) commit in your habita-lab fork yoursefl)
- (and their dependencies)

Datasets:
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

3\. Run exploration DD-PPO training:
```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -u run.py \
--config-name=ddppo_exploration.yaml \
benchmark=exploration_hm3d_10pct_depth_1scene_1episode \
habitat_baselines.torch_gpu_id=0 \
habitat_baselines.num_environments=1 \
habitat_baselines.evaluate=False
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
        "benchmark=exploration_hm3d_10pct_depth_1scene_1episode",
        "habitat_baselines.torch_gpu_id=0",
        "habitat_baselines.num_environments=1",
        "habitat_baselines.evaluate=False"
    ]
}
```
Note, if you are using Mac you may have problems with spawning processes by VectorEnv. In this case, you can use HABITAT_ENV_DEBUG=1 environment variable to run training using ThreadedVectorEnv.
