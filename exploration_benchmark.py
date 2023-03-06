import argparse

# from habitat_baselines.config.default import get_config
# from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
import habitat

from agents.exploration_agent import PPOAgent
from config import get_config

from config import get_config, MyConfigPlugin
from habitat_extensions.habitat_lab.tasks.exp.exp import ExplorationTask  # noqa (register ExplorationTask)

# registrations
import config.default_structured_configs  # noqa structured configs
import habitat_extensions.habitat_lab.tasks.exp.exp # noqa ExplorationVisitedLocationsReward
import habitat_extensions.habitat_lab.tasks.nav.nav # noqa TopDownMap
import habitat_extensions.habitat_lab.datasets.exploration_dataset # noqa register Exploration datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    overrides = args.overrides + [
        "+pth_gpu_id=0",
        "+model_path=" + args.model_path,
        "+random_seed=1",
    ]

    config = get_config(
        args.config_path,
        overrides,
    )

    agent = PPOAgent(config)
    challenge = habitat.Challenge(eval_remote=False)
    challenge.submit(agent)


if __name__ == "__main__":
    main()
