import argparse

import habitat

from agents.exploration_agent import PPOAgent
from config import get_config

# registration:
import config.default_structured_configs  # noqa
import habitat_extensions.habitat_lab.tasks.exp.task # noqa
import habitat_extensions.habitat_lab.tasks.exp.measures # noqa
import habitat_extensions.habitat_lab.tasks.nav.measures # noqa
import habitat_extensions.habitat_lab.datasets.exploration_dataset # noqa


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
