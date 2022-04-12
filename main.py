import argparse
import yaml
from rl_algo import get_rl_agent
from envs import get_env
from replay_buffer import get_replay_buffer
from main_stage import get_main_stage


def get_config():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        help="which RL algo",
    )
    given_configs, remaining = parser.parse_known_args()
    with open(f"config_files/{given_configs.algo}.yml", "r") as f:
        hyper = yaml.safe_load(f)
        parser.set_defaults(**hyper)

    parser.add_argument(
        "--wrapper_type",
        type=str,
        help="which RL algo",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Gym environment name, default: CartPole-v0",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, help="Seed, default: 1")
    parser.add_argument(
        "--main_stage_type",
        type=str,
        help="type of main stage(ex. off_policy, collect_expert)",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size, default: 256")
    parser.add_argument("--hidden_dim", type=int, help="dimension of hidden layer 256")
    parser.add_argument(
        "--use_ounoise", action="store_true", help="use ou noise or not"
    )
    parser.add_argument(
        "--continue_training", action="store_true", help="use ou noise or not"
    )
    parser.add_argument(
        "--expert_transition_num", type=int, help="number of expert data"
    )
    parser.add_argument("--expert_episode_num", type=int, help="number of expert data")
    parser.add_argument(
        "--buffer_warmup_step",
        type=int,
        help="number of step of random walk in the initial of training",
    )

    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


if __name__ == "__main__":
    config, config_text = get_config()
    env = get_env(config.env, config.wrapper_type)
    agent = get_rl_agent(env, config)
    storage = get_replay_buffer(env, config)
    main_fn = get_main_stage(config)
    main_fn.start(agent, env, storage)
