import argparse
import yaml
from rl_algo import get_rl_agent, get_util
from environment import get_env
from replay_buffer import get_replay_buffer
from main_stage import get_main_stage


def get_config():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--main_task",
        type=str,
        default="sac",
        help="which RL algo",
    )
    given_configs, remaining = parser.parse_known_args()
    with open(f"config_files/{given_configs.main_task}.yml", "r") as f:
        hyper = yaml.safe_load(f)
        parser.set_defaults(**hyper)

    parser.add_argument("--buffer_size", type=int, help="size of replay buffer")
    parser.add_argument(
        "--wrapper_type",
        type=str,
        help='basic for normal environment, gym robotic for training on robotic envs "directly", her is for training on robotic envs with hindsight experience replay',
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
        help="type of main stage(ex. off_policy, her_off_policy, collect_expert)",
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
    parser.add_argument(
        "--save_env_states", action="store_true", help="store the env state for 'set_state' or not"
    )
    parser.add_argument("--expert_episode_num", type=int, help="number of expert data")
    parser.add_argument(
        "--buffer_warmup_step",
        type=int,
        help="number of step of random walk in the initial of training",
    )
    parser.add_argument(
        "--render", action="store_true", help="whether to render when testing"
    )
    parser.add_argument(
        "--noisy_network", action="store_true", help="whether to render when testing"
    )
    parser.add_argument(
        "--no_bc", action="store_true", help="don't use behavior cloning when neighborhood il"
    )
    parser.add_argument(
        "--no_hard_negative_sampling", action="store_true", help="don't use hard negative sampling in neighborhood il"
    )
    parser.add_argument(
        "--entropy_loss_weight_decay_rate", type=float, help="decay rate of entropy loss weight"
    )
    parser.add_argument(
        "--log_name",
        type=str,
        help="nane of log file",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="",
        help="name of expert data",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        help="path of trained model weight",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        help="what kind of replay buffer",
    )
    args = parser.parse_args()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


if __name__ == "__main__":
    config, config_text = get_config()
    env = get_env(config.env, config.wrapper_type, config.terminate_when_unhealthy)
    agent = get_rl_agent(env, config)
    storage = get_replay_buffer(env, config)
    main_fn = get_main_stage(config)
    if hasattr(config, "util"):
        util_dict = get_util(env, config)
        main_fn.start(agent, env, storage, util_dict)
    else:
        main_fn.start(agent, env, storage)
