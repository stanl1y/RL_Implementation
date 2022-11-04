from .normal_replay_buffer import normal_replay_buffer
from .onpolicy_replay_buffer import onpolicy_replay_buffer

def get_replay_buffer(env, config):
    if config.buffer_type == "normal":
        return normal_replay_buffer(
            size=config.buffer_size,
            state_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim() if env.is_continuous() else 1,
        )
    elif config.buffer_type == "onpolicy":
        return onpolicy_replay_buffer(
            size=config.buffer_size,
            state_dim=env.get_observation_dim(),
            action_dim=env.get_action_dim() if env.is_continuous() else 1,
        )
    else:
        raise TypeError(f"replay buffer type : {config.buffer_type} not supported")
