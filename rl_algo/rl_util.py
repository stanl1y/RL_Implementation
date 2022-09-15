from .neighborhood_il import *


def get_util(env, config):
    util_dict = {}
    for util in config.util:
        if util == "NeighborhoodNet":
            util_dict[util] = NeighborhoodNet(
                input_dim=env.get_observation_dim() * 2, hidden_dim=config.hidden_dim
            )
        if util == "OracleNeighborhoodNet":
            util_dict[util] = OracleNeighborhoodNet(
                input_dim=env.get_observation_dim() * 2, hidden_dim=config.hidden_dim
            )
    return util_dict
