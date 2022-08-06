from .off_policy import *
from .collect_expert import collect_expert
from .neighborhood_il import neighborhood_il
from .collect_toy_oracle import collect_toy_oracle

def get_main_stage(config):
    if config.main_stage_type == "off_policy":
        return vanilla_off_policy_training_stage(config)
    elif config.main_stage_type == "her_off_policy":
        return her_off_policy_training_stage(config)
    elif config.main_stage_type == "collect_expert":
        return collect_expert(config)
    elif config.main_stage_type == "collect_toy_oracle":
        return collect_toy_oracle(config)
    elif config.main_stage_type == "neighborhood_il":
        return neighborhood_il(config)
    else:
        raise TypeError(
            f"training stage type : {config.training_stage_type} not supported"
        )
