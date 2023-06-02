from .off_policy import *
from .collect_expert import collect_expert
from .on_policy import vanilla_on_policy_training_stage
from .evaluate import evaluate

def get_main_stage(config):
    if config.main_stage_type == "off_policy":
        return vanilla_off_policy_training_stage(config)
    elif config.main_stage_type == "her_off_policy":
        return her_off_policy_training_stage(config)
    elif config.main_stage_type == "collect_expert":
        return collect_expert(config)
    elif config.main_stage_type == "on_policy":
        return vanilla_on_policy_training_stage(config)
    elif config.main_stage_type == "evaluate":
        return evaluate(config)
    else:
        raise TypeError(
            f"training stage type : {config.training_stage_type} not supported"
        )
