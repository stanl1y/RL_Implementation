import wandb
import numpy as np
import argparse
import yaml

class vanilla_off_policy_training_stage:
    def __init__(self, config):
        '''get neighbor model config'''
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        # self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        wandb.init(
            project="Neighborhood",
            name=f"{self.env_id}",
            config=config,
        )
    def start(self, agent, env, storage):
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        pass



