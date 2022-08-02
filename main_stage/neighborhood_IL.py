import wandb
import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class vanilla_off_policy_training_stage:
    def __init__(self, config):
        """get neighbor model config"""
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        # self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        self.batch_size = config.batch_size
        self.neighbor_model_alpha = config.neighbor_model_alpha
        self.neighbor_criteria = nn.BCELoss(reduction="none")
        wandb.init(
            project="Neighborhood",
            name=f"{self.env_id}",
            config=config,
        )

    def start(self, agent, env, storage, util_dict):
        self.NeighborhoodNet = util_dict["NeighborhoodNet"]
        self.NeighborhoodNet_optimizer = torch.optim.Adam(
            self.NeighborhoodNet.parameters(), lr=3e-4
        )
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        if self.buffer_warmup:
            state = env.reset()
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                    done = False
                else:
                    state = next_state
        for episode in range(self.episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                state = next_state
                loss = self.update_neighbor_model(storage)
                wandb.log({"loss": loss}, commit=False)
                


            if episode % self.save_weight_period == 0:
                path = f"./trained_model/neighborhood/{self.env_id}/"
                if not os.path.isdir(path):
                    os.makedirs(path)
                data = {
                    "episodes": episode,
                    "neighborhood_state_dict": self.NeighborhoodNet.state_dict(),
                    "neighborhood_optimizer_state_dict": self.NeighborhoodNet_optimizer.state_dict(),
                }

                file_path = os.path.join(
                    path, f"episode{episode}.pt"
                )
                torch.save(data, file_path)
                try:
                    os.remove(self.previous_checkpoint_path)
                except:
                    pass
                self.previous_checkpoint_path = file_path


    def update_neighbor_model(self, storage):
        state, action, reward, next_state, done = storage.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        next_state_shift = torch.roll(next_state, 1, 0)  # for negative samples
        label = (
            torch.LongTensor(
                [np.concatenate((np.ones(state.shape[0]), np.zeros(state.shape[0])))]
            )
            .view((-1, 1))
            .to(device)
        )
        loss_weight = torch.cat(
            (
                torch.ones(state.shape[0]) * self.neighbor_model_alpha,
                torch.ones(state.shape[0]) * (1 - self.neighbor_model_alpha),
            )
        ).view((-1, 1))
        # predict positive samples
        posivite = self.NeighborhoodNet(torch.cat((state, next_state), axis=1))
        negative = self.NeighborhoodNet(torch.cat((state, next_state_shift), axis=1))
        loss = self.neighbor_criteria(torch.cat((posivite, negative), axis=1), label)
        loss = torch.mean(loss*loss_weight)
        self.NeighborhoodNet_optimizer.zero_grad()
        loss.backward()
        self.NeighborhoodNet_optimizer.step()
        return loss.item()
