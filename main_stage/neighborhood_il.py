import wandb
import numpy as np
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import imageio
import time
import copy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class neighborhood_il:
    def __init__(self, config):
        """get neighbor model config"""
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        self.batch_size = config.batch_size
        self.neighbor_model_alpha = config.neighbor_model_alpha
        self.neighbor_criteria = nn.BCELoss(reduction="none")
        self.ood = config.ood
        self.bc_only = config.bc_only
        self.no_bc = config.no_bc
        self.update_neighbor_frequency = config.update_neighbor_frequency
        self.update_neighbor_step = config.update_neighbor_step
        self.update_neighbor_until = config.update_neighbor_until
        self.oracle_neighbor = config.oracle_neighbor
        self.discretize_reward = config.discretize_reward
        self.log_name = config.log_name
        self.duplicate_expert_last_state = config.duplicate_expert_last_state
        self.data_name = config.data_name
        self.auto_threshold_ratio = config.auto_threshold_ratio
        self.threshold_discount_factor = config.threshold_discount_factor
        self.fix_env_random_seed = config.fix_env_random_seed
        self.render = config.render
        self.hard_negative_sampling = not config.no_hard_negative_sampling
        self.use_env_done = config.use_env_done
        self.use_target_neighbor = config.use_target_neighbor
        self.tau = config.tau
        self.neighborhood_tau = config.neighborhood_tau
        self.entropy_loss_weight_decay_rate = config.entropy_loss_weight_decay_rate
        self.no_update_alpha = config.no_update_alpha
        self.infinite_neighbor_buffer = config.infinite_neighbor_buffer
        self.bc_pretraining = config.bc_pretraining
        self.hybrid = config.hybrid
        self.use_relative_reward = config.use_relative_reward
        self.state_only = config.state_only
        self.total_steps = 0
        self.critic_without_entropy = config.critic_without_entropy
        self.target_entropy_weight = config.target_entropy_weight
        self.reward_scaling_weight = config.reward_scaling_weight
        self.use_true_expert_relative_reward = config.use_true_expert_relative_reward
        self.low_hard_negative_weight = config.low_hard_negative_weight
        self.use_top_k= config.use_top_k
        self.use_pretrained_neighbor = config.use_pretrained_neighbor
        self.pretrained_neighbor_weight_path = config.pretrained_neighbor_weight_path
        if self.hard_negative_sampling:
            print("hard negative sampling")
        if self.auto_threshold_ratio:
            self.policy_threshold_ratio = 0.1
        else:
            self.policy_threshold_ratio = config.policy_threshold_ratio

        try:
            self.margin_value = config.margin_value
        except:
            self.margin_value = 0.1
        wandb.init(
            project="Neighborhood",
            name=f"{self.env_id}{self.log_name}",
            config=config,
        )

    def gen_data(self, storage):
        """
        generate expert next state data for reward calculation
        """
        expert_state, expert_action, _, expert_next_state, expert_done = storage.sample(
            -1, expert=True
        )
        if storage.to_tensor:
            self.expert_ns_data = torch.tile(
                expert_next_state, (self.batch_size, 1)
            ).to(device)
            self.testing_expert_ns_data = (
                torch.tile(expert_next_state, (1000, 1)).to(device).to(device)
            )
            self.testing_expert_ns_data0 = torch.repeat_interleave(
                expert_next_state, 1000, dim=0
            ).to(device)
        else:
            self.expert_ns_data = np.tile(expert_next_state, (self.batch_size, 1))
            self.expert_ns_data = torch.FloatTensor(self.expert_ns_data).to(device)
            self.testing_expert_ns_data = np.tile(expert_next_state, (1000, 1))
            self.testing_expert_ns_data = torch.FloatTensor(
                self.testing_expert_ns_data
            ).to(device)
            self.testing_expert_ns_data0 = torch.FloatTensor(
                np.repeat(expert_next_state, 1000, axis=0)
            ).to(device)

        self.expert_cartesian_product_state = torch.cat(
            (
                self.testing_expert_ns_data0,
                # in mujoco, 1000 is the number of expert states
                self.testing_expert_ns_data.reshape(
                    (-1, self.testing_expert_ns_data.shape[-1])
                ),
            ),
            dim=1,
        )

        """
        generate label and weight for neighbor model training
        """
        self.update_neighor_label = (
            torch.FloatTensor(
                np.concatenate(
                    (
                        np.ones(self.batch_size),
                        np.zeros(
                            self.batch_size * (2 if self.hard_negative_sampling else 1)
                        ),
                    )
                )
            )
            .view((-1, 1))
            .to(device)
        )
        if self.hard_negative_sampling:

            self.neighbor_loss_weight = (
                torch.cat(
                    (
                        torch.ones(self.batch_size) * self.neighbor_model_alpha,
                        torch.ones(self.batch_size) * (1 - self.neighbor_model_alpha),
                        torch.ones(self.batch_size)
                        * ((1 - self.neighbor_model_alpha)
                        if self.low_hard_negative_weight
                        else 1.0),
                    )
                )
                .view((-1, 1))
                .to(device)
            )
        else:
            self.neighbor_loss_weight = (
                torch.cat(
                    (
                        torch.ones(self.batch_size) * self.neighbor_model_alpha,
                        torch.ones(self.batch_size) * (1 - self.neighbor_model_alpha),
                    )
                )
                .view((-1, 1))
                .to(device)
            )
        self.expert_reward_ones = (
            torch.ones(self.batch_size).view((-1, 1)) * self.reward_scaling_weight
        )
        self.expert_reward_ones = self.expert_reward_ones.to(device)

    def start(self, agent, env, storage, util_dict):
        if self.oracle_neighbor:
            self.NeighborhoodNet = util_dict["OracleNeighborhoodNet"].to(device)
        else:
            self.NeighborhoodNet = util_dict["NeighborhoodNet"].to(device)
            if self.use_pretrained_neighbor:
                self.NeighborhoodNet.load_state_dict(
                    torch.load(self.pretrained_neighbor_weight_path)["neighborhood_state_dict"]
                )
                print(f"load pretrained neighbor model from {self.pretrained_neighbor_weight_path}")
            self.NeighborhoodNet_optimizer = torch.optim.Adam(
                self.NeighborhoodNet.parameters(), lr=3e-4
            )
        if self.use_target_neighbor:
            self.TargetNeighborhoodNet = copy.deepcopy(self.NeighborhoodNet).to(device)
        storage.load_expert_data(
            algo=self.algo,
            env_id=self.env_id,
            duplicate_expert_last_state=self.duplicate_expert_last_state,
            data_name=self.data_name,
        )
        self.gen_data(storage)
        self.train(agent, env, storage)

    # def test_with_neighborhood_model(self, agent, env):
    #     # agent.eval()
    #     total_reward = []
    #     for i in range(10):
    #         state_dim = env.observation_space.shape[0]
    #         traj_ns = np.ones((1000, state_dim))
    #         mask = np.zeros(1000)
    #         step_counter = 0
    #         state = env.reset()
    #         done = False
    #         while not done:
    #             action = agent.act(state, testing=True)
    #             # agent.q_network.reset_noise()
    #             next_state, _, done, info = env.step(action)
    #             state = next_state
    #             traj_ns[step_counter] = next_state
    #             step_counter += 1
    #         mask[:step_counter] = 1
    #         traj_ns = torch.FloatTensor(traj_ns).to(device)

    #         cartesian_product_state = torch.cat(
    #             (
    #                 torch.repeat_interleave(traj_ns, 1000, dim=0),
    #                 # in mujoco, 1000 is the number of expert states
    #                 self.testing_expert_ns_data.reshape((-1, state_dim)),
    #             ),
    #             dim=1,
    #         )

    #         with torch.no_grad():
    #             prob = self.NeighborhoodNet(cartesian_product_state)
    #         prob = prob.reshape((1000, 1000)).sum(dim=1)
    #         prob = prob.cpu().numpy() * mask
    #         reward = prob.sum()
    #         total_reward.append(reward)
    #     with torch.no_grad():
    #         expert_prob = self.NeighborhoodNet(self.expert_cartesian_product_state)
    #     expert_reward = expert_prob.cpu().numpy().sum()

    #     total_reward = np.array(total_reward)
    #     total_reward_mean = total_reward.mean()
    #     total_reward_std = total_reward.std()
    #     total_reward_min = total_reward.min()
    #     total_reward_max = total_reward.max()
    #     return {
    #         "neighborhood_agent_reward_mean": total_reward_mean,
    #         "neighborhood_expert_reward": expert_reward,
    #         "neighborhood_agent_reward_std": total_reward_std,
    #         "neighborhood_agent_reward_min": total_reward_min,
    #         "neighborhood_agent_reward_max": total_reward_max,
    #         "relative_neighborhood_agent_reward": total_reward_mean / expert_reward,
    #     }

    def test(self, agent, env, render_id=0):
        # agent.eval()
        total_reward = []
        total_neighborhood_reward = []
        render = self.render and render_id % 40 == 0
        if render:
            frame_buffer = []
            if not os.path.exists(f"./experiment_logs/{self.env_id}/{self.log_name}/"):
                os.makedirs(f"./experiment_logs/{self.env_id}/{self.log_name}/")
        for i in range(10):
            state_dim = env.observation_space.shape[0]
            traj_ns = np.ones((1000, state_dim))
            mask = np.zeros(1000)
            step_counter = 0
            state = env.reset()
            done = False
            episode_reward = 0
            if self.ood:
                for _ in range(5):
                    state, reward, done, info = env.step(
                        env.action_space.sample()
                    )  # env.action_space.sample()
            while not done:
                action = agent.act(state, testing=True)
                # agent.q_network.reset_noise()
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                if render:
                    frame_buffer.append(env.render(mode="rgb_array"))
                state = next_state
                traj_ns[step_counter] = next_state
                step_counter += 1
            total_reward.append(episode_reward)

            mask[:step_counter] = 1
            traj_ns = torch.FloatTensor(traj_ns).to(device)
            cartesian_product_state = torch.cat(
                (
                    torch.repeat_interleave(traj_ns, 1000, dim=0),
                    # in mujoco, 1000 is the number of expert states
                    self.testing_expert_ns_data.reshape((-1, state_dim)),
                ),
                dim=1,
            )
            with torch.no_grad():
                prob = self.NeighborhoodNet(cartesian_product_state)
            prob = prob.reshape((1000, 1000)).sum(dim=1)
            prob = prob.cpu().numpy() * mask
            reward = prob.sum()
            total_neighborhood_reward.append(reward)
        with torch.no_grad():
            expert_prob = self.NeighborhoodNet(self.expert_cartesian_product_state)
        expert_reward = expert_prob.cpu().numpy().sum()
        total_neighborhood_reward = np.array(total_neighborhood_reward)
        total_neighborhood_reward_mean = total_neighborhood_reward.mean()
        total_neighborhood_reward_std = total_neighborhood_reward.std()
        total_neighborhood_reward_min = total_neighborhood_reward.min()
        total_neighborhood_reward_max = total_neighborhood_reward.max()
        if render:
            imageio.mimsave(
                f"./experiment_logs/{self.env_id}/{self.log_name}/{render_id}.gif",
                frame_buffer,
            )
        total_reward = np.array(total_reward)
        total_reward_mean = total_reward.mean()
        total_reward_std = total_reward.std()
        total_reward_min = total_reward.min()
        total_reward_max = total_reward.max()
        # agent.train()
        return {
            "testing_reward_mean": total_reward_mean,
            "testing_reward_std": total_reward_std,
            "testing_reward_min": total_reward_min,
            "testing_reward_max": total_reward_max,
            "neighborhood_agent_reward_mean": total_neighborhood_reward_mean,
            "neighborhood_expert_reward": expert_reward,
            "neighborhood_agent_reward_std": total_neighborhood_reward_std,
            "neighborhood_agent_reward_min": total_neighborhood_reward_min,
            "neighborhood_agent_reward_max": total_neighborhood_reward_max,
            "relative_neighborhood_agent_reward": total_neighborhood_reward_mean
            / expert_reward,
        }

    def train(self, agent, env, storage):
        if self.bc_pretraining and not self.state_only:
            (expert_state, expert_action, _, _, _, _,) = storage.sample(
                self.batch_size,
                expert=True,
            )
            if not storage.to_tensor:
                expert_state = torch.FloatTensor(expert_state)
                expert_action = torch.FloatTensor(expert_action)
            expert_state = expert_state.to(device)
            expert_action = expert_action.to(device)
            for _ in range(50000):
                bc_loss = agent.bc_update(expert_state, expert_action, use_mu=False)
            print(f"BC pretraining finished, BC loss:{bc_loss}")
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
        if not self.use_pretrained_neighbor:
            for _ in range(1000):
                neighbor_loss = self.update_neighbor_model(storage)
        self.best_testing_reward = -1e7
        self.best_testing_neighborhood_reward = -1e7
        best_episode = 0
        print("warmup finished")
        for episode in range(self.episodes):
            if (
                not self.oracle_neighbor
                and episode % self.update_neighbor_frequency == 0
                and episode <= self.update_neighbor_until
                and not self.use_target_neighbor
                and not self.use_pretrained_neighbor
            ):
                for _ in range(self.update_neighbor_step):
                    neighbor_loss = self.update_neighbor_model(storage)
                wandb.log({"neighbor_model_loss": neighbor_loss}, commit=False)
            if self.hybrid and np.random.rand() < 0.2:
                state, _, _, _, done, expert_env_state = storage.sample(
                    batch_size=1,
                    expert=True,
                    return_expert_env_states=True,
                    exclude_tail_num=1,
                )
                env.reset()
                env.sim.set_state(expert_env_state)
                env.sim.forward()
                state = state[0]
            else:
                if self.fix_env_random_seed:
                    state = env.reset(seed=0)
                else:
                    state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                state = next_state
                neighbor_loss = 0
                if self.use_target_neighbor and not self.use_pretrained_neighbor:
                    neighbor_loss = self.update_neighbor_model(storage)
                loss_info = agent.update_using_neighborhood_reward(
                    storage,
                    self.NeighborhoodNet
                    if not self.use_target_neighbor
                    else self.TargetNeighborhoodNet,
                    self.expert_ns_data,
                    self.expert_reward_ones,
                    self.margin_value,
                    self.bc_only,
                    self.no_bc,
                    self.oracle_neighbor,
                    self.discretize_reward,
                    self.policy_threshold_ratio,
                    self.use_env_done,
                    self.no_update_alpha,
                    self.use_relative_reward,
                    self.state_only,
                    self.critic_without_entropy,
                    self.target_entropy_weight,
                    self.reward_scaling_weight,
                    self.use_true_expert_relative_reward,
                    self.use_top_k,
                )
                self.total_steps += 1
            agent.entropy_loss_weight *= self.entropy_loss_weight_decay_rate
            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": episode,
                    "buffer_size": len(storage),
                    "threshold_ratio": self.policy_threshold_ratio,
                    **loss_info,
                    "neighbor_model_loss": neighbor_loss,
                    "entropy_loss_weight": agent.entropy_loss_weight,
                    "total_steps": self.total_steps,
                }
            )
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()

            if episode % 5 == 0 and episode > 0:
                testing_reward = self.test(
                    agent, env, render_id=episode if self.render else None
                )
                if testing_reward["testing_reward_mean"] > self.best_testing_reward:
                    self.best_testing_reward = testing_reward["testing_reward_mean"]
                    self.save_model_weight(agent, episode)

                # neighbor_testing_reward = self.test_with_neighborhood_model(agent, env)
                if (
                    testing_reward["relative_neighborhood_agent_reward"]
                    > self.best_testing_neighborhood_reward
                ):
                    self.best_testing_neighborhood_reward = testing_reward[
                        "relative_neighborhood_agent_reward"
                    ]
                    self.save_model_weight(agent, episode, oracle_reward=False)
                wandb.log(
                    {
                        **testing_reward,
                        "testing_episode_num": episode,
                        "best_testing_reward": self.best_testing_reward,
                        "best_testing_neighborhood_reward": self.best_testing_neighborhood_reward,
                    }
                )
                if hasattr(env, "eval_toy_q"):
                    env.eval_toy_q(
                        agent,
                        self.NeighborhoodNet,
                        storage,
                        f"./experiment_logs/{self.env_id}{self.log_name}/",
                        episode,
                        self.oracle_neighbor,
                    )
            if self.auto_threshold_ratio:
                self.policy_threshold_ratio *= self.threshold_discount_factor

    def save_model_weight(self, agent, episode, oracle_reward=True):
        if oracle_reward:
            best_reward = self.best_testing_reward
        else:
            best_reward = self.best_testing_neighborhood_reward
        agent.cache_weight()
        agent.save_weight(
            best_testing_reward=best_reward,
            algo="neighborhood_il",
            env_id=self.env_id,
            episodes=episode,
            log_name=self.log_name + ("_oracle" if oracle_reward else "_neighbor"),
            oracle_reward=oracle_reward,
        )
        path = f"./trained_model/neighborhood/{self.env_id}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        data = {
            "episodes": episode,
            "neighborhood_state_dict": self.NeighborhoodNet.state_dict(),
            "neighborhood_optimizer_state_dict": self.NeighborhoodNet_optimizer.state_dict(),
        }

        file_path = os.path.join(
            path,
            f"episode{episode}_reward{round(best_reward,4)}_{self.log_name}"
            + ("_oracle" if oracle_reward else "_neighbor")
            + ".pt",
        )
        torch.save(data, file_path)
        if oracle_reward:
            try:
                os.remove(self.previous_checkpoint_path)
            except:
                pass
            self.previous_checkpoint_path = file_path
        else:
            try:
                os.remove(self.neighbor_reward_previous_checkpoint_path)
            except:
                pass
            self.neighbor_reward_previous_checkpoint_path = file_path

    def update_neighbor_model(self, storage):
        state, action, reward, next_state, done = storage.sample(
            self.batch_size, only_last_1m=not self.infinite_neighbor_buffer
        )
        if not storage.to_tensor:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
        next_state_shift = torch.roll(next_state, 1, 0)  # for negative samples
        """
        TODO:easy positive samples(itself)
        """
        posivite_data = torch.cat((state, next_state), axis=1)
        negative_data = torch.cat((state, next_state_shift), axis=1)
        if not self.hard_negative_sampling:
            input_data = torch.cat((posivite_data, negative_data), axis=0)
        else:
            negative_hard_data = torch.cat((next_state, state), axis=1)
            input_data = torch.cat(
                (posivite_data, negative_data, negative_hard_data), axis=0
            )
        input_data = input_data.to(device)
        prediction = self.NeighborhoodNet(input_data)

        # predict positive samples
        loss = self.neighbor_criteria(
            prediction,
            self.update_neighor_label,
        )
        loss = torch.mean(loss * self.neighbor_loss_weight)
        self.NeighborhoodNet_optimizer.zero_grad()
        loss.backward()
        self.NeighborhoodNet_optimizer.step()
        if self.use_target_neighbor:
            self.update_target_neighbor_model()
        return loss.item()

    def update_target_neighbor_model(self):
        for target_param, param in zip(
            self.TargetNeighborhoodNet.parameters(), self.NeighborhoodNet.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.neighborhood_tau)
                + param.data * self.neighborhood_tau
            )


# CUDA_VISIBLE_DEVICES=0 python3 main.py --main_stage neighborhood_il --main_task neighborhood_dsac --env LunarLander-v2 --wrapper basic --episode 2000 --log_name expectile99_ac_duplicateLast_nextStateReward_disReward
