import wandb
import numpy as np
import random
import copy
from envs import get_env


class vanilla_off_policy_training_stage:
    def __init__(self, config):
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        wandb.init(
            project="RL_Implementation",
            name=f"{self.algo}_{self.env_id}",
            config=config,
        )

    def test(self, agent, env):
        total_reward = 0
        for i in range(3):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
        total_reward /= 3
        return total_reward

    def start(self, agent, env, storage):
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        if self.continue_training:
            agent.load_weight(self.env_id)
        if self.buffer_warmup:
            state = env.reset()
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                storage.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                    done = False
                else:
                    state = next_state
        best_testing_reward = -1e7
        best_episode = 0
        for i in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                loss_info = agent.update(storage)
                wandb.log(loss_info, commit=False)
                state = next_state
            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": i,
                    "buffer_size": len(storage),
                }
            )

            if i % 5 == 0:
                testing_reward = self.test(agent, env)
                if testing_reward > best_testing_reward:
                    agent.cache_weight()
                    best_testing_reward = testing_reward
                    best_episode = i
                wandb.log({"testing_reward": testing_reward, "testing_episode_num": i})
            if i % self.save_weight_period == 0:
                agent.save_weight(
                    best_testing_reward, self.algo, self.env_id, best_episode
                )
        agent.save_weight(best_testing_reward, self.algo, self.env_id, best_episode)


class her_off_policy_training_stage:
    def __init__(self, config):
        self.episodes = config.episodes
        self.buffer_warmup = config.buffer_warmup
        self.buffer_warmup_step = config.buffer_warmup_step
        self.algo = config.algo
        self.env_id = config.env
        self.save_weight_period = config.save_weight_period
        self.continue_training = config.continue_training
        self.test_env = get_env(self.env_id, wrapper_type="gym_robotic")
        wandb.init(
            project="RL_Implementation",
            name=f"{self.algo}_{self.env_id}_HER",
            config=config,
        )

    def test(self, agent, env):
        total_reward = 0
        for i in range(3):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
        total_reward /= 3
        return total_reward

    def start(self, agent, env, storage):
        self.train(agent, env, storage)

    def train(self, agent, env, storage):
        if self.continue_training:
            agent.load_weight(self.env_id)
        if self.buffer_warmup:
            state = env.reset()
            state = np.append(
                state["observation"],
                state["desired_goal"],
            )
            done = False
            while len(storage) < self.buffer_warmup_step:
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                next_state = np.append(
                    next_state["observation"],
                    next_state["desired_goal"],
                )
                storage.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                    state = np.append(
                        state["observation"],
                        state["desired_goal"],
                    )
                    done = False
                else:
                    state = next_state
        best_testing_reward = -1e7
        best_episode = 0
        for i in range(self.episodes):
            # episodic storage is for HER
            episodic_state = []
            episodic_action = []
            episodic_next_state = []
            episodic_done = []
            episodic_info = []
            state = env.reset()
            episodic_state.append(state)
            state = np.append(
                state["observation"],
                state["desired_goal"],
            )
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                episodic_state.append(next_state)
                episodic_action.append(action)
                episodic_next_state.append(next_state)
                episodic_done.append(done)
                episodic_info.append(info)
                next_state = np.append(
                    next_state["observation"],
                    next_state["desired_goal"],
                )
                total_reward += reward
                storage.store(state, action, reward, next_state, done)
                loss_info = agent.update(storage)
                wandb.log(loss_info, commit=False)
                state = next_state

            # her part
            pseudo_goals_idx = np.random.randint(len(episodic_next_state), size=int(np.ceil(len(episodic_next_state) / 10)))
            for pseudo_goal_idx in pseudo_goals_idx:
                pseudo_goal = episodic_next_state[pseudo_goal_idx]["achieved_goal"]
                for idx in range(pseudo_goal_idx+1):
                    state = episodic_state[idx]
                    state = np.append(
                        state["observation"],
                        pseudo_goal,
                    )
                    action = episodic_action[idx]
                    next_state = episodic_next_state[idx]
                    info = episodic_info[idx]
                    pseudo_reward = env.compute_reward(
                        next_state["achieved_goal"], pseudo_goal, info
                    )
                    done = (next_state["achieved_goal"] == pseudo_goal).all()
                    next_state = np.append(
                        next_state["observation"],
                        pseudo_goal,
                    )
                    storage.store(state, action, pseudo_reward, next_state, done)
            wandb.log(
                {
                    "training_reward": total_reward,
                    "episode_num": i,
                    "buffer_size": len(storage),
                }
            )

            if i % 5 == 0:
                testing_reward = self.test(agent, self.test_env)
                if testing_reward > best_testing_reward:
                    agent.cache_weight()
                    best_testing_reward = testing_reward
                    best_episode = i
                wandb.log({"testing_reward": testing_reward, "testing_episode_num": i})
            if i % self.save_weight_period == 0:
                agent.save_weight(
                    best_testing_reward, self.algo + "_her", self.env_id, best_episode
                )
        agent.save_weight(
            best_testing_reward, self.algo + "_her", self.env_id, best_episode
        )
