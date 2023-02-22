import wandb
import numpy as np
import os
import imageio


class evaluate:
    def __init__(self, config):
        self.episodes = config.episodes
        self.algo = config.algo
        self.env_id = config.env
        self.render = config.render
        self.ood = config.ood
        self.weight_path = config.weight_path
        wandb.init(
            project="RL_Implementation",
            name=f"{self.algo}_{self.env_id}_eval" + ("_ood" if self.ood else ""),
            config=config,
        )

    def test(self, agent, env):
        agent.eval()
        render = self.render
        save_dir = ""
        if render:
            frame_buffer = []
            if self.ood:
                if not os.path.exists(
                    f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
                ):
                    os.makedirs(
                        f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
                    )
                    save_dir = f"./experiment_logs/{self.env_id}/{self.algo}_eval_ood/"
            else:
                if not os.path.exists(
                    f"./experiment_logs/{self.env_id}/{self.algo}_eval/"
                ):
                    os.makedirs(f"./experiment_logs/{self.env_id}/{self.algo}_eval/")
                    save_dir = f"./experiment_logs/{self.env_id}/{self.algo}_eval/"
        for i in range(self.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            if self.ood:
                for _ in range(10):
                    state, reward, done, info = env.step(
                        env.action_space.sample()
                    )  # env.action_space.sample()
                    if render:
                        frame_buffer.append(env.render(mode="rgb_array"))
            while not done:
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                if render:
                    frame_buffer.append(env.render(mode="rgb_array"))
                total_reward += reward
                state = next_state
            if render:
                imageio.mimsave(
                    f"{save_dir}{i}.gif",
                    frame_buffer,
                )
                frame_buffer=[]
            wandb.log(
                {
                    "testing_reward": total_reward,
                    "episode_num": i,
                }
            )

    def start(self, agent, env, storage):
        if self.weight_path:
            agent.load_weight(path=self.weight_path)
        else:
            agent.load_weight(algo=self.algo, env_id=self.env_id)
        agent.eval()
        self.test(agent, env)
