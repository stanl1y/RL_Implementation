from distutils.log import error


class collect_expert:
    def __init__(self, config):
        if config.expert_transition_num is not None:
            self.expert_transition_num = config.expert_transition_num
            self.based_on_transition_num = True
        elif config.expert_episode_num is not None:
            self.expert_episode_num = config.expert_episode_num
            self.based_on_transition_num = False
        else:
            raise ValueError(
                "either expert_transition_num or expert_episode_num must be set"
            )
        self.algo = config.algo
        self.env_id = config.env

    def start(self, agent, env, storage):
        agent.load_weight(self.algo,self.env_id)
        agent.eval()
        if self.based_on_transition_num:
            done = False
            total_reward=0
            episode_reward=0
            episode_num=0
            state = env.reset(seed=episode_num)
            while len(storage) < self.expert_transition_num:
                if done:
                    done = False
                    episode_num+=1
                    total_reward+=episode_reward
                    episode_reward=0
                    state = env.reset(seed=episode_num)
                action = agent.act(state, testing=True)
                next_state, reward, done, info = env.step(action)
                episode_reward+=reward
                storage.store(state, action, reward, next_state, done)
                state = next_state
            print(f"expert data recording finish, average reward : {round(total_reward/max(episode_num,1),4)}")
            storage.write_storage(
                self.based_on_transition_num,
                self.expert_transition_num,
                self.algo,
                self.env_id,
            )
        else:
            total_reward=0
            for e in range(self.expert_episode_num):
                done = False
                state = env.reset(seed=e)
                while not done:
                    action = agent.act(state, testing=True)
                    next_state, reward, done, info = env.step(action)
                    total_reward+=reward
                    storage.store(state, action, reward, next_state, done)
                    state = next_state
            storage.write_storage(
                self.based_on_transition_num,
                self.expert_episode_num,
                self.algo,
                self.env_id,
            )
            print(f"expert data recording finish, average reward : {round(total_reward/self.expert_episode_num,4)}")

