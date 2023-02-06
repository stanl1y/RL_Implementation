import numpy as np
import os
import pickle


class normal_replay_buffer:
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.storage_index = 0
        self.states = np.empty((size, state_dim))
        self.actions = np.empty((size, action_dim))
        self.rewards = np.empty((size, 1))
        self.next_states = np.empty((size, state_dim))
        self.dones = np.empty((size, 1))

    def store(self, s, a, r, ss, d):
        index = self.storage_index % self.size
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.next_states[index] = ss
        self.dones[index] = d
        self.storage_index += 1

    def clear(self):
        self.storage_index = 0

    def sample(self, batch_size, expert=False):
        if expert:
            if batch_size == -1:
                indices = np.random.permutation(len(self.expert_states))
            else:
                indices = np.random.choice(len(self.expert_states), batch_size)
            return (
                self.expert_states[indices],
                self.expert_actions[indices],
                self.expert_rewards[indices],
                self.expert_next_states[indices],
                self.expert_dones[indices],
            )
        else:
            index = np.random.randint(
                min(self.storage_index, self.size), size=batch_size
            )
            return (
                self.states[index],
                self.actions[index],
                self.rewards[index],
                self.next_states[index],
                self.dones[index],
            )

    def write_storage(
        self, based_on_transition_num, expert_data_num, algo, env_id, data_name=""
    ):
        if data_name != "":
            data_name = f"_{data_name}"
        path = f"./saved_expert_transition/{env_id}/{algo}/"
        if not os.path.isdir(path):
            os.makedirs(path)
        save_idx = min(self.storage_index, self.size)
        print(save_idx)
        data = {
            "states": self.states[:save_idx],
            "actions": self.actions[:save_idx],
            "rewards": self.rewards[:save_idx],
            "next_states": self.next_states[:save_idx],
            "dones": self.dones[:save_idx],
        }
        if based_on_transition_num:
            file_name = f"transition_num{expert_data_num}{data_name}.pkl"
        else:
            file_name = f"episode_num{expert_data_num}{data_name}.pkl"
        print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), "wb") as handle:
            pickle.dump(data, handle)

    def load_expert_data(self, algo, env_id, duplicate_expert_last_state, data_name=""):
        if data_name == "":
            path = f"./saved_expert_transition/{env_id}/{algo}/"
            if not os.path.isdir(path):
                path = f"./saved_expert_transition/{env_id}/oracle/"
            assert os.path.isdir(path)
            onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
            path = onlyfiles[0]
        else:
            path = f"./saved_expert_transition/{env_id}/{data_name}.pkl"
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        print(f"load expert data from path : {path}")
        self.expert_states = data["states"]
        self.expert_actions = data["actions"]
        self.expert_rewards = data["rewards"]
        self.expert_next_states = data["next_states"]
        self.expert_dones = data["dones"]
        if duplicate_expert_last_state:
            done_idx = np.argwhere(self.expert_dones.reshape(-1) == 1).reshape(-1)
            done_expert_states = self.expert_states[done_idx]
            done_expert_actions = self.expert_actions[done_idx]
            done_expert_rewards = self.expert_rewards[done_idx]
            done_expert_next_states = self.expert_next_states[done_idx]
            done_expert_dones = self.expert_dones[done_idx]
            for _ in range(5):
                self.expert_states = np.concatenate(
                    (
                        self.expert_states,
                        done_expert_states,
                    ),
                    axis=0,
                )
                self.expert_actions = np.concatenate(
                    (
                        self.expert_actions,
                        done_expert_actions,
                    ),
                    axis=0,
                )
                self.expert_rewards = np.concatenate(
                    (
                        self.expert_rewards,
                        done_expert_rewards,
                    ),
                    axis=0,
                )
                self.expert_next_states = np.concatenate(
                    (
                        self.expert_next_states,
                        done_expert_next_states,
                    ),
                    axis=0,
                )
                self.expert_dones = np.concatenate(
                    (
                        self.expert_dones,
                        done_expert_dones,
                    ),
                    axis=0,
                )

    def __len__(self):
        return min(self.storage_index, self.size)
