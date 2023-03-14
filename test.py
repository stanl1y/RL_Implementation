import pickle
import os
env_id="Humanoid-v3"
algo="sac"
path = f"./saved_expert_transition/{env_id}/{algo}/"
onlyfiles = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
path = onlyfiles[0]
with open(path, "rb") as handle:
    data = pickle.load(handle)
rewards=0
for r in data["rewards"]:
    rewards+=r
print(rewards)