import pickle
import os
import numpy as np
# path = f"./saved_expert_transition/Humanoid-v3/sac/episode_num1_6494.pkl"#humanoid_sgld_max.pkl
# with open(path, "rb") as handle:
#     data = pickle.load(handle)
#     # print(data["states"][0])
#     print(data["actions"][0])
#     # print(data["rewards"].shape)
#     # print(data["next_states"].shape)
#     # print(data["dones"].shape)
#     # print(len(data["env_states"]))
# print("-----")
path = f"./saved_expert_transition/Humanoid-v3/sac/episode_num1_7463.pkl"#humanoid_sgld_max.pkl
with open(path, "rb") as handle:
    data = pickle.load(handle)
    #clipping
    # data["actions"]=np.clip(data["actions"],-0.4,0.4)
    # print(data["states"][0])
    print(data["actions"][0])
    # print(data["rewards"].shape)
    # print(data["next_states"].shape)
    # print(data["dones"].shape)
    # print(len(data["env_states"]))
    # data["rewards"] = data["rewards"].reshape(-1,1)
    # data["dones"] = data["dones"].reshape(-1,1)

# write back
# dump_path = f"./saved_expert_transition/Humanoid-v3/sac/episode_num1_7463.pkl"
# with open(dump_path, "wb") as handle:
#     pickle.dump(data, handle)


# print(data.keys())
# path = f"./saved_expert_transition/Humanoid-v3/sac/humanoid_sgld_max.pkl"#humanoid_sgld_max.pkl
# with open(path, "rb") as handle:
#     data = pickle.load(handle)
# tmp={}
# tmp["states"]=[]
# tmp["actions"]=[]
# tmp["rewards"]=[]
# tmp["next_states"]=[]
# tmp["dones"]=[]
# tmp["env_states"]=[]
# for (s,o,a,o2,r,ep_l) in data:
#     tmp["states"].append(o)
#     tmp["actions"].append(a)
#     tmp["rewards"].append(r)
#     tmp["next_states"].append(o2)
#     tmp["dones"].append(0)
#     tmp["env_states"].append(s)
# tmp["states"]=np.array(tmp["states"])
# tmp["actions"]=np.array(tmp["actions"])
# tmp["rewards"]=np.array(tmp["rewards"])
# tmp["next_states"]=np.array(tmp["next_states"])
# tmp["dones"]=np.array(tmp["dones"])
# print(tmp["rewards"].sum())

# dump_path = f"./saved_expert_transition/Humanoid-v3/sac/episode_num1_7463.pkl"
# with open(dump_path, "wb") as handle:
#     pickle.dump(tmp, handle)




