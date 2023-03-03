# import gym
# env=gym.make("HalfCheetah-v3")
# env.reset()
# s=env.sim.get_state()
# for _ in range(995):
#     env.step(env.action_space.sample())
# s=env.sim.get_state()
# env.reset()
# env.sim.set_state(s)
# env.sim.forward()
# for _ in range(10):
#     _,_,d,_=env.step(env.action_space.sample())
#     print(d)




import numpy as np
import pickle

env_id = "HalfCheetah-v3"
data_name = "sac/episode_num1_15251"
path = f"./saved_expert_transition/{env_id}/{data_name}.pkl"
with open(path, "rb") as handle:
    data = pickle.load(handle)

# print(data["next_states"].shape)
# print(data["next_states"].strides)
# 1000,20,17
ns = data["next_states"]
explore_state = 20
# for _ in range(explore_state-1 ):
#     ns = np.concatenate(
#         (
#             ns,
#             ns[-1:],
#         ),
#         axis=0,
#     )
ns_strides = ns.strides
# https://tinyurl.com/2zgc7mcb
tmp = np.lib.stride_tricks.as_strided(
    ns,
    shape=(1000-(explore_state-1), explore_state, ns.shape[-1]),
    strides=(ns_strides[0], ns_strides[0], ns_strides[1]),
)

print(tmp[-1,-1])
print(ns[-1])