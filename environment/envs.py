import gym
from .wrapper import *
from .custom_env import *

          

def get_env(env_name,wrapper_type):
    if env_name in ["highway-v0","merge-v0","roundabout-v0","parking-v0","intersection-v0","racetrack-v0"]:
        '''
        highway_env:
        https://github.com/eleurent/highway-env
        '''
        import highway_env
    if env_name in ["Maze-v0", "Maze-v1", "Maze-v2"]:
        '''
        custom_env:
        '''
        gym.envs.register(id="Maze-v0", entry_point=Maze_v0, max_episode_steps=250)          
        gym.envs.register(id="Maze-v1", entry_point=Maze_v1, max_episode_steps=400)          
        gym.envs.register(id="Maze-v2", entry_point=Maze_v2, max_episode_steps=400)
        env=gym.make(env_name,**dict(random_reset=True,combine_s_g=True))
    else:
        env=gym.make(env_name)
    if wrapper_type=="basic":
        return BasicWrapper(env)
    elif wrapper_type=="gym_robotic":
        return GymRoboticWrapper(env)
    elif wrapper_type=="her":
        return HERWrapper(env)
    elif wrapper_type=="normobs":
        return NormObs(env)
    else:
        raise TypeError(f"env wrapper type : {wrapper_type} not supported")

