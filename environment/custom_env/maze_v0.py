import gym
from gym import spaces
import numpy as np
class Maze_v0(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_size=100, random_reset=False):
        super(Maze_v0, self).__init__()
        
        self.maze_size = maze_size
        self.state = np.asarray((self.maze_size-1,0))
        self.bomb = np.zeros((self.maze_size,self.maze_size))
        self.bomb
        self.random_reset=random_reset
        self.agent_step=0


        # left, up, right, down
        self.ACTIONS = [np.array([-1, 0]),
                        np.array([0 , 1]),
                        np.array([1 , 0]),
                        np.array([0 , -1])]

        self.action_space = spaces.Discrete(4)    
        self.observation_space = spaces.Box(low=0, high=maze_size, shape=(2,), dtype=np.uint8)

    def is_terminal(self, state):
        x, y = state
        return (x == 0 and y == self.maze_size - 1)

    def is_small_reward(self, state):
        x, y = state
        return (x >= 3 and x <= self.maze_size-4 and y >= 3 and y <= self.maze_size-4)

    def expert_step(self):
        x,y=self.state
        if x > self.maze_size - 1 -y:
            action=0
        else:
            action=1
        return action

    def step(self, action):
        self.agent_step+=1
        next_state = (np.array(self.state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if x < 0 or x >= self.maze_size or y < 0 or y >= self.maze_size:
            next_state = self.state
        
        self.state = np.array(next_state)
        
        # if self.is_small_reward(self.state):
        #     return self.state, 0.01, True, {}
        
        if self.is_terminal(self.state):
            return self.state, 1.0, True, {"x":self.state[0],"y":self.state[1]}  
        
        reward = 0.0
        if self.agent_step==self.maze_size*5:
            return self.state, reward, True, {"x":self.state[0],"y":self.state[1]}
        return self.state, reward, False, {"x":self.state[0],"y":self.state[1]}
        
    def reset(self):
        if self.random_reset:
            self.state = np.asarray((np.random.randint(self.maze_size//2,self.maze_size),np.random.randint(self.maze_size//2)))
        else:
            self.state = np.asarray((self.maze_size-1,0))
        self.agent_step=0
        return self.state