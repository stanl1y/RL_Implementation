# # from environment.custom_env import Maze_v6
# # import numpy as np
# # import gym
# # from environment.wrapper import *
# # gym.envs.register(id="Maze-v6", entry_point=Maze_v6, max_episode_steps=400)
# # env = gym.make("Maze-v6")
# # env = NormObs(env)
# # np.savetxt('test.out', env.maze, fmt='%i', delimiter=',')
# # # expert_states=set(tuple(x) for x in env.expert_states)
# # # expert_states.remove((49,49))
# # # for item in env.expert_transitions:
# # #     print(item)
# # print(env.expert_state_action_set)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# read csv file and plot
df = pd.read_csv('all.csv')
x=df['eval_total_steps']

l2=df['l2']
gail=df['gail']
tdil=df['tdil']
#moving average window size 5
l2 = np.convolve(l2, np.ones((5,))/5, mode='valid')
gail = np.convolve(gail, np.ones((5,))/5, mode='valid')
tdil = np.convolve(tdil, np.ones((5,))/5, mode='valid')
fig = plt.figure()
ax = plt.subplot(111)
lw=5
ax.plot(x[:len(l2)], l2, label='L2',linewidth=lw)
ax.plot(x[:len(gail)], gail, label='IRL',linewidth=lw)
ax.plot(x[:len(tdil)], tdil, label='TDIL',linewidth=lw)
ax.axhline(23, color='r', linestyle='dotted',label='Optimal solution',linewidth=lw)
plt.xlabel('Total Steps',fontsize=20)
plt.ylabel('Step per episode',fontsize=20)
plt.gca().set_ylim(bottom=22)
# plt.legend(loc="lower left", ncol=4,fontsize='large')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*1.2, box.height * 1.3])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2,fontsize=20)
# save plot to file
plt.savefig('toy_all.png', dpi=900, bbox_inches='tight')
print(tdil)

# from PIL import Image, ImageDraw

# # Create a 500x500 image
# image_width = 1000
# image_height = 1000
# image = Image.new("RGB", (image_width, image_height), "white")
# draw = ImageDraw.Draw(image)

# # Define the number of rows and columns in the grid
# rows = 10
# columns = 10

# # Calculate the width and height of each grid cell
# cell_width = image_width // columns
# cell_height = image_height // rows

# # Draw horizontal grid lines
# for i in range(1, rows):
#     y = i * cell_height
#     draw.line([(0, y), (image_width, y)], fill="black")

# # Draw vertical grid lines
# for i in range(1, columns):
#     x = i * cell_width
#     draw.line([(x, 0), (x, image_height)], fill="black")

# # Save the image
# image.save("grid.png")
