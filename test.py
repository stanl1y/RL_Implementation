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
window_size = 100
l2 = np.convolve(l2, np.ones((window_size,))/window_size, mode='valid')
gail = np.convolve(gail, np.ones((window_size,))/window_size, mode='valid')
tdil = np.convolve(tdil, np.ones((window_size,))/window_size, mode='valid')
drop_window=20
l2=l2[::drop_window]
gail=gail[::drop_window]
tdil=tdil[::drop_window]
fig = plt.figure(figsize=(8, 1.8))  # Adjust the figsize as per your requirement
ax = plt.subplot(111)
lw=5
fs=15
ax.plot(x[:len(l2)]*drop_window, l2, label='L2',linewidth=lw)
ax.plot(x[:len(gail)]*drop_window, gail, label='IRL',linewidth=lw)
ax.plot(x[:len(tdil)]*drop_window, tdil, label='TDIL',linewidth=lw)
ax.axhline(23, color='r', linestyle='dotted',label='Optimal Solution',linewidth=lw)
ax.axhline(50, color='gray', linestyle='dotted',label='BC',linewidth=lw)
plt.xlabel('Total Steps',fontsize=fs)
plt.ylabel('Step Per Episode',fontsize=fs)
plt.gca().set_ylim(bottom=22)
# plt.legend(loc="lower left", ncol=4,fontsize='large')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width*1.2, box.height * 1.3])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5,fontsize=fs)
# save plot to file
plt.savefig('toy_all.pdf', dpi=900, bbox_inches='tight')
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
