import imageio
import argparse
import pathlib
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import cv2
parser = argparse.ArgumentParser()
parser.add_argument("-e","--env", type=str, default="", help="environment")
parser.add_argument("-m","--margin", type=str, default="", help="which exp to make gif")
args = parser.parse_args()
images = []
images_path = []
exp_name=f"{args.env}_margin{args.margin}"
data_root = pathlib.Path(f"experiment_logs/{exp_name}/")
img_num=0
max_reward=0.005050505050505



# if not data_root.joinpath("reward.png").exists():
plt.figure(figsize=(9, 2), dpi=70)
reward_df=pd.read_csv(f"{str(data_root)}/{exp_name}.csv")
rewards=np.array(reward_df[f"{args.env} - testing_reward"])
plt.plot(np.array(range(len(rewards))), rewards)
plt.savefig(str(data_root.joinpath("reward.png")))
reward_img=imageio.imread(str(data_root.joinpath("reward.png")))
reward_img=reward_img[20:,100:-84]

for filename in data_root.joinpath("color").iterdir():
    if not str(filename)[-3:]=="png":
        continue
    img_num+=1

color_images=[]
path_images=[]
arr_images=[]
testing_reward=[]
for idx,filename in enumerate(data_root.joinpath("color").iterdir()):
    if not str(filename)[-3:]=="png":
        continue
    img=imageio.imread(filename)
    color_images.append(img)

for idx,filename in enumerate(data_root.joinpath("path").iterdir()):
    if not str(filename)[-3:]=="png":
        continue
    img=imageio.imread(filename)
    path_images.append(img)

for idx,filename in enumerate(data_root.joinpath("arr").iterdir()):
    if not str(filename)[-3:]=="png":
        continue
    img=imageio.imread(filename)
    img=img[58:-52,144:-126,:]
    arr_images.append(img)

for idx,filename in enumerate(data_root.joinpath("testing_reward").iterdir()):
    if not str(filename)[-3:]=="pkl":
        continue
    with open(filename, "rb") as fp:
        r=pickle.load(fp)
    testing_reward.append(r)


for idx,data in enumerate(zip(color_images,path_images,arr_images)):
    color_img,path_img,arr_img=data
    # reward=reward_df.loc[idx][f"{args.exp_name} - Reward"]
    reward=testing_reward[idx]
    extend_img=np.zeros((arr_img.shape[0]+130,arr_img.shape[1]+color_img.shape[1]+10,arr_img.shape[2])).astype(np.uint8)
    extend_img[:color_img.shape[0],:color_img.shape[1]]=color_img
    extend_img[color_img.shape[0]+50:color_img.shape[0]+50+path_img.shape[0],:path_img.shape[1]]=path_img
    extend_img[:arr_img.shape[0],color_img.shape[1]+10:color_img.shape[1]+10+arr_img.shape[1]]=arr_img
    # extend_img[-19:,:]=np.array([255,255,255,255])
    extend_img[-128:-122,:int(reward/max_reward*extend_img.shape[1])]=np.array([255,0,0,255])
    # extend_img[-8:,:int(idx/img_num*extend_img.shape[1])]=np.array([0,0,0,255])
    reward_img_progress=reward_img.copy()
    # mask=reward_img_progress[:,int(idx/img_num*reward_img_progress.shape[1])].mean(axis=1)>50
    reward_img_progress[:,int(idx/img_num*reward_img_progress.shape[1])]=0
    reward_img_progress[reward_img_progress<0]=0
    extend_img[-120:,15:15+reward_img_progress.shape[1]]=reward_img_progress

    images.append(extend_img)
    # images_path.append(filename)
# images_path.sort()

gif_idx=0
while pathlib.Path(f"{str(data_root)}/{gif_idx}.gif").exists():
    gif_idx+=1
# video = cv2.VideoWriter(f"{str(data_root)}/{gif_idx}.avi", 0, 1, (images[0].shape[0],images[0].shape[1]))
# for image in images:
#     video.write(image)

# cv2.destroyAllWindows()
# video.release()

imageio.mimsave(f"{str(data_root)}/{gif_idx}.gif", images, duration=0.15)

