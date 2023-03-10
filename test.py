# import numpy as np
# import time
# all_update_time = 0.
# couter=0
# with open("exec_time_log_cpuTensor.txt", "r") as f:
#     for(line) in f:
#         if ("update bc time" in line):
#             couter+=1
#             all_update_time += float(line.split(" ")[-1])
# print(all_update_time/couter)

import torch
import time
a=torch.randn(100,100)
b=torch.randn(100,100)

t=time.time()
c=torch.cat((a,b),dim=0)
c=c.cuda()
print(time.time()-t)

d=torch.randn(100,100)
e=torch.randn(100,100)

t=time.time()
d=d.cuda()
e=e.cuda()
f=torch.cat((d,e),dim=0)
print(time.time()-t)
t=time.time()
c=torch.cat((a,b),dim=0)
c=c.cuda()
print(time.time()-t)