import numpy as np
import time
a=np.arange(10000)
a=a.reshape((1000,10))
t=time.time()
np.tile(a,(5,1))
print(time.time()-t)
t=time.time()
np.broadcast_to(a,(5,1000,10)).reshape((-1,10))
print(time.time()-t)