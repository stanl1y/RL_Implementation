import numpy as np
a=np.array(range(10)).reshape((-1,2))
b=np.array(range(10,22)).reshape(-1,2)
# print(np.tile(b,(len(a),1)))
# print(np.repeat(a,len(b),axis=0))
tmp=np.concatenate((np.repeat(a,len(b),axis=0),np.tile(b,(len(a),1))),axis=1)
print(tmp.sum(axis=1,keepdims=True).reshape((len(a),len(b))).mean(axis=1))
# print(tmp.reshape(len(a),len(b),4))


# aa,bb=np.meshgrid(a,b)
# tmp=np.array([aa,bb])
# print(tmp.shape)
# print(tmp)
# print(tmp.transpose((2,1,0)).reshape((-1,2)))

