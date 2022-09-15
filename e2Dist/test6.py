import numpy as np

li=[[0,1],[1,2],[2,3]]
print(li)
print(np.asarray(li))

d=[0.7+0.1*i for i in range(6)]
w=[0.4+0.2*i for i in range(5)]
ww, dd = np.meshgrid(np.asarray(w),np.asarray(d))
print(dd.flatten())
