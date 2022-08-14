import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

filepath = "./Documents/TRNETO/data/Pb_Pb_d0.7_w0.4.txt"
data = np.asarray(pd.read_table(filepath))
nch = np.asarray(data.iloc[:,3])
e2 = np.asarray(data.iloc[:,4])
I = nch > percentile(nch,95) # 0 to 5% centrality
e2 = e2(I) # subset of e2 data
e2 = e2/np.mean(e2)
y, x = np.histogram(e2,50,density=True)
x = x[:-1] + (x[1] - x[0])/2 # convert bin edges to centres
plt.plot(x,y)
