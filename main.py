import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['font.size'] = 14
rcParams['text.color'] = 'black'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
rcParams['pdf.fonttype'] = 42

file = "./Documents/TRENTO/data/Pb_Pb_d0.7_w0.4.csv"
data = pd.read_csv(file)
nch = np.asarray(data.iloc[:,3])
e2 = np.asarray(data.iloc[:,4])
I = nch > np.percentile(nch,95) # 0 to 5% centrality
e2 = e2[I] # subset of e2 data
e2 = e2/np.mean(e2)
y, x = np.histogram(e2,50,density=True)
x = x[:-1] + (x[1] - x[0])/2 # convert bin edges to centres

fig, ax = plt.subplots()
ax.plot(x,y,color='blue',linestyle='dashed',linewidth=2)
ax.set_xlabel('Normalized e2')
ax.set_title('Distribution for 0 to 5% centrality')
plt.show()