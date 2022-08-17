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

import os
from os import path
import gdown

_DATA = "tmp/e2Dist/data/"
_RESULTS = "tmp/e2Dist/results/"

def _gdownload(share_link, filename):
    file_id = share_link.split('/')[-2]
    url = f"https://drive.google.com/uc?id={file_id}"
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        gdown.download(url, out_file, quiet="False")
    return out_file

share_link = "https://drive.google.com/file/d/1K8nlHIcM5NDmjbRXxv5BDCxg95YgKX8h/view?usp=sharing"
filename = "trento_Pb_Pb_4000000_-x_6.4_-n_13.94_-k_1.19_-p_0.007_-d_0.7_-w_0.4.csv"

dataString = _gdownload(share_link,filename)
data = pd.read_csv(dataString)
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

if not path.exists(_RESULTS):
    os.makedirs(_RESULTS)
figString = path.join(_RESULTS,"figure1.png")
if not path.isfile(figString):
    plt.savefig(figString,format='png')
plt.show()

