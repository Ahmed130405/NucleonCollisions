from iminuit import cost, Minuit
import numpy as np
import math as m
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.special import iv
import yaml

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

def en_dist(file_path,n,cent): # for example, cent = [0, 5]
    filetype=file_path.split('.')[-1]
    if filetype=='csv':
        data = pd.read_csv(file_path,header=None)
    if filetype=='txt':
        data = pd.DataFrame(np.loadtxt(file_path))
    nch = np.asarray(data.iloc[:,3])
    en = np.asarray(data.iloc[:,n+2])
    if cent[0]==0:
        I = nch>np.percentile(nch,100-cent[1])
    else:
        I = (nch>np.percentile(nch,100-cent[1]))&(nch<np.percentile(nch,100-cent[0]))
    en = en[I] # subset of e2 data
    en = en/np.mean(en)
    n, x = np.histogram(en,50,density=False)
    integral = (x[1]-x[0])*sum(n)
    y, yerr = (n,n**0.5)/integral
    x = x[:-1] + (x[1] - x[0])/2 # convert bin edges to centres
    x, y, yerr = x[y>0], y[y>0], yerr[y>0] # ignore empty bins
    return x, y, yerr

def BG(x,a,b):
    return (x/(b**2))*iv(0,a*x/(b**2))*np.exp(-((x**2+a**2)/(2*(b**2))))

d=[0.7+0.1*i for i in range(6)]
w=[0.4+0.2*i for i in range(5)]
ww, dd = np.meshgrid(np.asarray(w),np.asarray(d))
ww, dd = ww.flatten(), dd.flatten()

sharelinks_link='https://drive.google.com/file/d/1UKVokgsHcpxmi5DDXd9rnfbXtdJlGPJW/view?usp=sharing'
filename='sharelinks1.txt'

sharelinks=pd.read_table(_gdownload(sharelinks_link,filename),header=None)
# print(sharelinks)
M=[]
z=[]
for i in range(6):
    for j in range(5):
        share_link = sharelinks.iloc[5*i+j,0]
        if (i==0)&(j==0):
            filename = 'trento_Pb_Pb_4000000_-x_6.4_-n_13.94_-k 1.19_-p_0.007_-d_0.7_-w_0.4.csv'
        else:
            filename = f'trento_Pb_Pb_4000000_-x_6.4_-n_13.94_-k 1.19_-p_0.007_-d_{round(0.7+0.1*i,1)}_-w_{round(0.4+0.2*j,1)}.txt'
        c = cost.LeastSquares(*en_dist(_gdownload(share_link,filename),2,[0,1]),BG)
        M.append(Minuit(c,a=0.5,b=0.5))
        M[-1].limits[0] = (0,None)
        M[-1].limits[1] = (0,None)
        M[-1].migrad()
        z.append(M[-1].values[1])
print(z)

plt.figure(6)
fig, ax = plt.subplots()
plt.subplots_adjust(hspace=.0,left=0.12,right=0.85,top=0.98)

pc = ax.scatter(dd,ww,c=z,cmap='RdBu_r')
fig.colorbar(pc,ax=ax,extend='both')
plt.show()