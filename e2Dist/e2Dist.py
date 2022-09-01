from iminuit import cost, Minuit
import numpy as np
import math as m
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.special import iv
import yaml

from matplotlib import rcParams

rcParams['font.size'] = 14
rcParams['text.color'] = 'black'
# rcParams['text.usetex'] = True
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

def en_dist(file_path,n,cent): # for example, cent = [0, 5]
    data = pd.read_csv(file_path)
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
    return x, y, yerr

def vn_dist(file_path):
    with open(file_path, "r") as f:
        contents = yaml.safe_load(f)
    vn = np.asarray([contents['independent_variables'][0]['values'][i]['value'] for i in range(len(contents['independent_variables'][0]['values']))])
    prob = np.asarray([contents['dependent_variables'][0]['values'][i]['value'] for i in range(len(contents['dependent_variables'][0]['values']))])
    stat = np.asarray([contents['dependent_variables'][0]['values'][i]['errors'][0]['symerror'] for i in range(len(contents['dependent_variables'][0]['values']))])
    x, y, yerr = vn/np.mean(vn), prob*np.mean(vn), stat*np.mean(vn)
    return x, y, yerr

def BG(x,a,b):
    return (x/(b**2))*iv(0,a*x/(b**2))*np.exp(-((x**2+a**2)/(2*(b**2))))

share_link = "https://drive.google.com/file/d/1K8nlHIcM5NDmjbRXxv5BDCxg95YgKX8h/view?usp=sharing"
filename = "trento_Pb_Pb_4000000_-x_6.4_-n_13.94_-k_1.19_-p_0.007_-d_0.7_-w_0.4.csv"

c = cost.LeastSquares(*en_dist(_gdownload(share_link,filename),2,[0,5]),BG)
m1 = Minuit(c,a=0,b=0.5)
m1.limits[1] = (0,None)
m1.migrad()

plt.figure(1)
fig, ax = plt.subplots()
plt.subplots_adjust(hspace=.0,left=0.12,right=0.85,top=0.98)

ax.tick_params(which='major',bottom=True, top=True, left=True, right=True,direction='in',length=6)
ax.tick_params(which='minor',bottom=True, top=True, left=True, right=True,direction='in',length=3)
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

ax.errorbar(c.x,c.y,c.yerror,fmt='ok',mfc='blue',mec='blue',ecolor='blue',elinewidth=1,label='data')
ax.plot(c.x,BG(c.x,*m1.values),color='green',linestyle='dashed',linewidth=2,label='fit')
ax.legend(loc='upper right',frameon=False,fontsize=11)

ax.text(0.16,0.85,fr'$\langle\epsilon_{{RP}}\rangle$ = {round(m1.values[0],4)} $\pm$ {round(m1.errors[0],4)}',fontsize=11)
ax.text(0.16,0.81,fr'$\sigma_{{\epsilon}}$ = {round(m1.values[1],4)} $\pm$ {round(m1.errors[1],4)}',fontsize=11)
ax.text(0.16,0.89,f'chi2/ndof = {round(m1.fmin.reduced_chi2,4)}',fontsize=11)

ax.set_ylim([0,0.95])
ax.set_xlim([0.0,4.15])
ax.set_xlabel(r'$\epsilon_{2}$')
ax.set_ylabel(r'p($\epsilon_{2}$)')
ax.text(2.85,0.09,'centrality: 0-5%',fontsize=11)

if not path.exists(_RESULTS):
    os.makedirs(_RESULTS)
figString = path.join(_RESULTS,"figure1.pdf")
if not path.isfile(figString):
    plt.savefig(figString,format='pdf')
plt.show()

# Atlas, centrality: 0-1%
share_link = "https://drive.google.com/file/d/1Q9FNc4iGIYAyqi6IOnR9an-YtL3LJaaM/view?usp=sharing"
filename = "HEPData-ins1233359-v1-Table_157.yaml"

c = cost.LeastSquares(*vn_dist(_gdownload(share_link,filename)),BG)
m2 = Minuit(c,a=0,b=0.5)
m2.limits[1] = (0,None)
m2.migrad()

plt.figure(2)
fig, ax = plt.subplots()
plt.subplots_adjust(hspace=.0,left=0.12,right=0.85,top=0.98)

ax.tick_params(which='major',bottom=True, top=True, left=True, right=True,direction='in',length=6)
ax.tick_params(which='minor',bottom=True, top=True, left=True, right=True,direction='in',length=3)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))

ax.errorbar(c.x,c.y,c.yerror,fmt='ok',mfc='blue',mec='blue',ecolor='blue',elinewidth=1,label='data')
ax.plot(c.x,BG(c.x,*m2.values),color='green',linestyle='dashed',linewidth=2,label='fit')
ax.legend(loc='upper right',frameon=False,fontsize=11)

ax.text(0.09,1.215,fr'$\langle\epsilon_{{RP}}\rangle$ = {round(m2.values[0],4)} $\pm$ {round(m2.errors[0],4)}',fontsize=11)
ax.text(0.09,1.16,fr'$\sigma_{{\epsilon}}$ = {round(m2.values[1],4)} $\pm$ {round(m2.errors[1],4)}',fontsize=11)
ax.text(0.09,1.27,f'chi2/ndof = {round(m2.fmin.reduced_chi2,4)}',fontsize=11)

ax.set_ylim([0,1.35])
ax.set_xlim([0.0,2.25])
ax.set_xlabel(r'$v_{2}$')
ax.set_ylabel(r'p($v_{2}$)')
ax.text(0.21,0.09,'centrality: 0-1%',fontsize=11)

if not path.exists(_RESULTS):
    os.makedirs(_RESULTS)
figString = path.join(_RESULTS,"figure2.pdf")
if not path.isfile(figString):
    plt.savefig(figString,format='pdf')
plt.show()

# Atlas, centrality: 5-10%
share_link = "https://drive.google.com/file/d/1digscKRKrCUBk_8oU1KSvqi6PK0c744t/view?usp=sharing"
filename = "HEPData-ins1233359-v1-Table_61.yaml"

c = cost.LeastSquares(*vn_dist(_gdownload(share_link,filename)),BG)
m3 = Minuit(c,a=0,b=0.5)
m3.limits[1] = (0,None)
m3.migrad()

plt.figure(3)
fig, ax = plt.subplots()
plt.subplots_adjust(hspace=.0,left=0.12,right=0.85,top=0.98)

ax.tick_params(which='major',bottom=True, top=True, left=True, right=True,direction='in',length=6)
ax.tick_params(which='minor',bottom=True, top=True, left=True, right=True,direction='in',length=3)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))

ax.errorbar(c.x,c.y,c.yerror,fmt='ok',mfc='blue',mec='blue',ecolor='blue',elinewidth=1,label='data')
ax.plot(c.x,BG(c.x,*m3.values),color='green',linestyle='dashed',linewidth=2,label='fit')
ax.legend(loc='upper right',frameon=False,fontsize=11)

ax.text(0.09,1.29,fr'$\langle\epsilon_{{RP}}\rangle$ = {round(m3.values[0],4)} $\pm$ {round(m3.errors[0],4)}',fontsize=11)
ax.text(0.09,1.23,fr'$\sigma_{{\epsilon}}$ = {round(m3.values[1],4)} $\pm$ {round(m3.errors[1],4)}',fontsize=11)
ax.text(0.09,1.35,f'chi2/ndof = {round(m3.fmin.reduced_chi2,4)}',fontsize=11)

ax.set_ylim([0,1.45])
ax.set_xlim([0.0,2.15])
ax.set_xlabel(r'$v_{2}$')
ax.set_ylabel(r'p($v_{2}$)')
ax.text(0.21,0.09,'centrality: 5-10%',fontsize=11)

if not path.exists(_RESULTS):
    os.makedirs(_RESULTS)
figString = path.join(_RESULTS,"figure3.pdf")
if not path.isfile(figString):
    plt.savefig(figString,format='pdf')
plt.show()