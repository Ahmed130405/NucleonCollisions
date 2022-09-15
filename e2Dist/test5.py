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

d=[0.7+0.1*i for i in range(6)]
w=[0.4+0.2*i for i in range(5)]

for i in w:
    for j in d:
        print(f'trento_Pb_Pb_4000000_-x_6.4_-n_13.94_-k 1.19_-p_0.007_-d_{round(i,1)}_-w_{round(j,1)}.txt')