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

file_path='e2Dist/test1.0.txt'
print(file_path.split('.')[-1])
data = np.loadtxt('e2Dist/test1.0.txt')
df = pd.DataFrame(data)
print(df)