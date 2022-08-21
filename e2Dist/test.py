from iminuit import cost, Minuit
import numpy as np
import math as m
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import iv

def BG(x,a,b):
    return (x/(b**2))*iv(0,a*x/(b**2))*np.exp(-((x**2+a**2)/(2*(b**2))))
x=np.linspace(0,4,50)
a = 0.9
b = 0.4
plt.plot(x,BG(x,a,b))
plt.show()