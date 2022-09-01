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

share_link = "https://drive.google.com/file/d/1Q9FNc4iGIYAyqi6IOnR9an-YtL3LJaaM/view?usp=sharing"
filename = "HEPData-ins1233359-v1-Table_157.yaml"

def yaml_to_df(file_path):
    with open(file_path, "r") as f:
        contents = yaml.safe_load(f)
    v_n = [contents['independent_variables'][0]['values'][i]['value'] for i in range(len(contents['independent_variables'][0]['values']))]
    prob = [contents['dependent_variables'][0]['values'][i]['value'] for i in range(len(contents['dependent_variables'][0]['values']))]
    error_stat = [contents['dependent_variables'][0]['values'][i]['errors'][0]['symerror'] for i in range(len(contents['dependent_variables'][0]['values']))]
    d = {'v_n':v_n,'prob':prob,'error_stat':error_stat}
    df = pd.DataFrame(d)
    return df

print(yaml_to_df(_gdownload(share_link,filename)).iloc[:,0])