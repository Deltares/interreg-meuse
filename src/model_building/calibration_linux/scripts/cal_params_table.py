# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:58:27 2022

@author: bouaziz
"""

import numpy as np
import itertools
import pandas as pd

#%% NB: all parameters should either be given as floats or all as integers 
#value
ksathorfrac = np.array([50.,100.,150.,200.,250.,300.,500.,750.,1000.,1500.,2000.,3000.])

#value
maxleakage = np.array([0. , 0.2, 0.6])

#mult 
soilthickness = np.array([1. , 2. , 3. ])

#mult
rootingdepth = np.arange(0.5, 2., 0.5)

#mult 
floodplain_volume = np.arange(1., 3., 1)
# floodplain_volume = np.arange(1., 2., 1)

a = [ksathorfrac, maxleakage, soilthickness, rootingdepth, floodplain_volume]
result = list(itertools.product(*a))
print(len(result))

df = pd.DataFrame(columns=["ksathorfrac", "maxleakage", "soilthickness", "rootingdepth", "floodplain_volume"], index=np.arange(0,len(result)))

for i, pset in enumerate(result):
    print(pset)
    df.iloc[i]["ksathorfrac"] = pset[0]
    df.iloc[i]["maxleakage"] =   pset[1]
    df.iloc[i]["soilthickness"] = pset[2]
    df.iloc[i]["rootingdepth"] = pset[3]
    df.iloc[i]["floodplain_volume"] = pset[4]

#duplicate soilthickness to soilminthickness
df["soilminthickness"] = df["soilthickness"]

df.to_csv(r"d:\SPW\Plots\calibration\calibration_parameters.csv", index=False)


#%%
