# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:58:27 2022

@author: bouaziz
"""

import numpy as np
import itertools
import os
import pandas as pd

#%% NB: all parameters should either be given as floats or all as integers 
#value
# ksathorfrac = np.array([50.,100.,150.,200.,250.,300.,500.,750.,1000.,1500.,2000.,3000.])
#mult
ksathorfrac = np.array([0.3, 0.5, 0.7, 1, 1.5, 2, 3])

#value
maxleakage = np.array([0. , 0.2, 0.6])

#mult 
soilthickness = np.array([0.5, 1. , 2. ])

#mult
rootingdepth = np.array([0.8, 1. , 1.2 ])

#mult 
# floodplain_n = np.array([0.072*0.5, 0.072, 0.072*2])
# floodplain_volume = np.arange(1., 2., 1)

#offset
storage_wood = np.array([0, 2])

#manning n river  mult
n = np.array([0.7, 1, 1.5])

#manning n  mult
# n = np.array([0.7, 1, 1.5])

a = [ksathorfrac, maxleakage, soilthickness, rootingdepth, storage_wood, n]
result = list(itertools.product(*a))
print(len(result))

df = pd.DataFrame(columns=["ksathorfrac", "maxleakage", "soilthickness", "rootingdepth", "storage_wood", "n"], index=np.arange(0,len(result)))

for i, pset in enumerate(result):
    print(pset)
    df.iloc[i]["ksathorfrac"] = pset[0]
    df.iloc[i]["maxleakage"] =   pset[1]
    df.iloc[i]["soilthickness"] = pset[2]
    df.iloc[i]["rootingdepth"] = pset[3]
    df.iloc[i]["storage_wood"] = pset[4]
    df.iloc[i]["n"] = pset[5]

#duplicate all n params
# df["n"] = df["nriver"]
# df["floodplainN"] = df["nriver"]

#duplicate soilthickness to soilminthickness
df["soilminthickness"] = df["soilthickness"]

df.to_csv(r"..\config\calibration_parameters.csv", index=False)


#%%
