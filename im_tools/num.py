#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:38:13 2024

@author: julie
"""
import os
import pandas as pd
from fit import fit

def GetRunNumber (path):
    file = os.listdir(path)
    run = pd.DataFrame(file, columns=['FileName'])
    run['run'] = run['FileName'].str.split('_').str.get(1)
    run['filter'] = run['FileName'].str.split('_').str.get(3).str.split('.').str.get(0)
    return run


def DoFit (data, path, run):
    if len(data)==1:
        file = pd.read_hdf(path+data[0])
    elif len(data)==2:
        file = pd.concat((pd.read_hdf(path+data[0]),pd.read_hdf(path+data[1])))
    else : 
        return
    resu_fit = fit(file, run)
    return resu_fit, file

