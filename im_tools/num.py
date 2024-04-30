#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:38:13 2024

@author: julie
"""
import os
import pandas as pd
from fit import  fit

def GetRunNumber (path) -> pd.DataFrame() :
    """
    Get the list of run number in 'ptc_resu' and the filter

    Parameters
    ----------
    path : str
        path to ../ptc_resu/ (result of the ptc)

    Returns
    -------
    run : pd.DataFrame
        list of all the run (file_name, run, filter)

    """
    file = os.listdir(path)  #list the file in the folder
    run = pd.DataFrame(file, columns=['FileName'])
    run['run'] = run['FileName'].str.split('_').str.get(1)
    run['filter'] = run['FileName'].str.split('_').str.get(3).str.split('.').str.get(0)
    return run


def DoFit (data, path, run) -> pd.DataFrame() :
    """
    Process the fit 

    Parameters
    ----------
    data : pd.Dataframe()
        Result of the PTC 
    path : str
        Path to acess the data 
    run : int
        Run nulber

    Returns
    -------
    resu_fit : pd.DataFrame
        Result of the fit ('run','raft','type','sensor','ampli',
                           'param_quadra', 'gain_quadra',
                           'param_lin', 'gain_lin',
                           'turnoff')
    new_data : pd.DataFrame()
        Reduced data
    new_data_lin : pd.DataFrame()
        Reduced data used to do the linear fit

    """
    #If there are several filters, group the data together

    if len(data)==1:
        file = pd.read_hdf(path+data[0])
    elif len(data)==2:
        file = pd.concat((pd.read_hdf(path+data[0]),pd.read_hdf(path+data[1])))
    else : 
        return
    resu_fit, new_data_lin  = fit(file, run)
    return resu_fit, new_data_lin, file






