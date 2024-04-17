#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:38:13 2024

@author: julie
"""
import os
import pandas as pd
from fit import bin_data, fit

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
    resu_fit, new_data, new_data_lin  = fit(file, run)
    return resu_fit, new_data, new_data_lin

def ReducedData (data, path, run) -> pd.DataFrame() :
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
    bdata = pd.DataFrame()
    #If there are several filters, group the data together
    if len(data)==1:
        file = pd.read_hdf(path+data[0])
    elif len(data)==2:
        file = pd.concat((pd.read_hdf(path+data[0]),pd.read_hdf(path+data[1])))
    else : 
        return
    
    raft = file['raft'].unique()
    sensor = file['sensor'].unique() 
    for r in raft:
        for s in sensor:
            for p in range(1,17):
                idx = file[(file['ampli']==p)&
                          (file['raft']==r)&
                          (file['sensor']==s)].index
                if not list(idx):
                    continue
                else:
                    print(r, s, p)
                    mean = file['mean'][idx].reset_index(drop=True)
                    var = file['var'][idx].reset_index(drop=True)
                    
                    new_data = bin_data(mean, var)
                    bdata = pd.concatenate((bdata, new_data))
                    
    return bin_data


