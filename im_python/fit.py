#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:58:39 2024

@author: julie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress



def fit_lin (var,mean) :
    mean = mean.to_numpy()
    var = var.to_numpy()
    diff_var_=[]
    mean_=[]   
    mean_max = 5000 #max value for fit (we want to make the fit for low illuminations)
    for i in range (len(mean)):
        if  mean[i] < mean_max :
            diff_var_.append(var[i])
            mean_.append(mean[i])
    a, b, r, p_value, std_err = linregress(mean_, diff_var_)
    return a,b


def quadra (x,a,b,c):
    return a*x**2+b*x+c

def fit_quadra (var,mean,err) :
    
    mean = mean.to_numpy()
    var = var.to_numpy()
    #max value for fit (we want to make the fit for low illuminations)
    mean_max = 50000
    x_data=[]
    y_data=[]
    err_data=[]
    for i in range (len(mean)):
        if  mean[i] < mean_max:
            y_data.append(var[i])
            x_data.append(mean[i])
            err_data.append(np.sqrt(err[i]))
            #err_data.append(0.02*var[i])
    y_params, cov, infodict,_,_ = curve_fit(quadra,x_data,y_data,sigma=err_data,full_output=True) #equation of order 2

    #chisq_dof = res / (len(x_data)-3)
    chisq_dof = np.sum((infodict['fvec'])**2)/(len(x_data)-3)
    
    return y_params, chisq_dof



def fit (tab, run) -> pd.DataFrame() : 
    """
    Make a quadratic fit of the data and calculate the turnoff ang gain
        - turnoff : mean illumination value where difference in illumnitation is maximum
        - gain : K = 1/b (b parameter of the fit)

    Parameters
    ----------
    tab : pd.dataframe
        Mean and variance for each raft/sensor/HDU
        
    Returns
    -------
    parameters : pd.dataframe
        output : 'raft','sensor','ampli','a','b','c','chisq_dof','gain','turnoff'
          (a,b,c are the parameters of a quadratic fit ax²+bx+c)                                    

    """
    c = []
    raft = tab['raft'].unique()
    sensor = tab['sensor'].unique()

    for r in raft:
        for s in sensor:
            for p in range(1,17):
                idx = tab[(tab['ampli']==p)&
                          (tab['raft']==r)&
                          (tab['sensor']==s)].index
                if not list(idx):
                    continue
                else:
                    mean = tab['mean'][idx].reset_index(drop=True)
                    var = tab['var'][idx].reset_index(drop=True)
                    err = tab['var(var)'][idx].reset_index(drop=True)
                    a,b = fit_lin(var, mean)
                    y_params, chisq_dof = fit_quadra(var, mean, err)
                 
                    max_var = np.max(var)
                
                    
                    turnoff_idx = var[var==max_var].index
                    turnoff = mean[turnoff_idx[0]]
                
                    c.append((run,r,s,p,
                              y_params[0],y_params[1],y_params[2],chisq_dof,1/y_params[1],turnoff))
            
    parameters = pd.DataFrame(c ,columns=['run','raft','sensor','ampli',
                                          'a','b','c','chisq_dof','gain','turnoff'])                                  
    return parameters


def data_fit (data):

    return 