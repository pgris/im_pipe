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
from scipy import interpolate
from scipy.optimize import minimize_scalar
import scipy.interpolate as spi
import itertools


def typeCCD (data):
    ITL = ['R00', 'R01', 'R02', 'R03', 'R04', 'R10',
           'R20', 'R40', 'R41', 'R42', 'R43', 'R44']
    e2v = ['R11', 'R12', 'R13', 'R14', 'R21', 'R22', 'R23', 
           'R24', 'R30', 'R31', 'R32', 'R33', 'R34']
    
    data['type'] = 'ITL' if data['raft'].all() in ITL else 'e2v'

    return data
    

def quadra (x,a,b,c):
    return a*x**2+b*x+c
def linear (x,a,b):
    return a*x+b

def fit_quadra (var, mean, p0) :
    """
    Make a quadratic fit (ax^2 + bx + c)

    Parameters
    ----------
    var : pd.Dataframe
        Variance of (flat0-flat1) of one raft/sensor/ampli
    mean : pd.Dataframe
        Mean value of (flat0+flat1) of one raft/sensor/ampli
    p0 : float
        Initial value estimated by a linear fit

    Returns
    -------
    y_params : list
        parameters of the fit (a,b,c)
    chisq_dof : value
        chi square

    """
    mean = mean.to_numpy()
    var = var.to_numpy()
    #max value for the fit (we want to make the fit for low illuminations)
    mean_max = 50000
    x_data=[]
    y_data=[]
    err_data=[]
    for i in range (len(mean)):
        if  mean[i] < mean_max:
            y_data.append(var[i])
            x_data.append(mean[i])
            
            err_data.append(0.02*var[i])
    y_params, cov, infodict,_,_ = curve_fit(quadra,x_data,y_data,
                                            sigma=err_data,full_output=True)

    chisq_dof = np.sum((infodict['fvec'])**2)/(len(x_data)-3)
    
    return y_params, chisq_dof


def fit (tab, run) -> pd.DataFrame() : 
    """
    Make a quadratic fit of the data and calculate the turnoff ang gain
        - turnoff : mean illumination value where difference in illumnitation is maximum
        - gain : K = 1/a (a parameter of the linear fit (ax+b))
        - gain quadratic : K = 1/b (b parameter of the quadratic fit (ax²+bx+c))

    Parameters
    ----------
    tab : pd.dataframe
        Mean and variance for each raft/sensor/HDU
        
    Returns
    -------
    parameters : pd.dataframe
        output : ('run','raft','type','sensor','ampli',
                  'param_quadra', 'gain_quadra',
                  'param_lin', 'gain_lin',
                  'turnoff')
    df : pd.DataFrame
        Reduced data
    dd : pd.DataFrame 
        Data used for the linear fit
    """
    parameters = pd.DataFrame()
    col = ['run','raft','type','sensor','ampli',
           'param_quadra', 'gain_quadra',
           'param_lin', 'gain_lin',
           'turnoff']
    raft = tab['raft'].unique()
    sensor = tab['sensor'].unique()
    # raft = ['R02']
    # sensor = ['S01']
    datafit = pd.DataFrame()
    tab = tab.reset_index(drop=True)
    typeCCD(tab)
    for r, s, p in itertools.product(raft, sensor, range(1, 17)):
        idx = tab[(tab['ampli']==p)&
                  (tab['raft']==r)&
                  (tab['sensor']==s)].index
        if not list(idx):
            print('pas de ', r, s, p)
            continue
        else:
            print(r, s, p)
            #Select the data for each raft/sensor/ampli
            mean = tab['mean'][idx].reset_index(drop=True)
            var = tab['var'][idx].reset_index(drop=True)
          
            bdata = pd.DataFrame({'mean':mean, 'var':var})
            
            ndata, y_param_lin = fit_lin_pr(bdata)
            nd = add_col(ndata, r, s, p)
            datafit = pd.concat((datafit, nd))
           
            if str(y_param_lin) == 'impossible':
                Lgain = 'undetermined'
                tt = 'undetermined'
            else:
                Lgain = 1/y_param_lin[0]
                tt = turnoff (bdata, Lgain)
                if Lgain > 3 or Lgain < 1:
                    Lgain = 'undetermined'
                
            y_param_qua, chisq_dof = fit_quadra(var, mean, y_param_lin[0])
            Qgain = 1/y_param_qua[1]

            ccd = 1

            
            c = (run,r,ccd,s,p, y_param_qua, Qgain, y_param_lin, Lgain, tt)
            
            
            df = pd.DataFrame([c], columns= col)
                   
            parameters = pd.concat([parameters, df]).reset_index(drop=True)  
    #residue_linear(bdata, parameters)
    return parameters, datafit



def add_col (df,r,s,p):
    """
    Add columns to a DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to which columns will be added
    r : int
        Value for the 'raft' column
    s : int
        Value for the 'sensor' column
    p : int
        Value for the 'ampli' column
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with the added columns
    """
    df['raft'] = r
    df['sensor'] = s
    df['ampli'] = p
    return df

def residue_linear (data, param) -> pd.DataFrame():
    """
    

    Parameters
    ----------
    data : pd.dataframe
        Reduce data of the PTC
    Param : pd.dataframe
        parameter of the fit 
          
    Returns
    -------
    parameters : pd.dataframe
        output : 'run','raft','type','sensor','ampli','a','b','c','chisq_dof','gain','turnoff'
          (a,b,c are the parameters of a quadratic fit ax²+bx+c)                                    

    """
    x_data = data['mean']
    y_data = data['var']
    
    a = param['param_lin'].values[0][0]
    b = param['param_lin'].values[0][1]
    y_fit_lin = a*x_data+b
    
    res =  y_fit_lin - y_data 
    plt.plot(x_data, res/y_data, '.')

    plt.ylabel('(var(flat0-flat1) - y_fit)/var')
    plt.xlabel('mean')
    plt.title()
    plt.legend()
    return 

def turnoff (data, gain) -> pd.DataFrame() :
    """
    Determine the turnoff from the PTC.
    The turnoff will depend on the gain, defining 2 ranges for the turnoff 
    depending on the gain.
    
    Parameters
    ----------
    data : pd.DataFrame
        Reduced data of the PTC.
    gain : float
        Gain of the raft/sensor/ampli.
    
    Returns
    -------
    turnoff : float
        Turnoff value.
    """
    #Define the range of the turnoff value
    lim_gain = 1.57
    if gain < lim_gain:
        inf = 60000
        sup = 100000
    else:
        inf = 60000
        sup = 90000
        
    idx = data[(data['mean'] >=inf) & (data['mean']<=sup)].index
    if not list(idx):
        print('impossible de déterminer le turnoff')   
        tt = 'undetermined'
    else : 
        xx = data['mean'][idx]
        yy = data['var'][idx]
        
        
        f = spi.interp1d(xx,yy,fill_value="extrapolate")
        x = np.linspace(min(xx), max(xx))
        y = f(x)
    
        vmax_idx = np.argmax(y)
        tt = x[vmax_idx]
        # if tt < 60000:
        #     tt = 'undetermined'
        
    return tt
 
       

def fit_lin_pr (data):
    """
    Do a linear fit for low illuminations

    Parameters
    ----------
    data : pd.Dataframe
        Reduced data (mean, var)

    Returns
    -------
    fdata : pd.Dataframe
        Data used for the fit (mean, var)
    popt : np.array
        Parameters of the fit (a, b)

    """
    dd_new = data[data['mean']<2000].reset_index(drop=True).copy()
    sigma = 100
    n = len(dd_new['mean'])    
    #print('début = ',n)
    if len(dd_new['var'])<10:
        popt = 'impossible'
    else:
        for j in range (2,n-1):
           
            popt, pcov = curve_fit(linear, dd_new['mean'][0:j], dd_new['var'][0:j]);
            
            #Check if the next value is closed enough to the linear fit
            if abs(linear(dd_new['mean'][j+1], *popt)-dd_new['mean'][j+1]) > 50*sigma :
                    if abs(linear(dd_new['mean'][j+2], *popt)-dd_new['mean'][j+2]) < 50*sigma:
                        sigma = np.sqrt(np.diag(pcov))[0]
                        datafit = dd_new.drop(index = j+1).copy()  #delete value j+1 if >5*sigma and keep j+2 if <50*sigma
                    else:
                        break
            else:
                sigma = np.sqrt(np.diag(pcov))[0]
                
        
        popt, pcov = curve_fit(linear, dd_new['mean'], dd_new['var']);
        #print('fin = ', len(data))
    
    return dd_new, popt
 



































