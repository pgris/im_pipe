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
def gauss (x, amp, mu, sigma):
    return amp*np.exp(-(x-mu)**2/(2*sigma**2))


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
    """
    df['raft'] = r
    df['sensor'] = s
    df['ampli'] = p
    return df

def residue_linear (data, param) -> pd.DataFrame():
    """
    Calculate the residuals of a linear fit

    Parameters
    ----------
    data : pd.dataframe
        Reduce data of the PTC
    Param : pd.dataframe
        Parameter of a linear fit 
          
    Returns
    -------
    parameters : pd.dataframe
        output : Add a column with the residuals 'res_lin' to the 'data'                              
    """
    x_data = data['mean']
    y_data = data['var']
    a = param[0]
    b = param[1]
    y_fit = linear(x_data, a, b)
    res =  y_fit - y_data 
    data['res_lin'] = res
    return data

def residue_quadra (data, param) -> pd.DataFrame():
    """
    Calculate the residuals of a quadratic fit
    
    Parameters
    ----------
    data : pd.dataframe
        Reduce data of the PTC
    Param : pd.dataframe
        Parameter of a quadratic fit 
          
    Returns
    -------
    parameters : pd.dataframe
        output : Add a column with the residuals 'res_quadra' to the 'data'                              
    """

    x_data = data['mean']
    y_data = data['var']
    a = param[0]
    b = param[1]
    c = param[2]
    y_fit = quadra(x_data, a, b, c)
    res =  y_fit - y_data 
    data['res_quadra'] = res
    return data

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
    if gain > lim_gain:
        inf = 60000
        sup = 110000
    else:
        inf = 50000
        sup = 100000
        
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
        if max(y)>55000:
            tt = 'outsider'
        else :
            vmax_idx = np.argmax(y)
            tt = x[vmax_idx]
        # if tt < 60000:
        #     tt = 'undetermined'
            # plt.figure()
            # plt.plot(xx,yy,'.')
            # plt.plot(x,y, '.r')
            # plt.plot(tt,max(y),'x')
            # plt.show()
    return tt

def fit_lin_pr(data):
    """
    Do a linear fit for low illuminations

    Parameters
    ----------
    data : pd.DataFrame
        Reduced data (mean, var)

    Returns
    -------
    fdata : pd.DataFrame
        Data used for the fit (mean, var)
    popt : np.array
        Parameters of the fit (a, b)
    """
    
    data = bin_data(data)
    popt = np.array([100, 100])
    sigma = 100

    # Filter the data
    # Range for linear fit
    dd = data[(data['mean'] < 1000) & (data['mean'] > 50)].reset_index(drop=True)
    if len(dd) < 10: #less than 10 values -> no need to adjust
        return data, popt, sigma

    # Initial fit
    popt, _ = curve_fit(linear, dd['mean'], dd['var'])
    dd = residue_linear(dd, popt)
    p = plothist(dd['res_lin'])
    
    if p[0] == 0:
        return dd, popt, sigma

    amp, mu, sigma = p
    cond = 3 * sigma
    # Keep the values : 5*sigma < mean < 5*sigma
    dd = dd[(dd['res_lin'] < mu + cond) & (dd['res_lin'] > mu - cond)].copy()

    # Iterative fitting
    for _ in range(6): #Maximum iteration of the fit = 3
        if len(dd) <= 10:
            break
        
        popt_, _ = curve_fit(linear, dd['mean'], dd['var'])
        new_dd = residue_linear(dd, popt_)
        p = plothist(new_dd['res_lin'])
        
        if p[0] == 0:
            break
        
        amp, mu, sigma = p
        cond = 3 * sigma
        new_dd = new_dd[(new_dd['res_lin'] < mu + cond) & (new_dd['res_lin'] > mu - cond)].copy()
        if len(new_dd) >= len(dd) or len(new_dd)<10:
            break
        
        else:
            dd = new_dd
            popt = popt_
            
    return dd, popt, sigma

def fit_quadra_pr (data):
    """
    Do a quadratic fit for low illuminations

    Parameters
    ----------t(lin
    data : pd.Dataframe
        Reduced data (mean, var)

    Returns
    -------
    fdata : pd.Dataframe
        Data used for the fit (mean, var)
    popt : np.array
        Parameters of the fit (a, b, c)
    """
    popt = np.array([100, 100, 100])
    sigma = 100
    data = bin_data(data)
    param = 'res_quadra'
    
    # Filter the data
    # Range for quadratic fit
    data = data[(data['mean']<10000)&(data['mean']>50)].reset_index(drop=True).copy()
    dd = data[(data['var']<30000)&(data['var']>0)].reset_index(drop=True).copy()

    if len(dd) < 10:
        return data, popt, sigma

    #Initial fit 
    popt, pcov = curve_fit(quadra, dd['mean'], dd['var']);
    dd = residue_quadra(dd, popt)
    p = plothist(dd[param])

    if p[0] == 0:
        return dd, popt, sigma

    amp, mu, sigma = p
    
    cond = 5 * sigma
    # Keep the values : 5*sigma < mean < 5*sigma
    dd = dd[(dd[param] < mu + cond) & (dd[param] > mu - cond)].copy() 

    # Iterative fitting
    for _ in range(5): #Maximum iteration of the fit = 3
        if len(dd) <= 10:
            break    
        popt_, _ = curve_fit(quadra, dd['mean'], dd['var'])
        new_dd = residue_quadra(dd, popt_)
        p = plothist(new_dd[param])
        if p[0] == 0:
            break  
        amp, mu, sigma = p
        cond = 3 * sigma
        new_dd = new_dd[(dd[param] < mu + cond) & (new_dd[param] > mu - cond)]  

        if len(new_dd) >= len(dd) | len(new_dd)<=10:
            break
        else:
            dd = new_dd
            popt = popt_

    return dd, popt, sigma



def plothist(data):

    m = np.median(data)
    std = np.std(data)

    n, bins = np.histogram(data, bins='auto')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    try: 
        popt, pcov = curve_fit(gauss, bin_centers, n, p0=[np.max(n), int(m), int(std)])
        return popt
    except RuntimeError:
        return [0]


def fit (tab) -> pd.DataFrame() : 
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
        output : ('raft','sensor','ampli', 'param_lin', 'param_quadra',
                  'gain_lin', 'gain_quadra', 'turnoff')
        df : pd.DataFrame
            Reduced data
        dd : pd.DataFrame 
            Data used for the linear fit
    """
    raft = tab['raft'].unique()
    sensor = tab['sensor'].unique()
    
    resu = pd.DataFrame()
    tab = tab.reset_index(drop=True)
    typeCCD(tab)
    datafit = pd.DataFrame()
    datafitqua = pd.DataFrame() 
    col = ['raft','sensor','ampli', 'param_lin', 'param_quadra',
           'gain_lin', 'gain_quadra', 'turnoff']
 
    ampli = range(0,17)
    for r, s, p in itertools.product(raft, sensor, ampli):
        idx = tab[(tab['ampli']==p)&
                  (tab['raft']==r)&
                  (tab['sensor']==s)].index
        if not list(idx):
            continue
        else:
            print(r, s, p)
            nd=0
            
            #Select the data for each raft/sensor/ampli
            mean = tab['mean'][idx].reset_index(drop=True)
            var = tab['var'][idx].reset_index(drop=True)
            bdata = pd.DataFrame({'mean':mean, 'var':var})
            
            #bin les données
            
            
            #linear fit
            ndata, y_param_lin, sigl = fit_lin_pr(bdata)
            Lgain =  1/y_param_lin[0]
            
            #quadratic fit
            qdata, y_param_qua, sigg = fit_quadra_pr(bdata)
            Qgain =  1/y_param_qua[1]
            
            #turnoff
            tt = turnoff(bdata, Lgain)
            
            #resultats des fits
            c = (r,s,p,y_param_lin,y_param_qua, Lgain, Qgain, tt)
            nd = pd.DataFrame([c], columns= col)
            resu = pd.concat((resu, nd))
            
            #data des fits
            ndata = add_col(ndata, r, s, p)
            qdata = add_col(qdata, r, s, p)
            datafitqua = pd.concat((datafitqua, qdata))
            datafit = pd.concat((ndata, datafit))

    return datafit, datafitqua, resu


def bin_data(f):
    f = f.sort_values(by='mean').reset_index(drop=True)
    n = len(f)
    dd = pd.DataFrame()
    for i in range(0, n, 10):
        r = i + 10
        if r > n:
            r = n
        d = f.iloc[i:r].copy()  
        m = np.median(d['var'])
        d.loc[:, 'div'] = d['var'] / m
        divi = d[d['div'] < 1.1]
        dd = pd.concat([dd, divi])
    d_m = dd.drop_duplicates(subset=['mean'])
    d_m = d_m.drop('div', axis=1)
    return d_m
















