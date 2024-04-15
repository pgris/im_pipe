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



def fit_lin (var,mean) :
    mean = mean.to_numpy()
    var = var.to_numpy()
    err_data=[]
    diff_var_=[]
    mean_=[]   
    mean_max = 5000 #max value for fit (we want to make the fit for low illuminations)
    for i in range (len(mean)):
        if  mean[i] < mean_max :
            diff_var_.append(var[i])
            mean_.append(mean[i])
    y_params, cov= curve_fit(linear, mean_, diff_var_)
    return y_params[0],y_params[1]



def quadra (x,a,b,c):
    return a*x**2+b*x+c
def linear (x,a,b):
    return a*x+b

def fit_quadra (var,mean,p0) :
    
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
            
            #err_data.append(np.sqrt(err[i]))
            err_data.append(0.02*var[i])
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
        output : 'run','raft','type','sensor','ampli','a','b','c','chisq_dof','gain','turnoff'
          (a,b,c are the parameters of a quadratic fit ax²+bx+c)                                    

    """
    c = []
    raft = tab['raft'].unique()
    sensor = tab['sensor'].unique()
    dd = pd.DataFrame()
    for r in raft:
        for s in sensor:
            for p in range(1,17):
                idx = tab[(tab['ampli']==p)&
                          (tab['raft']==r)&
                          (tab['sensor']==s)].index
                if not list(idx):
                    continue
                else:
                    print(r, s, p)
                    mean = tab['mean'][idx].reset_index(drop=True)
                    var = tab['var'][idx].reset_index(drop=True)
                    a,b = fit_lin(var, mean)

                    y_param_qua, chisq_dof = fit_quadra(var, mean,b)
                    Qgain = 1/y_param_qua[1]
                    new_data, ndata, y_param_lin= fit_lin_pr(mean, var)
                    Lgain = 1/y_param_lin[0]
                    
                    tt = turnoff (pd.DataFrame({'mean':mean,'var':var}), Lgain)

                    #residue(y_params,a,b,mean, var, p)
                    
                    #ccd = tab['type'][tab['raft']==r]
                    ccd = 1
                    c.append((run,r,ccd,s,p,
                              y_param_qua,Qgain,
                              y_param_lin, Lgain,
                              tt))
                    
                    nd = add_col(ndata, r, s, p)
                    dd = pd.concat((dd, nd))
    parameters = pd.DataFrame(c ,columns=['run','raft','type','sensor','ampli',
                                          'param_quadra', 'gain_quadra',
                                          'param_lin', 'gain_lin',
                                          'turnoff'])       
    
    return parameters, dd

def add_col (df,r,s,p):
    df['raft'] = r
    df['sensor'] = s
    df['ampli'] = p
    return df

def residue (param, a, b, x_data, y_data, ampli):
    
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    y_fit_quadra = quadra(x_data,param[0],param[1], param[2])
    y_fit_lin = a*x_data+b
    
    res = y_data - y_fit_quadra
    res_lin = y_data - y_fit_lin
    plt.plot(x_data, res_lin/x_data, '.', c = colors[ampli-1], label = ampli-1)
    plt.xlim(0,3000)
    plt.ylim(-0.06,0.30)
    plt.ylabel('(var(flat0-flat1) - y_fit)/mean')
    plt.xlabel('mean')
    

    plt.legend()
    return 


def turnoff (data, gain):
    inf = 50000
    sup = 100000
    idx = data[(data['mean'] >=inf) & (data['mean']<=sup)].index
    if not list(idx):
        print('impossible de déterminer le turnoff')   
        tt = 0
    else : 
        xx = data['mean'][idx]
        yy = data['var'][idx]
        
        
        #f = interpolate.interp1d(data['mean'], data['var'])
        f = spi.interp1d(xx,yy,fill_value="extrapolate")
        x = np.linspace(min(xx), max(xx))
        y = f(x)
    
        vmax_idx = np.argmax(y)
        tt = x[vmax_idx]
    # plt.figure()
    # plt.plot(data['mean'], data['var'], '.')
    # plt.plot(x, y, label = tt)
    # plt.legend()
    # plt.show()
    
    return tt
        
    
def fit_lin_pr (x, y):
    
    #fait un fit par bin (valeur moyenne)
    data = pd.DataFrame({'mean':x, 'var':y})
    data = data.sort_values(by='mean').reset_index(drop=True)
    
    #Keep the positive value 
    dd = data[data['mean']>0].reset_index(drop=True)

    med = np.median(data['var'])
    sig = 0.25
    dd_new= dd[dd['var'] < sig * med].reset_index(drop=True)

   
    sigma = 100
    n = len(dd_new['mean'])
    bim=[]
    biv=[]
    m=[]
    
    for i in range (2,n-1,2):
        if dd_new['mean'][i] in m:  #vérifie que la valeur n'est pas comptée dans le bin précédent
            continue
        else:
            inf = dd_new['mean'][i]
            sup = inf+3
            m=[]
            v=[]
            
            #Regroupe des valeurs 
            for j in dd_new['mean']:
                if inf <= j <= sup:
                    m.append(j)
                    v.append(dd_new['var'][dd_new['mean']==j].values[0])
   
            #calcule la moyenne
            bim.append(np.mean(m))
            biv.append(np.mean(v))
    

    
    data = pd.DataFrame({'mean':bim, 'var':biv})   

    n = len(data['mean'])     
    for j in range (2,n-1):
       
        popt, pcov = curve_fit(linear, data['mean'][0:j], data['var'][0:j])

        
        if abs(linear(data['mean'][j+1], *popt)-data['mean'][j+1]) > 10*sigma :
                if abs(linear(data['mean'][j+2], *popt)-data['mean'][j+2]) < 100*sigma:
                    sigma = pcov[0][0]
                    data = data.drop(j+1).copy()  #supprime val j+1 en >100 sigma et garde la j+2 <100 sigma
                else:
                    break
        else:
            sigma = pcov[0][0]

    popt, pcov = curve_fit(linear, data['mean'], data['var'])
    return dd_new, data, popt
 
        
        
   

# resa = pd.read_hdf("../fit_resu/fit_resu2_13144.hdf5")
# run ='13144'
# raft = 'R22'
# sensor = 'S11'
# ampli = '10'
# ptcdir = '../ptc_resu/'
# file = pd.concat([pd.read_hdf(ptcdir+'ptc_{}_flat_ND.hdf5'.format(run)),
#                   pd.read_hdf(ptcdir+'ptc_{}_flat_empty.hdf5'.format(run))]).reset_index(drop=True)

# resa = file[(file['raft']==raft)&(file['sensor']==sensor)&(file['ampli']==10)]
# mn, p= fit_lin_pr(resa['var'], resa['mean'])
# tt = turnoff(resa, 1/p[0] )




































