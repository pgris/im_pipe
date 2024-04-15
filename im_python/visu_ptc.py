#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:15:50 2024

@author: julie
"""

import matplotlib.pyplot as plt
import numpy as np

UnBias='1D'
#UnBias='2D'

# def visualizePTC (data, parameters) -> plt.figure :
#     """
#     Plot the ptc for each raft/sensor for the 16 HDU

#     Parameters
#     ----------
#     file_name : 
#     data : pd.dataframe
#         Mean and variance for each raft/sensor/HDU
        
#     parameters : pd.dataframe
#         parameters of fit + turnoff + gain
        
#     Returns
#     -------
#     None.

#     """
#     l=np.linspace(0,60000,50)
#     xx = 60000/2
#     yy=100
#     legend = "HDU {} \nGain = {}\nChisq_dof = {}\nTurnoff = {}"
#     raft = parameters['raft'].unique()
#     sensor = parameters['sensor'].unique()
#     for r in raft:
#         for s in sensor:
#             idxc = data[(data['raft']==r)&(data['sensor']==s)].index
#             if not list(idxc):
#                 continue
#             else:
#                 fig, axs = plt.subplots(4,4,sharex=True,figsize=(18,9))
#                 for i in range (16):
#                     irow = int(i/4)
#                     icol = i%4
#                     idx = data[(data['ampli']==i+1)&
#                                (data['raft']==r)&
#                                (data['sensor']==s)].index
#                     mean = data['mean'][idx] 
#                     var = data['var'][idx]
#                     idx = parameters[(parameters['ampli']==i+1)&
#                                      (parameters['raft']==r)&
#                                      (parameters['sensor']==s)].index
#                     a = parameters['a'][idx].values[0]
#                     b = parameters['b'][idx].values[0]
#                     c = parameters['c'][idx].values[0]
#                     a_lin = 1/parameters['gain_lin'][idx].values[0]
#                     b_lin = parameters['blin'][idx].values[0]
#                     turnoff = parameters['turnoff'][idx].values[0]
#                     gain = parameters['gain'][idx].values[0]
#                     chisq_dof = parameters['chisq_dof'][idx].values[0]
#                     axs[irow, icol].plot(mean,var,'.')
#                     axs[irow, icol].plot(l,a*l**2+b*l+c)
#                     axs[irow, icol].plot(l,a_lin*l+b_lin, 'g')
#                     axs[irow,icol].text(xx,yy,legend.
#                                         format(i+1, np.round(gain,2), 
#                                                np.round(chisq_dof,2),
#                                                np.round(turnoff,4)),
#                                         fontsize=12)
#                 run = data['run'][(data['raft']==r)&(data['sensor']==s)]
#                 name = 'PTC for {} {} {}'.format(r, s, run)
#                 fig.suptitle(name)
#                 fig.supxlabel('mean signal (ADU)')
#                 fig.supylabel('var ($ADU^{2}$)')
                
#                 fig.savefig('../fig/'+name)
#     return 


def visualizePTC (ndata, data, parameters) -> plt.figure :
    """
    Plot the ptc for each raft/sensor for the 16 HDU

    Parameters
    ----------
    file_name : 
    data : pd.dataframe
        Mean and variance for each raft/sensor/HDU
        
    parameters : pd.dataframe
        parameters of fit + turnoff + gain
        
    Returns
    -------
    None.

    """
    l=np.linspace(0,60000,50)
    xx = 60000/2
    yy=100
    legend = "HDU {} \nGain quadratique = {}\nGain linéaire = {}\nTurnoff = {}"
    raft = parameters['raft'].unique()
    sensor = parameters['sensor'].unique()
    for r in raft:
        for s in sensor:
            idxc = data[(data['raft']==r)&(data['sensor']==s)].index
            if not list(idxc):
                continue
            else:
                fig, axs = plt.subplots(4,4,sharex=True,figsize=(18,9))
                for i in range (16):
                    irow = int(i/4)
                    icol = i%4
                    idx = data[(data['ampli']==i+1)&
                               (data['raft']==r)&
                               (data['sensor']==s)].index
                    mean = data['mean'][idx] 
                    var = data['var'][idx]
                    idx = parameters[(parameters['ampli']==i+1)&
                                     (parameters['raft']==r)&
                                     (parameters['sensor']==s)].index
                    
                    yq = parameters['param_quadra'][idx].values[0]
                    a, b, c = yq[0], yq[1], yq[2]
                    
                    yl = parameters['param_lin'][idx].values[0]
                    a_lin, b_lin = yl[0], yl[1]
                   
                    
                    turnoff = parameters['turnoff'][idx].values[0]
                    Qgain = parameters['gain_quadra'][idx].values[0]
                    Lgain = parameters['gain_lin'][idx].values[0]
                    
                    
                    
                    axs[irow, icol].axvline(turnoff, linestyle = '--')
                    axs[irow, icol].plot(mean,var,'+k')
                    
                    axs[irow, icol].plot(ndata['mean'][(ndata['ampli']==i+1)&
                               (ndata['raft']==r)&
                               (ndata['sensor']==s)],ndata['var'][(ndata['ampli']==i+1)&
                                          (ndata['raft']==r)&
                                          (ndata['sensor']==s)],'+y')
                    axs[irow, icol].plot(l,a*l**2+b*l+c, 'r')
                    axs[irow, icol].plot(l,a_lin*l+b_lin, 'g')
                  
                    axs[irow,icol].text(xx,yy,legend.
                                        format(i+1, np.round(Qgain,2), 
                                               np.round(Lgain,2),
                                               np.round(turnoff,4)),
                                        fontsize=12)
                run = '13144'
                #run = data['run'][(data['raft']==r)&(data['sensor']==s)]
                name = 'PTC for {} {} {}'.format(r, s, run)
                fig.suptitle(name)
                fig.supxlabel('mean signal (ADU)')
                fig.supylabel('var ($ADU^{2}$)')
                fig.savefig('../fig/'+name)
                plt.show()
    return 