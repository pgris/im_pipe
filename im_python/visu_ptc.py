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

def visualizePTC (parameters, data, fitdata, run) :
    """
    

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    fitdata : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    l=np.linspace(0,60000,50)
    xx = 60000/2
    yy=100
    legend = "HDU {} \nGain quadratique = {}\nGain linéaire = {}\nTurnoff = {}"
    raft = parameters['raft'].unique()
    data = data.reset_index(drop=True)
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
                    
                    
                    mean_fit = fitdata['mean'][(fitdata['ampli']==i+1)&
                               (fitdata['raft']==r)&
                               (fitdata['sensor']==s)]
                    var_fit = fitdata['var'][(fitdata['ampli']==i+1)&
                                          (fitdata['raft']==r)&
                                          (fitdata['sensor']==s)]
                    
                    
                    Qgain = parameters['gain_quadra'][idx].values[0]
                    Lgain = parameters['gain_lin'][idx].values[0]

                    axs[irow, icol].plot(mean,var,'+k')
                    if Lgain == 'undetermined':
                        continue
                    else:
                        turnoff = parameters['turnoff'][idx].values[0]
                        if turnoff == 'undetermined':
                            turnoff = 'undetermined'
                        else:
                            turnoff = np.round(turnoff,4)
                            
                        yq = parameters['param_quadra'][idx].values[0]
                        a, b, c = yq[0], yq[1], yq[2]
                        
                        yl = parameters['param_lin'][idx].values[0]
                        a_lin, b_lin = yl[0], yl[1]
     
                        
                        axs[irow, icol].axvline(turnoff, linestyle = '--')
                        
                        
                        axs[irow, icol].plot(mean_fit,var_fit,'+y')
                                                                          
                        axs[irow, icol].plot(l,a*l**2+b*l+c, 'r')
                     
                        axs[irow, icol].plot(l,a_lin*l+b_lin, 'g')
                      
                        axs[irow,icol].text(xx,yy,legend.
                                            format(i+1, np.round(Qgain,2), 
                                                   np.round(Lgain,2),
                                                   turnoff), fontsize=12)
                
                #run = data['run'][(data['raft']==r)&(data['sensor']==s)]
                name = 'PTC_{}_{}_{}'.format(r, s, run)
                fig.suptitle(name)
                fig.supxlabel('mean signal (ADU)')
                fig.supylabel('var ($ADU^{2}$)')
                plt.show()
                plt.close(fig)
                fig.savefig('../fig/'+name)
    return 