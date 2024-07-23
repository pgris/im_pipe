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

def visualizePTCccd (parameters, data, fitdata, fitdataquadra, run, rsa) :
    """
    PTC visualization of an entire CCD (16 HDU)

    Parameters
    ----------
    parameters : pd.Dataframe
        Fit result of the PTC : ('raft','sensor','ampli',
                                 'param_lin', 'param_quadra',
                                 'gain_lin', 'gain_quadra', 'turnoff')
    data : pd.Dataframe
        Data 
    fitdata : pd.Dataframe
        Data used for linear fitting
    run : str
        Run number

    Returns
    -------
    Save the plot of the PTC

    """
    if rsa != 'all':
        raft = [str(''.join(rsa[0:3]))]
        sensor = [str(''.join(rsa[4:7]))]
    else :
        raft = parameters['raft'].unique()
        sensor = parameters['sensor'].unique()
    l=np.linspace(0,60000,50)
    xx = 60000/2
    yy=100
    legend = "HDU {} \nGain quadratique = {}\nGain linéaire = {}\nTurnoff = {}"
    data = data.reset_index(drop=True)
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
                    #Data
                    idx = data[(data['ampli']==i+1)&
                               (data['raft']==r)&
                               (data['sensor']==s)].index
                    mean = data['mean'][idx] 
                    var = data['var'][idx]
                    #Gains and turnoff
                    idx = parameters[(parameters['ampli']==i+1)&
                                     (parameters['raft']==r)&
                                     (parameters['sensor']==s)].index
                    Qgain = parameters['gain_quadra'][idx].values[0]
                    Lgain = parameters['gain_lin'][idx].values[0]
                    turnoff = parameters['turnoff'][idx].values[0]  
                    #Data of the linear fitting
                    mean_fit = fitdata['mean'][(fitdata['ampli']==i+1)&
                               (fitdata['raft']==r)&
                               (fitdata['sensor']==s)]
                    var_fit = fitdata['var'][(fitdata['ampli']==i+1)&
                                          (fitdata['raft']==r)&
                                          (fitdata['sensor']==s)]   
                    mean_fitquadra = fitdataquadra['mean'][(fitdataquadra['ampli']==i+1)&
                               (fitdataquadra['raft']==r)&
                               (fitdataquadra['sensor']==s)]
                    var_fitquadra = fitdataquadra['var'][(fitdataquadra['ampli']==i+1)&
                                          (fitdataquadra['raft']==r)&
                                          (fitdataquadra['sensor']==s)]   

                    axs[irow, icol].plot(mean,var,'+k')
                    if Lgain == 'undetermined':
                        Lgain = 'undetermined'
                    else :
                        Lgain = np.round(Lgain,2)
                    if Qgain == 'undetermined':
                        Qgain ='undetermined'
                    else :
                        Qgain =  np.round(Qgain,2) 
                    if turnoff == 'undetermined':
                        turnoff='undetermined'
                    elif turnoff == 'outsider':
                        turnoff = 'outsider'
                    else:
                        turnoff = np.round(turnoff,4)
                            
                    yq = parameters['param_quadra'][idx].values[0]
                    a, b, c = yq[0], yq[1], yq[2]
                    
                    yl = parameters['param_lin'][idx].values[0]
                    a_lin, b_lin = yl[0], yl[1]
 
                    
                    axs[irow, icol].axvline(turnoff, linestyle = '--')    
                    axs[irow, icol].plot(mean_fitquadra,var_fitquadra,'+', color = 'fuchsia')  
                    axs[irow, icol].plot(mean_fit,var_fit,'+y')  
                    axs[irow, icol].plot(l,a*l**2+b*l+c, 'r')                 
                    axs[irow, icol].plot(l,a_lin*l+b_lin, 'g')
                    axs[irow,icol].text(xx,yy,
                                        legend.format(i+1, Qgain,Lgain,turnoff),
                                        fontsize=12)
                    if max(var)>70000:
                        axs[irow,icol].set_ylim(0,70000)
                #run = data['run'][(data['raft']==r)&(data['sensor']==s)]
                name = 'PTC_{}_{}_{}_bas_quadra'.format(r, s, run)
                fig.suptitle(name)
                fig.supxlabel('mean signal (ADU)')
                fig.supylabel('var ($ADU^{2}$)')
                plt.close(fig)
                fig.savefig('../fig/'+name)
    return 




def visualizePTChdu(parameters, data, fitdata,fitdataquadra, run, hdu):
    """
    PTC visualization of an entire CCD (16 HDU)

    Parameters
    ----------
    parameters : pd.Dataframe
        Fit result of the PTC : ('raft','sensor','ampli',
                                 'param_lin', 'param_quadra',
                                 'gain_lin', 'gain_quadra', 'turnoff')
    data : pd.Dataframe
        Data 
    fitdata : pd.Dataframe
        Data used for linear fitting
    run : str
        Run number

    Returns
    -------
    Save the plot of the PTC

    """
    
    l=np.linspace(0,60000,50)
    xx = 60000/2
    yy=100
    legend = "HDU {} \nGain quadratique = {}\nGain linéaire = {}\nTurnoff = {}"
    r = ''.join(hdu[0:3])
    s = ''.join(hdu[4:7])
    i = int(''.join(hdu[8:]))
    print(r,s,i)
    plt.figure()
    idxc = data[(data['raft']==r)&(data['sensor']==s)].index
    if not list(idxc):
        print('No data')
    else:
        #Data
        idx = data[(data['ampli']==i)&
                   (data['raft']==r)&
                   (data['sensor']==s)].index
        
        mean = data['mean'][idx] 
        var = data['var'][idx]
        #Gains and turnoff
        idx = parameters[(parameters['ampli']==i)&
                         (parameters['raft']==r)&
                         (parameters['sensor']==s)].index
        Qgain = parameters['gain_quadra'][idx].values[0]
        Lgain = parameters['gain_lin'][idx].values[0]
        turnoff = parameters['turnoff'][idx].values[0]  
        #Data of the linear fitting
        mean_fit = fitdata['mean'][(fitdata['ampli']==i)&
                   (fitdata['raft']==r)&
                   (fitdata['sensor']==s)]
        var_fit = fitdata['var'][(fitdata['ampli']==i)&
                              (fitdata['raft']==r)&
                              (fitdata['sensor']==s)]   
                        
        mean_fitquadra = fitdataquadra['mean'][(fitdataquadra['ampli']==i+1)&(fitdataquadra['raft']==r)&
                (fitdataquadra['sensor']==s)]
        var_fitquadra = fitdataquadra['var'][(fitdataquadra['ampli']==i+1)&(fitdataquadra['raft']==r)&                        (fitdataquadra['sensor']==s)]   
        plt.plot(mean,var,'+k')
        if Lgain == 'undetermined':
            Lgain = 'undetermined'
        else :
            Lgain = np.round(Lgain,2)
        if Qgain == 'undetermined':
            Qgain ='undetermined'
        else :
            Qgain =  np.round(Qgain,2) 
        if turnoff == 'undetermined':
            turnoff='undetermined'
        elif turnoff == 'outsider':
            turnoff = 'outsider'
        else:
            turnoff = np.round(turnoff,4)
                
        yq = parameters['param_quadra'][idx].values[0]
        a, b, c = yq[0], yq[1], yq[2]
        
        yl = parameters['param_lin'][idx].values[0]
        a_lin, b_lin = yl[0], yl[1]
        
        plt.axvline(turnoff, linestyle = '--')                    
        plt.plot(mean_fit,var_fit,'+y')     
        plt.plot(mean_fitquadra,var_fitquadra,'+', color = 'fuchsia')                                                                   
        plt.plot(l,a*l**2+b*l+c, 'r')                 
        plt.plot(l,a_lin*l+b_lin, 'g')
        plt.text(xx,yy,legend.format(i, Qgain,Lgain,turnoff),fontsize=12)
        if max(var)>50000:
            plt.set_ylim(0,50000)
        #run = data['run'][(data['raft']==r)&(data['sensor']==s)]
        name = 'PTC_{}_{}_{}_{}'.format(r, s, i, run)
        plt.title(name)
        plt.xlabel('mean signal (ADU)')
        plt.ylabel('var ($ADU^{2}$)')
        #plt.close()
        plt.show()
        #plt.savefig('../fig/'+name)
    return