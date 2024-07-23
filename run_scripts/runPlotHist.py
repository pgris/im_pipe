#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:34:21 2024

@author: julie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import matplotlib.pyplot as plt
import os 
from optparse import OptionParser
import numpy as np
from num import GetRunNumber, DoFit
from visu_ptc import visualizePTC
from ptc import process
from fit import residue_linear
from scipy.optimize import curve_fit

def process_ptc_multi(toproc, params, j=0, output_q=None):
    """
    Function to process ptc using multiprocessing

    Parameters
    ----------
    toproc : list(int)
        List of indexes (row num) to process.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        Where to dump the data. The default is None.

    Returns
    -------
    pandas dataframe
        Output Data.

    """

    imin = np.min(toproc)
    imax = np.max(toproc)+1
    data = params['data'][imin:imax]
    res = process_ptc(data)

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def process_ptc(data) -> pd.DataFrame:
    """
    Function to process the ptc

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    resa : pandas df
        Processed data.

    """

    resa = pd.DataFrame()
    for i, row in data.iterrows():
        flat0 = row['full_path_flat0']
        flat1 = row['full_path_flat1']
        res = process(flat0, flat1)
        resa = pd.concat([res, resa]).reset_index(drop=True)

    return resa


parser = OptionParser(description='Script to estimate the gain from the PTC on FP images')

parser.add_option('--dataPtcFitDir', type=str,
                  default='../fit_resu/',
                  help='Path to data [%default]')
parser.add_option('--ptcdir', type=str,
                  default='../data/',
                  help='Path to data [%default]')



opts, args = parser.parse_args()
dataPtcFitDir = opts.dataPtcFitDir
ptcdir = opts.ptcdir


def gauss (x, amp, mu, sigma):
    return amp*np.exp(-(x-mu)**2/(2*sigma**2))

#histrogramme des gains sur le plan focal pour un run 
#en fonction du type de ccd
def plothistFabr(file, param, run):
    ITL = [ 'R01', 'R02', 'R03', 'R10',
            'R20',  'R41', 'R42', 'R43']
    e2v = ['R11', 'R12', 'R13', 'R14', 'R21', 'R22', 'R23', 
            'R24', 'R30', 'R31', 'R32', 'R33', 'R34']
     
    # std = np.std(file['gain_lin'])
    # m = np.mean(file['gain_lin'])
    plt.figure()
    raft = file['raft'].unique()
    tITL =[]
    satITL=pd.DataFrame()
    te2v = []
    sate2v = pd.DataFrame()
    gainsg = []
    gain_sg = pd.DataFrame()
    for r in raft:
        if r in ITL:
            tITL=(file[param][(file['raft']==r)&(file[param]!='undetermined')])
            satITL = pd.concat((satITL,tITL))
        elif r in e2v: 
            te2v=(file[param][(file['raft']==r)&(file[param]!='undetermined')])
            sate2v = pd.concat((sate2v,te2v))
        else:
            gainsg = (file[param][(file['raft']==r)&(file[param]!='undetermined')])
            gain_sg = pd.concat((gain_sg,gainsg))
            
    ITL_std = np.std(satITL)
    ITL_mean = np.mean(satITL)
    e2v_std = np.std(sate2v)
    e2v_mean = np.median(sate2v)
    sg_std = np.std(gain_sg)
    sg_mean = np.median(gain_sg)

    ni, binsi, patchesi = plt.hist(satITL, bins = 'auto', histtype = 'stepfilled', alpha=0.5,
                                      linewidth = 1, color='red',
                                      label ='ITL', edgecolor="black");
       
    n, bins, patches = plt.hist(sate2v, bins = 'auto', histtype = 'stepfilled',alpha=0.5,
                                linewidth = 1, label ='e2v',  edgecolor="black");
    
    #plt.hist(gain_sg, bins = 50, histtype = 'step', label ='coin : mean = {} +- {}'.format(np.round(sg_mean,2), np.round(sg_std[0])));
    #plt.xlabel('Saturation [photoelectron]')
    plt.xlabel(param)
    plt.ylabel('count')
    plt.show()
    popt, pcov = curve_fit(gauss, bins[1:], n, p0=[np.max(n), int(e2v_mean), int(e2v_std)])
    popti, pcovi = curve_fit(gauss, binsi[1:], ni, p0=[np.max(ni), int(ITL_mean), int(ITL_std)])

    plt.plot(bins[0:-1], gauss(bins[0:-1], *popt), '-b', linewidth = 1)
    plt.plot(binsi[0:-1], gauss(binsi[0:-1], *popti), '-r',  linewidth = 1)
    
    plt.legend(['ITL : mean = {} +- {}'.format(int(popti[1]), int(popti[2])),
                'e2v : mean = {} +- {}'.format(int(popt[1]), int(popt[2]))])
    
    plt.savefig('../graphique/histsat')
    return popt, pcov 




def plothist(file, param, run):
    
    std = np.std(file[param])
    m = np.mean(file[param])
    plt.figure()
       
    file = file[param][file[param]!='undetermined']

    n, bins, patches = plt.hist(file, bins = 'auto', histtype = 'stepfilled',alpha=0.5,
                                linewidth = 1, label ='focalplane',  edgecolor="black");
    
    #plt.hist(gain_sg, bins = 50, histtype = 'step', label ='coin : mean = {} +- {}'.format(np.round(sg_mean,2), np.round(sg_std[0])));
    #plt.xlabel('Saturation [photoelectron]')
    plt.xlabel(param)
    plt.ylabel('count')
    plt.show()
    popt, pcov = curve_fit(gauss, bins[0:-1], n, p0=[np.max(n), int(m), int(std)])
  

    plt.plot(bins[0:-1], gauss(bins[0:-1], *popt), '-b', linewidth = 1)

    
    plt.legend(['focalplane : mean = {} +- {}'.format(int(popt[1]), int(popt[2]))])
    
    plt.savefig('../graphique/histsatall')
    return popt, pcov


file=pd.read_hdf('../gain/gain.hdf5')
data = file[file['gain_lin']!='undetermined']
data = data[data['turnoff']!='undetermined']
data['turnoff(photoel)'] = data['gain_lin']*data['turnoff']


popt, pcov = plothist(data, 'turnoff(photoel)', 'tous')


#Déterminer les outliers
n = 3
inf = data['turnoff(photoel)'] > popt[1] + n * popt[2]
sup = data['turnoff(photoel)'] < popt[1] - n * popt[2]
outliers = data[(inf)|(sup)]



popt, pcov = plothist(data, 'gain_lin', 'tous')
popt, pcov = plothistFabr(data, 'gain_lin', 'all run')
popt, pcov = plothistFabr(data, 'turnoff(photoel)', 'all run')

num_run = data['run'].unique()
num_run = ['13144']
for run in num_run:
    plothist(data[data['run']==run], 'turnoff(photoel)', run)
    plothistFabr(data[data['run']==run], 'turnoff(photoel)', run)











