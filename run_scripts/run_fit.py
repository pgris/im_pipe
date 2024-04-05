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
import os
import matplotlib.pyplot as plt
from optparse import OptionParser
import numpy as np
from num import GetRunNumber, DoFit
from ptc import process
from visu_ptc import visualizePTC

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




parser.add_option('--dataPtcDir', type=str,
                  default='../ptc_resu/',
                  help='Path to data [%default]')

opts, args = parser.parse_args()
dataPtcDir = opts.dataPtcDir
runs = GetRunNumber(dataPtcDir)


parser.add_option('--run_num', type=str,
                  default=runs['run'].unique(),
                  help='list of runs to process [%default]')

parser.add_option('--outDir', type=str,
                  default="../fit_resu",
                  help='output dir for the processed data[%default]')

parser.add_option('--outName', type=str,
                  default="fit_resu2_",
                  help='output file name[%default]')


parser.add_option('--prefix', type=str,
                  default=runs['filter'].unique(),
                  help='prefix for Flat files [%default]')

opts, args = parser.parse_args()
run_num = opts.run_num
outDir = opts.outDir
outName = opts.outName
prefix = opts.prefix
 
gain = pd.DataFrame()

raft = 'R22'
sensor = 'S11'

for run in run_num : 
    print(run) 
    dd = list(runs['FileName'][runs['run']==run])
    resu_fit, data = DoFit(dd, dataPtcDir, run)
    g = resu_fit.drop(['a','b','c','chisq_dof'],axis=1)
    gain = pd.concat([gain, g]).reset_index(drop=True)
    
    
    #plot ptc
    d = data[(data['raft']==raft)&(data['sensor']==sensor)]
    dd = resu_fit[(resu_fit['raft']==raft)&(resu_fit['sensor']==sensor)]
    visualizePTC(d, dd)
    
    
    # save data
    outPath = '{}/{}{}.hdf5'.format(outDir, outName, run)
    resu_fit.to_hdf(outPath, key='ptc')
    gain.to_hdf('../gain/gain.hdf5', key='gain')
    







#a mettre dans run_plot
gg = gain['gain']
rr = gain['run']    
plt.scatter(rr, gg)


path = '../fit_resu/'
file = os.listdir(path)
gain = pd.DataFrame()


for f in file :
    a = pd.read_hdf(path+f)
    gain = gain = pd.concat([gain, a]).reset_index(drop=True)

    
# Générer des couleurs uniques pour chaque numéro d'ampli
num_run = rr.unique()
num_colors = len(num_run)
colors = plt.cm.tab20(np.linspace(0, 1, num_colors))


#scatter plot du gain par run
# Parcourir chaque numéro d'ampli et tracer les points avec la couleur correspondante
for i, num in enumerate(num_run):
    print(num)
    data = gain[gain["run"] == num]
    plt.scatter(data["run"], data["gain"], label=num, c=[colors[i]])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.yscale('log')
plt.show()


#hist du gain par run
for i, num in enumerate(num_run):
    data = gain[gain["run"] == num]
    plt.hist(data["gain"], label=num, color=[colors[i]], bins =100, histtype='step')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim((0.75,2))
plt.show()



































