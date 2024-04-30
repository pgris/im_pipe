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
from io_tools import checkDir

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
allrun = runs['run'].unique()

parser.add_option('--run_num', type=str,
                  default=allrun,
                  help='list of runs to process [%default]')

parser.add_option('--outDir', type=str,
                  default="../fit_resu/",
                  help='output dir for the processed data[%default]')

parser.add_option('--outName', type=str,
                  default="fit_resu_",
                  help='output file name [%default]')


parser.add_option('--prefix', type=str,
                  default=runs['filter'].unique(),
                  help='prefix for Flat files [%default]')

parser.add_option('--dirgain', type=str,
                  default='../gain/',
                  help='Out dir for the calculated gain [%default]')

parser.add_option('--OutdirData', type=str,
                  default='../gain/',
                  help='Out dir for the reduces data [%default]')

parser.add_option('--outNameData', type=str,
                  default='reduced_data_',
                  help='output file name [%default]')


opts, args = parser.parse_args()
run_num = opts.run_num
outDir = opts.outDir
outName = opts.outName
prefix = opts.prefix
dirgain = opts.dirgain
OutdirData = opts.OutdirData
outNameData = opts.outNameData

checkDir(dirgain)
checkDir(OutdirData)


gain = pd.DataFrame()



res = pd.DataFrame()
for run in run_num : 
    print(run) 
    dd = list(runs['FileName'][runs['run']==run])  
    resu_fit,datafit, data= DoFit(dd, dataPtcDir, run)
    #res = pd.concat([res,resu_fit]).reset_index(drop=True)
    g = resu_fit.drop(['param_quadra','param_lin'],axis=1)
    gain = pd.concat([gain, g]).reset_index(drop=True)
  
    
    #save fit result
    outPath = '{}{}{}.hdf5'.format(outDir, outName, run)
    resu_fit.to_hdf(outPath, key='ptc')
    #save Reduced data 
    outPathData = '{}{}{}.hdf5'.format(OutdirData, outNameData, run)
    data.to_hdf(outPathData, key='data')
    
#save gain
gain.to_hdf(dirgain+'gain.hdf5', key='gain')   
outPathData = '{}{}{}.hdf5'.format(OutdirData, outNameData, run_num[0])


run = allrun[0]
dd = list(runs['FileName'][runs['run']==run])  
resu_fit,datafit, data= DoFit(dd, dataPtcDir, run)
visualizePTC(resu_fit, data, datafit, run)



gain = pd.read_hdf('../gain/gain.hdf5')  
plt.hist(gain['gain_lin'][gain['gain_lin']!='undetermined'], bins = 100)
    


    

    








































