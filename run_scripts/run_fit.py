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
from optparse import OptionParser
from num import GetRunNumber, DoFit
from io_tools import checkDir

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20



parser = OptionParser(
    description='Script to estimate the gain and turnoff from the PTC on FP images')


parser.add_option('--dataPtcDir', type=str,
                  default='../ptc_resu/',
                  help='Path to data [%default]')
parser.add_option('--run_num', type=str,
                  default=['13144'],
                  help='list of runs to process [%default] or [allrun]')
parser.add_option('--outDir', type=str,
                  default="../fit_resu/",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="fit_resu_",
                  help='output file name [%default]')

parser.add_option('--OutdirData', type=str,
                  default='../fit_data/',
                  help='Out dir for the reduces data [%default]')

parser.add_option('--outNameData', type=str,
                  default='reduced_data',
                  help='output file name [%default]')
parser.add_option('--suffix', type=str,
                  default='',
                  help='suffix output file name [%default]')



opts, args = parser.parse_args()
dataPtcDir = opts.dataPtcDir
runs = GetRunNumber(dataPtcDir)

run_num = opts.run_num
outDir = opts.outDir
suffix = opts.suffix
outName = opts.outName
OutdirData = opts.OutdirData
outNameData = opts.outNameData
r = runs['run'].unique()
checkDir(OutdirData)

if run_num == 'allrun':
    run_num = r

for run in run_num:
    res = pd.DataFrame()
    lin = pd.DataFrame()
    qua = pd.DataFrame()
    print(run)
    dd = list(runs['FileName'][runs['run'] == run])
    lineardata, quadradata, file, resu_fit = DoFit(dd, dataPtcDir, run) 
    res = pd.concat([res, resu_fit]).reset_index(drop=True)
    lin = pd.concat([lin, lineardata]).reset_index(drop=True)
    qua = pd.concat([qua, quadradata]).reset_index(drop=True)

    # save fit result
    outPath = '{}{}{}{}.hdf5'.format(outDir, outName, run, suffix)
    resu_fit.to_hdf(outPath, key='ptc')
    # save Reduced data
    outPathData = '{}{}_lin_{}{}.hdf5'.format(OutdirData, outNameData, run, suffix)
    lin.to_hdf(outPathData, key='data')
    outPathData = '{}{}_quadra_{}{}.hdf5'.format(OutdirData, outNameData, run, suffix)
    qua.to_hdf(outPathData, key='data')
    


