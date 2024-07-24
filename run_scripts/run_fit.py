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


import glob
import os
from io_tools import checkDir
import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser
from num import GetRunNumber, do_fit
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


parser = OptionParser(
    description='Script to estimate the gain and turnoff from the PTC on FP images')


parser.add_option('--dataPtcDir', type=str,
                  default='../ptc_resu',
                  help='Path to data [%default]')
parser.add_option('--run_num', type=str,
                  default='13144',
                  help='list of runs to process [%default] or [allrun]')
parser.add_option('--outDir', type=str,
                  default="../fit_resu",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="fit_resu",
                  help='output file name [%default]')

parser.add_option('--OutdirData', type=str,
                  default='../fit_data',
                  help='Out dir for the reduces data [%default]')

parser.add_option('--outNameData', type=str,
                  default='reduced_data',
                  help='output file name [%default]')
parser.add_option('--suffix', type=str,
                  default='',
                  help='suffix output file name [%default]')


opts, args = parser.parse_args()
dataPtcDir = opts.dataPtcDir
# runs = GetRunNumber(dataPtcDir)

runs = os.listdir(dataPtcDir)  # list the file in the folder
runs = list(map(int, runs))
run_num_ = list(map(int, opts.run_num.split(',')))
run_num = list(set(run_num_) & set(runs))

if not run_num:
    print('The runs to process', run_num_, 'do not exist.')
    print('List of available runs:')
    for vv in runs:
        print(vv)
    exit()


outDir = opts.outDir
suffix = opts.suffix
outName = opts.outName
OutdirData = opts.OutdirData
outNameData = opts.outNameData

for run in run_num:
    dd = glob.glob('{}/{}/*'.format(dataPtcDir, run))
    print('processing', run, dd)
    lineardata, quadradata, file, resu_fit = do_fit(dd)
    # save fit result
    outDir_n = '{}/{}'.format(outDir, run)
    checkDir(outDir_n)
    outPath = '{}/{}{}.hdf5'.format(outDir_n, outName, suffix)

    resu_fit.to_hdf(outPath, key='ptc')
    # save Reduced data
    outDirData_n = '{}/{}'.format(OutdirData, run)
    checkDir(outDirData_n)
    outPathData = '{}/{}_lin_{}.hdf5'.format(
        outDirData_n, outNameData, suffix)
    lineardata.to_hdf(outPathData, key='data')
    outPathData = '{}/{}_quadra_{}.hdf5'.format(
        outDirData_n, outNameData, suffix)
    quadradata.to_hdf(outPathData, key='data')
