#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from visu_ptc import visualizePTCccd
import pandas as pd
from num import concatenate_data, GetRunNumber 
from io_tools import checkDir




parser = OptionParser(
    description='Script to plot the PTC')

parser.add_option('--dataPtcDir', type=str,
                  default='../ptc_resu/',
                  help='Path to data [%default]')
parser.add_option('--outDir', type=str,
                  default="../fit_resu/",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="fit_resu_",
                  help='output file name [%default]')
parser.add_option('--prefix', type=str,
                  default=['empty','ND'],
                  help='prefix for Flat files [%default]')
parser.add_option('--dirgain', type=str,
                  default='../gain/',
                  help='Out dir for the calculated gain [%default]')
parser.add_option('--OutdirData', type=str,
                  default='../fit_data/',
                  help='Out dir for the reduces data [%default]')
parser.add_option('--outNameData', type=str,
                  default='reduced_data',
                  help='output file name [%default]')
parser.add_option('--run_num', type=str,
                  default=['13144'],
                  help='output file name [%default]')
parser.add_option('--suffix', type=str,
                  default='',
                  help='suffix output file name [%default]')


opts, args = parser.parse_args()
dataPtcDir = opts.dataPtcDir
outDir = opts.outDir
outName = opts.outName
prefix = opts.prefix
dirgain = opts.dirgain
OutdirData = opts.OutdirData
outNameData = opts.outNameData
suffix = opts.suffix

checkDir(dirgain)
checkDir(OutdirData)
opts, args = parser.parse_args()
run_num = opts.run_num
runs = GetRunNumber(dataPtcDir)
if run_num == 'all' :
    run_num = runs['run'].unique()




for run in run_num:
    print(run)
    dd = list(runs['FileName'][runs['run'] == run])
    df = concatenate_data(dd, dataPtcDir, run).reset_index(drop=True)
    data_fit = outName + run
    df_fit = pd.read_hdf('{}{}{}.hdf5'.format(outDir, data_fit, suffix)).reset_index(drop=True)
    data_fit_lin = '{}{}{}'.format(outNameData, run, suffix)
    df_fit_lin = pd.read_hdf('{}{}{}.hdf5'.format(OutdirData, data_fit_lin, suffix)).reset_index(drop=True)
    visualizePTCccd(df_fit, df, df_fit_lin, run)
    
    


    