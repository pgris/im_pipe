#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Mar  19 09:49:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from im_tools.io_tools import checkDir, get_runList
from im_tools.im_batchutils import BatchIt
from optparse import OptionParser
import numpy as np
import sys

parser = OptionParser(description='Script to estimate the PTC on FP images - batch')

parser.add_option('--dataDir', type=str,
                  default='/sps/lsst/groups/FocalPlane/SLAC/run5',
                  help='Path to data [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--prefix', type=str,
                  default='flat_ND,flat_empty',
                  help='prefix for Flat files [%default]')
parser.add_option('--outDir', type=str,
                  default="/sps/lsst/users/gris/ptc_resu",
                  help='output dir for the processed data[%default]')
parser.add_option('--run_num', type=str,
                  default='13144',
                  help='list of runs to process [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
nproc = opts.nproc
outDir = opts.outDir
prefix = opts.prefix.split(',')
run_num = opts.run_num

# get the runList
runList = get_runList('{}/*'.format(dataDir))

print(runList)
runs = []
print('allo',run_num)
if run_num != 'all':
    run_num = run_num.split(',')
    runs = list(set(runList).intersection(run_num))

    if not runs:
        print('Run(s) not found', run_num)
        sys.exit(0)
else:
    runs = runList

# create outdir if it does not exist
checkDir(outDir)

# make batches to run

script = 'run_scripts/run_ptc_cc.py'
params = {}
params['dataDir'] = dataDir
params['nproc']=nproc

for run in runs:
    for pref in prefix:
        processName = 'ptc_{}_{}'.format(run,pref)
        print('procName',processName)
        batch = BatchIt(processName=processName,time='40:00:00',conda_activate=False,setup_activate=True)
        
        params['outDir'] = '{}/{}'.format(outDir,run)
        params['outName'] = 'ptc_{}_{}.hdf5'.format(run,pref)
        params['prefix'] = pref
        params['run_num'] = run

        batch.add_batch(script,params)
        batch.go_batch()
