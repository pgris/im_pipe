#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Mar  18 14:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from ptc import process
from utils import multiproc
from io_tools import checkDir, get_runList, get_flat_pairs
from optparse import OptionParser
import numpy as np
import sys


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
        flat0 = '{}/{}'.format(row['dataDir_flat0'], row['file_flat0'])
        flat1 = '{}/{}'.format(row['dataDir_flat1'], row['file_flat1'])
        res = process(flat0, flat1)
        resa = pd.concat([res, resa]).reset_index(drop=True)

    return resa


parser = OptionParser(description='Script to estimate the PTC on FP images')

parser.add_option('--dataDir', type=str,
                  default='/sps/lsst/groups/FocalPlane/SLAC/run5',
                  help='Path to data [%default]')
parser.add_option('--run_num', type=str,
                  default='13144',
                  help='list of runs to process [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--outDir', type=str,
                  default="ptc_resu",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="ptc_test.hdf5",
                  help='output file name[%default]')
parser.add_option('--prefix', type=str,
                  default='flat_ND,flat_empty',
                  help='prefix for Flat files [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
run_num = opts.run_num.split(',')
#run_num = list(map(int, run_num))
nproc = opts.nproc
outDir = opts.outDir
outName = opts.outName
prefix = opts.prefix.split(',')

# get the runList
runList = get_runList('{}/*'.format(dataDir))

runs = list(set(runList).intersection(run_num))

if not runs:
    print('Run(s) not found', run_num)
    sys.exit(0)

# create outdir if it does not exist
checkDir(outDir)

# get the list of data
data = pd.DataFrame()
for run in runs:
    for pref in prefix:
        dd = get_flat_pairs(dataDir, run, pref)
        dd['runNum'] = run
        dd['prefix'] = pref
        data = pd.concat((data, dd))

idx = data['raft_sensor'].str.contains('SW')
data = data[~idx]

# processing of the data
resa = pd.DataFrame()

params = {}
params['data'] = data
vv = list(range(len(data)))
resa = multiproc(vv, params, process_ptc_multi, nproc)

# save data
outPath = '{}/{}'.format(outDir, outName)
resa.to_hdf(outPath, key='ptc')
