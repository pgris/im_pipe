#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from ptc import merge_data, process
from utils import multiproc
from io_tools import checkDir
from optparse import OptionParser
import numpy as np
import glob


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


parser = OptionParser(description='Script to estimate the PTC on FP images')

parser.add_option('--flat_empty', type=str,
                  default='input/flat_13144_flat_empty.csv',
                  help='[%default]')
parser.add_option('--flat_ND', type=str,
                  default='input/flat_13144_flat_ND.csv',
                  help='[%default]')
parser.add_option('--data_path', type=str,
                  default="/home/philippe/Bureau/FP_flats",
                  help='[%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--outDir', type=str,
                  default="ptc_resu",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="ptc_test.hdf5",
                  help='output file name[%default]')

opts, args = parser.parse_args()

flat_empty = opts.flat_empty
flat_ND = opts.flat_ND
data_path = opts.data_path
nproc = opts.nproc
outDir = opts.outDir
outName = opts.outName


# create outdir if it does not exist
checkDir(outDir)

flat = [flat_empty, flat_ND]


# grab the files in list pair(str)
doss = glob.glob('{}/*.fits'.format(data_path))
data = merge_data(doss, flat)

# processing of the data
resa = pd.DataFrame()

params = {}
params['data'] = data
vv = list(range(len(data)))
resa = multiproc(vv, params, process_ptc_multi, nproc)

# save data
outPath = '{}/{}'.format(outDir, outName)
resa.to_hdf(outPath, key='ptc')
