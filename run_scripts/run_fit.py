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
from ptc import merge_data, process, fit, plot_ampli
from utils import multiproc
from io_tools import checkDir
from optparse import OptionParser
import numpy as np
import time
import matplotlib.pyplot as plt
from fit import lecture_ptc
from num_run import GetRunNumber


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





path = "/home/julie/fm_study/im_pipe/input"
run = GetRunNumber(path)


name_empty = '../input/ptc_13018_flat_empty.hdf5'
name_ND = '../input/ptc_13018_flat_ND.hdf5'


parser.add_option('--run', type=str,
                  default=run,
                  help='[%default]')
parser.add_option('--flat_empty_ptc', type=str,
                  default=name_empty,
                  help='[%default]')
parser.add_option('--flat_ND_ptc', type=str,
                  default=name_ND,
                  help='[%default]')
parser.add_option('--outDir', type=str,
                  default="../fit_resu",
                  help='output dir for the processed data[%default]')
parser.add_option('--outName', type=str,
                  default="fit_test.hdf5",
                  help='output file name[%default]')

opts, args = parser.parse_args()


flat_empty_ptc = opts.flat_empty_ptc
flat_ND_ptc = opts.flat_ND_ptc
outDir = opts.outDir
outName = opts.outName

checkDir(outDir)

data_ptc= pd.concat([pd.read_hdf(flat_empty_ptc),pd.read_hdf(flat_ND_ptc)],axis=0)


raft = data_ptc['raft'].unique()

res_fit = lecture_ptc(data_ptc)


outPath = '{}/{}'.format(outDir, outName)
res_fit.to_hdf(outPath, key='ptc')


#plot_ampli(data_ptc[data_ptc['raft']=='R12'], res_fit[res_fit['raft']=='R12'])


# from datetime import date
# def time (data):
#     t = pd.DataFrame()
#     t['time'] = data['flat0'].str.split('_').str.get(2)
#     t['date'] = date.fromisoformat(t['time'])
#     return t


# t =time(a)





