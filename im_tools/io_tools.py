#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Mar  18 14:38:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import os
import glob
import pandas as pd


def checkDir(outDir):
    """
    function to check whether a directory exist
    and create it if necessary
    """
    if not os.path.isdir(outDir):
        os.makedirs(outDir)


def get_flat1(data, prefix='flat1'):
    """
    Function to get flat1 images

    Parameters
    ----------
    data : array
        Data ??.
    prefix : str, optional
        prefix for the search. The default is 'flat1'.

    Returns
    -------
    r : list(str)
        List of output files.

    """

    r = []
    for vv in data:
        vva = vv.split('/')
        main_dir = '/'.join(vva[:-1])
        flatnum = vva[-1].split('_')[-1]
        flatnum = int(flatnum)
        # increase to 1 to get flat1
        flatnum += 1
        fis = glob.glob('{}/*flat1_{}'.format(main_dir, str(flatnum).zfill(3)))
        r.append((vv, fis[0]))

    return r


def assemble_flats(llist,runNum):
    """
    Function to make pairs of (flat0, flat1) files

    Parameters
    ----------
    llist : list(str)
        List of files.
    runNum: int
      run number

    Returns
    -------
    df_tot : pandas df
        Output data.

    """

    df_tot = pd.DataFrame()
    for ll in llist:
        bb_flat0 = list_df(ll[0], 'flat0',runNum)
        bb_flat1 = list_df(ll[1], 'flat1',runNum)

        # print(bb_flat0)

        res = bb_flat0.merge(bb_flat1, left_on=['raft_sensor','runNum'], right_on=[
                             'raft_sensor','runNum'], suffixes=['', ''])

        df_tot = pd.concat((df_tot, res))

    return df_tot


def list_df(theDir, prefix,runNum):
    """
    Function to make list and put it in dataFrame

    Parameters
    ----------
    theDir : str
        Data dir.
    prefix : str
        Prefix for the search.
    runNum: int
        Run number

    Returns
    -------
    df : pandas df
        output data.

    """
    fis = glob.glob('{}/*R*_S*.fits'.format(theDir))
    colName = 'file_{}'.format(prefix)
    df = pd.DataFrame(fis, columns=[colName])
    df['raft_sensor'] = df[colName].map(lambda x: strip(x))
    df[colName] = df[colName].str.split('/').str.get(-1)
    df['dataDir_{}'.format(prefix)] = theDir
    df['runNum'] = runNum

    return df


def strip(grp):
    """
    Function to split

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : str
        Strip data.

    """

    vv = grp.split('.fits')[0].split('/')[-1].split('_')

    res = '{}_{}'.format(vv[-2], vv[-1])

    return res


def get_runList(dataDir):
    """
    Function to grab the list of runs in the dataDir

    Parameters
    ----------
    dataDir : str
        data directory.

    Returns
    -------
    runList : list(int)
        List of runs.

    """

    runs = glob.glob('{}'.format(dataDir))

    runList = list(map(lambda elem: elem.split('/')[-1], runs))

    #runList = list(map(int, runList))

    return runList


def get_flat_pairs(dataDir, runNum, prefix):
    """
    Method to grab flat pairs

    Parameters
    ----------
    dataDir : str
        Data dir.
    runNum : int
        Run number.
    prefix : str
        Prefix for file search.

    Returns
    -------
    df : pandas df
        output data.

    """

    fis_flat0 = glob.glob('{}/{}/{}*flat0_*'.format(dataDir, runNum, prefix))

    fis_flat = get_flat1(fis_flat0)

    df = assemble_flats(fis_flat,runNum)

    return df
