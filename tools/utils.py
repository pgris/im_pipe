#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:43:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import multiprocessing
import pandas as pd
from astropy.table import Table, vstack, Column
import operator


def multiproc(data, params, func, nproc):
    """
    Function to perform multiprocessing

    Parameters
    ---------------
    data: array
      data to process
    params: dict
      fixed parameters of func
    func: function
      function to apply for multiprocessing
    nproc: int
      number of processes

    """
    nproc = min([len(data), nproc])
    # multiprocessing parameters
    nz = len(data)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    # print('multi', nz, t)
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=func,
                                     args=(data[t[j]:t[j+1]], params, j, result_queue))
             for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = gather_results(resultdict)

    return restot


def gather_results(resultdict):
    """
    Function to gather results of a directory

    Parameters
    ----------------
    resultdict: dict
      dictory of data

    Returns
    ----------
    gathered results. The type is determined from resultdict.
    Supported types: pd.core.frame.DataFrame, Table, np.ndarray,
    np.recarray, int

    """
    supported_types = ['pd.core.frame.DataFrame', 'Table', 'np.ndarray',
                       'np.recarray', 'int', 'dict']

    # get outputtype here
    first_value = None
    for key, vals in resultdict.items():
        if vals is not None:
            first_value = vals
            break

    restot = None
    if first_value is None:
        return restot

    if isinstance(first_value, pd.core.frame.DataFrame):
        restot = pd.DataFrame()

        def concat(a, b):
            return pd.concat((a, b), sort=False)

    if isinstance(first_value, Table):
        restot = Table()

        def concat(a, b):
            return vstack([a, b])

    if isinstance(first_value, np.ndarray) or isinstance(first_value, np.recarray):
        restot = []

        def concat(a, b):
            if isinstance(a, list):
                return b
            else:
                return np.concatenate((a, b))

    if isinstance(first_value, int):
        restot = 0

        def concat(a, b):
            return operator.add(a, b)

    if isinstance(first_value, dict):
        restot = {}

        def concat(a, b):
            return dict(a, **b)

    if isinstance(first_value, list):
        restot = []

        def concat(a, b):
            return a+b

    if isinstance(first_value, tuple):
        restot = ([], [])
        tlength = len(first_value)
        restot = tuple([] for _ in range(tlength))

        def concat(a, b):
            bo = []
            for i in range(tlength):
                bo.append(a[i]+b[i])

            myres = tuple(bo[i] for _ in range(tlength))
            return myres

    if restot is None:
        print('Sorry to bother you but: unknown data type', type(first_value))
        print('Supported types', supported_types)
        return restot

    # gather the results
    for key, vals in resultdict.items():
        restot = concat(restot, vals)

    return restot
