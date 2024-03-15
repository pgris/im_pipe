#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import os
from optparse import OptionParser
from ptc_5 import merge_data, process
import pandas as pd
import sys 
sys.path.append("/home/julie/fm_study/im_pipe/python")


parser = OptionParser(description='Script to estimate the PTC on FP images')

parser.add_option('--flat_empty', type=str,
                  default='/home/julie/fm_study/im_pipe/input/flat_13144_flat_empty.csv',
                  help='[%default]')

parser.add_option('--flat_ND', type=str,
                  default='/home/julie/fm_study/im_pipe/input/flat_13144_flat_ND.csv',
                  help='[%default]')

opts, args = parser.parse_args()

flat_empty = opts.flat_empty
flat_ND = opts.flat_ND


path = "/home/julie/Téléchargements/CCD2-20240307T134109Z-002/CCD2/"
filelist = os.listdir(path)
flat = [flat_empty,flat_ND]

data = merge_data(filelist,flat)

resa = pd.DataFrame()

for i,row in data.iterrows():
    flat0 = path+row['full_path_flat0']
    flat1 = path+row['full_path_flat1']
    res = process(flat0,flat1)
    resa = pd.concat([res,resa]).reset_index(drop=True)





