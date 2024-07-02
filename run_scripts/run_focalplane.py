#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:58:08 2024

@author: julie
"""

import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser
from planfocal import focalplane

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
    description='Script to plot the focal plane')

parser.add_option('--path', type=str,
                  default='../fit_resu/',
                  help='Path to data [%default]')

parser.add_option('--filename', type=str,
                  default='fit_resu_',
                  help='file name [%default]')

parser.add_option('--num_run', type=str,
                  default='13144',
                  help='num run : [%default]')

parser.add_option('--parameter', type=str,
                  default='gain_lin',
                  help='parameter to plot : [%default]')

opts, args = parser.parse_args()

num_run = opts.num_run
path = opts.path
filename = opts.filename
param = opts.parameter


Path_data = '{}{}{}.hdf5'.format(path, filename, num_run)

df = pd.read_hdf(str(Path_data))

focalplane(df, param)
