#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:27:55 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from io_tools import get_runList

parser = OptionParser(description='Script to estimate the PTC on FP images')

parser.add_option('--dataDir', type=str,
                  default='/sps/lsst/groups/FocalPlane/SLAC/run5',
                  help='Path to data [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir

# get the runList
runList = get_runList('{}/*'.format(dataDir))


print(runList)
print('Nruns:', len(runList))
