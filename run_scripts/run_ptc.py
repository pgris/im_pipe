#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:35:25 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser

parser = OptionParser(description='Script to estimate the PTC on FP images')

parser.add_option('--imList', type=str,
                  default='list_im.csv',
                  help='List of images to process[%default]')

opts, args = parser.parse_args()

imList = opts.imList

print(imList)
