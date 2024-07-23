#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:25:36 2024

@author: julie
"""

from optparse import OptionParser
from focalplane import FocalPlanePlotter, DataProcessor, clean_data, GainStudy

parser = OptionParser(description='Script to plot the focal plane')

parser.add_option('--path', type=str, 
                  default='../fit_resu/', 
                  help='Path to data [%default]')

parser.add_option('--filename', type=str, 
                  default='fit_resu_', 
                  help='file name [%default]')

parser.add_option('--run', type=str, 
                  default='13144', 
                  help='num run : [%default]')

parser.add_option('--parameter', type=str, 
                  default='gain_lin', 
                  help='parameter to plot [%default]')

parser.add_option('--faulty', type=str, 
                  default='no', 
                  help='Show faulty CCDs ? [%default]')

parser.add_option('--suffix', type=str, 
                  default='', 
                  help='suffix file name : [%default]')

parser.add_option('--min_value', type=float, 
                  default=None, 
                  help='Minimum value for parameter filtering [%default]')

parser.add_option('--max_value', type=float, 
                  default=None, 
                  help='Maximum value for parameter filtering [%default]')

parser.add_option('--color', type=str, 
                  default='viridis', 
                  help='Scale color [%default]')

parser.add_option('--sigma', type=float, 
                  default=3, 
                  help='Sigma value [%default]')


opts, args = parser.parse_args()
opts = vars(opts)
c = opts['color']

# Process data
processor_opts = {k: opts[k] for k in ('path', 'filename', 'run', 'suffix')}
processor = DataProcessor(**processor_opts)
df = processor.load_data()

# Plot data
if opts['min_value'] or opts['max_value'] != None:
    df,_,_ = clean_data(df, opts['parameter'])
    plotter_opts = {k: opts[k] for k in ('parameter', 'faulty', 'sigma', 'min_value', 'max_value')}
    plotter = FocalPlanePlotter(df, **plotter_opts)
else:    
    df, min_value, max_value = clean_data(df, opts['parameter'])
    plotter_opts = {k: opts[k] for k in ('parameter', 'faulty', 'sigma')}
    plotter = FocalPlanePlotter(df, min_value, max_value, **plotter_opts)
    

plotter.focalplane(color = str(c))

