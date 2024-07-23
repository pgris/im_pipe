#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:18:02 2024

@author: julie
"""

from optparse import OptionParser
from focalplane import GainStudy, DataProcessor, clean_data

# Création du parser d'options
parser = OptionParser(description='Script pour plot la distribution d\'un paramètre et déduire les CCD faulty')

parser.add_option('--path', type=str, 
                  default='../fit_resu/', 
                  help='Path to data [%default]')

parser.add_option('--filename', type=str, 
                  default='fit_resu_', 
                  help='File name [%default]')

parser.add_option('--run', type=str, 
                  default='13144', 
                  help='Num run : [%default]')

parser.add_option('--parameter', type=str, 
                  default='gain_lin', 
                  help='Parameter to plot [%default]')

parser.add_option('--suffix', type=str, 
                  default='', 
                  help='Suffix file name : [%default]')

parser.add_option('--sigma', type=float, 
                  default=3, 
                  help='Sigma value [%default]')

parser.add_option('--hist', type=str, 
                  default='yes', 
                  help='Plot histogram? [%default]')

parser.add_option('--max_value', type=float, 
                  default=None, 
                  help='Maximum value for parameter [%default]')

parser.add_option('--min_value', type=float, 
                  default=None, 
                  help='Minimum value for parameter [%default]')

opts, args = parser.parse_args()
opts = vars(opts)
hist = opts['hist']

# Process data
processor_opts = {k: opts[k] for k in ('path', 'filename', 'run', 'suffix')}
processor = DataProcessor(**processor_opts)
df = processor.load_data()

# Plot data
if opts['min_value'] and opts['max_value'] != None:
    df,_,_ = clean_data(df, opts['parameter'])
    plotter_opts = {k: opts[k] for k in ('min_value', 'max_value', 'parameter', 'run', 'sigma')}
    gain_study_instance = GainStudy(df, **param_opts)
else:    
    df, min_value, max_value = clean_data(df, opts['parameter'])
    plotter_opts = {k: opts[k] for k in ('parameter', 'run', 'sigma')}
    gain_study_instance = GainStudy(df, min_value, max_value, **plotter_opts)
    
# Appeler la méthode histogramme
if hist == 'yes':
    gain_study_instance.plot_hist() 
else :
    for i in ['ITL', 'e2v']:
        gain_study_instance.FaultyCCDs(fabr=i)
        

