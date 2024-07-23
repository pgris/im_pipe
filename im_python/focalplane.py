#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:47:15 2024

Author: julie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches

plt.rcParams.update({
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'axes.labelsize': 30,
    'figure.titlesize': 20,
    'legend.fontsize': 30,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'font.size': 20
})

class DataProcessor:
    def __init__(self, path, filename, run, suffix):
        self.path = path
        self.filename = filename
        self.num_run = run
        self.suffix = suffix
        self.df = None

    def load_data(self):
        Path_data = f'{self.path}{self.filename}{self.num_run}{self.suffix}.hdf5'
        print('Path data: ',Path_data)
        self.df = pd.read_hdf(Path_data)
        return self.df

def clean_data(df, parameter):
    if 'turnoff' in parameter:
        if 'turnoff' in df.columns:
            df[parameter[0:7]].replace(['outsider', 'undetermined'], 0, inplace=True)
            if parameter == 'turnoff':
                min_value = 50000
                max_value = 140000
            else : 
                min_value = 70000
                max_value = 180000
        if 'gain_quadra' in df.columns and 'turnoff_quadra' not in df.columns:
            if 'quadra' in parameter:
                df[parameter] = df['gain_quadra'] * df['turnoff']
        if 'gain_lin' in df.columns and 'turnoff_lin' not in df.columns:
            if 'lin' in parameter:
                df[parameter] = df['gain_lin'] * df['turnoff']
    if 'gain' in parameter:   
        if parameter not in df.columns:
            print('Difference between linear gain and quadratic gain')
            df[parameter] = df['gain_lin']-df['gain_quadra']
            min_value = -0.05
            max_value = 0.05
        else :
            min_value = 0
            max_value = 3
    return df, min_value, max_value
      
class FocalPlanePlotter:
    def __init__(self, df, min_value, max_value,parameter, faulty, sigma):
        self.df = df
        self.parameter = parameter
        self.min_value = min_value
        self.max_value = max_value
        self.Faulty = faulty
        self.sigma = sigma
    @staticmethod
    def safe_convert(value, multiplier):
        try:
            return int(value) + multiplier
        except ValueError:
            return value
        
    def ConvertDataframe(self):
            for col in self.merge_columns:
                if col in self.df.columns:
                    if isinstance(self.df[col].iloc[0], (np.ndarray, list)):
                        self.df[col] = self.df[col].apply(lambda x: ','.join(map(str, x)))
                    elif not pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_string_dtype(self.df[col]):
                        self.df[col] = self.df[col].astype(str)
                        
    def border(self):
        data = {
            'raft': ['R00', 'R00', 'R44', 'R44', 'R40', 'R40', 'R04', 'R04'],
            'sensor': ['SG0', 'SG1', 'SG0', 'SG1', 'SG0', 'SG1', 'SG0', 'SG1'],
            'new_sensor': ['S21', 'S12', 'S10', 'S01', 'S01', 'S12', 'S21', 'S10'],
        }
        self.df_file = pd.DataFrame(data)
        self.df['new_raft'] = self.df['raft'] + '_' + self.df['sensor']
        self.df_file['new_raft'] = self.df_file['raft'] + '_' + self.df_file['sensor']

    def update_border(self):
        for i, cond in enumerate(self.df_file['new_raft']):
            condition = self.df['new_raft'] == cond
            self.df.loc[condition, 'sensor'] = self.df_file['new_sensor'][i]

    def map_raft(self):
        self.df['num_col_raft'], self.df['num_line_raft'] = zip(*[(int(raft[2]), int(raft[1])) for raft in self.df['raft']])

    def map_sensor(self):
        self.df['num_col_sensor'] = [self.safe_convert(sensor[2], 3 * num_col) for sensor, num_col in zip(self.df['sensor'], self.df['num_col_raft'])]
        self.df['num_line_sensor'] = [self.safe_convert(sensor[1], 3 * num_line) for sensor, num_line in zip(self.df['sensor'], self.df['num_line_raft'])]

    def map_ampli(self):
        self.df = self.df.reset_index(drop=True)
        condition = self.df['ampli'] <= 8
        self.df['num_col_ampli'] = np.where(condition, self.df['ampli'] + 8 * self.df['num_col_sensor'], self.df['ampli'] - 8 + 8 * self.df['num_col_sensor'])
        self.df['num_line_ampli'] = np.where(condition, 2 * self.df['num_line_sensor'], 2 * self.df['num_line_sensor'] + 1)

    def filter_data(self):
        self.df_plot = self.df[(self.df[self.parameter] < self.max_value) & (self.df[self.parameter] > self.min_value)].copy()
    
    def bord(self, ax):
       #vertical line
       ax.axvline(x=0- 0.5, ymin=6/30, ymax=6*4/30, color='black', linewidth=2)  
       ax.axvline(x=8*3 - 0.5, ymin=0, ymax=6/30, color='black', linewidth=2)  
       ax.axvline(x=8*3 - 0.5, ymin=6*4/30, ymax=6*5/30, color='black', linewidth=2) 
       ax.axvline(x=8*12 - 0.5, ymin=0, ymax=6/30, color='black', linewidth=2)  
       ax.axvline(x=8*12 - 0.5, ymin=6*4/30, ymax=6*5/30, color='black', linewidth=2) 
       ax.axvline(x=8*15 - 0.5, ymin=6/30, ymax=6*4/30, color='black', linewidth=2)  
       ax.axvline(x=8*14 - 0.5, ymin=24/30, ymax=26/30, color='black', linewidth=2) 
       ax.axvline(x=8*13 - 0.5, ymin=26/30, ymax=28/30, color='black', linewidth=2)
       ax.axvline(x=8*13 - 0.5, ymin=2/30, ymax=4/30, color='black', linewidth=2) 
       ax.axvline(x=8*14 - 0.5, ymin=4/30, ymax=6/30, color='black', linewidth=2)
       ax.axvline(x=8 - 0.5, ymin=24/30, ymax=26/30, color='black', linewidth=2) 
       ax.axvline(x=8*2 - 0.5, ymin=26/30, ymax=28/30, color='black', linewidth=2)
       ax.axvline(x=8*2 - 0.5, ymin=2/30, ymax=4/30, color='black', linewidth=2) 
       ax.axvline(x=8 - 0.5, ymin=4/30, ymax=6/30, color='black', linewidth=2)
       #horizontal line
       ax.axhline(y=0 - 0.5, xmin=8*3/120, xmax=8*3*4/120, color='black', linewidth=2)  
       ax.axhline(y=6 - 0.5, xmin=0, xmax=8*3/120, color='black', linewidth=2)  
       ax.axhline(y=6 - 0.5, xmin=8*3*4/120, xmax=8*3*5/120, color='black', linewidth=2) 
       ax.axhline(y=24 - 0.5, xmin=0, xmax=8*3/120, color='black', linewidth=2)  
       ax.axhline(y=24 - 0.5, xmin=8*3*4/120, xmax=8*3*5/120, color='black', linewidth=2) 
       ax.axhline(y=30 - 0.5, xmin=8*3/120, xmax=8*3*4/120, color='black', linewidth=2) 
       ax.axhline(y=2 - 0.5, xmin=16/120, xmax=24/120, color='black', linewidth=2) 
       ax.axhline(y=4 - 0.5, xmin=8/120, xmax=16/120, color='black', linewidth=2) 
       ax.axhline(y=26 - 0.5, xmin=8/120, xmax=16/120, color='black', linewidth=2)
       ax.axhline(y=28 - 0.5, xmin=16/120, xmax=24/120, color='black', linewidth=2) 
       
       ax.axhline(y=2 - 0.5, xmin=24*4/120, xmax=8*13/120, color='black', linewidth=2) 
       ax.axhline(y=4 - 0.5, xmin=8*13/120, xmax=8*14/120, color='black', linewidth=2) 
       ax.axhline(y=26 - 0.5, xmin=8*13/120, xmax=8*14/120, color='black', linewidth=2)
       ax.axhline(y=28 - 0.5, xmin=8*12/120, xmax=8*13/120, color='black', linewidth=2) 
       
    
    def focalplane_faulty(self, ax):
        self.merge_columns = self.df.columns.tolist()
        self.ConvertDataframe()
        a = GainStudy(self.df, self.min_value, self.max_value, self.parameter, '13144' , self.sigma)
        data = pd.DataFrame()
        for i in ['ITL', 'e2v']:
            dd = a.FaultyCCDs(i)
            data = pd.concat([data, dd])

        self.df = self.df.merge(data, how='inner', on=self.merge_columns).copy()
               

        if f'{self.parameter}_ok' in data.columns:
            frame_condition = (data[f'{self.parameter}_ok'] == 1)
            for _, row in data[frame_condition].iterrows():
                rect = patches.Rectangle(
                    (row['num_col_ampli'] - 1.5, row['num_line_ampli'] - 0.5),
                    1, 1, linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)
        

    def focalplane(self, color='viridis'):
        self.border()
        self.update_border()
        self.map_raft()
        self.map_sensor()
        self.map_ampli()
        self.filter_data()
        self.color = color
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(False)
        
        
        grid = self.df_plot.pivot_table(index='num_line_ampli', columns='num_col_ampli', values=self.parameter)
        im = ax.imshow(grid, cmap=color, interpolation='none', aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(self.parameter)

        ax.invert_yaxis()
        ax.set_title('Plan Focal de la Caméra LSST')
        
        if self.Faulty != 'no':
            self.focalplane_faulty(ax)
        self.bord(ax)
        plt.tight_layout()
        plt.show()

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

class GainStudy:
    def __init__(self, file, min_value, max_value, parameter, run, sigma):
        self.ITL = ['R01', 'R02', 'R03', 'R10', 'R20', 'R41', 'R42', 'R43']
        self.e2v = ['R11', 'R12', 'R13', 'R14', 'R21', 'R22', 'R23', 
                    'R24', 'R30', 'R31', 'R32', 'R33', 'R34']
        self.df = file
        self.parameter = parameter
        self.run = run
        self.nb_sigma = sigma
        self.min_value = min_value
        self.max_value = max_value
    
    def FaultyCCDs(self, fabr):
        new_col = f'{self.parameter}_ok'
        
        self.fabr = self.ITL if fabr == 'ITL' else self.e2v
        self.dff = self.df[self.df['raft'].isin(self.fabr)].reset_index(drop=True)

        self.dff[new_col] = np.where((self.dff[self.parameter] < self.max_value) & (self.dff[self.parameter] > self.min_value), 0, 1).copy() 

        m = np.median(self.dff[self.parameter][self.dff[new_col] == 0])
        std = np.std(self.dff[self.parameter][self.dff[new_col] == 0])

        n, bins = np.histogram(self.dff[self.parameter], bins='auto')
        self.bin_centers = (bins[:-1] + bins[1:]) / 2

        popt, _ = curve_fit(gauss, self.bin_centers, n, p0=[np.max(n), m, std])
        self.amp, self.mu, self.sig = popt

        cond = self.nb_sigma * self.sig
        c = self.dff[self.parameter].between(self.mu - cond, self.mu + cond)
        self.dff.loc[c, new_col] = 0
        self.dff.loc[~c, new_col] = 1
        p = len(self.dff[self.dff[new_col] == 1]) / len(self.dff) * 100
        print(f'Number of faulty CCDs for {fabr} with {self.parameter}: '
              f'{len(self.dff[self.dff[new_col] == 1])} out of {len(self.dff)} or {np.round(p, 2)}%')
        return self.dff
    
    def histogramme(self, fabr, color):
        self.FaultyCCDs(fabr)
        
        plt.hist(self.dff[self.parameter], bins='auto', histtype='step', alpha=1,
                 linewidth=2, color=color, linestyle='dotted', edgecolor=color,
                 label=f'{fabr} : {np.round(self.mu, 2)} ± {np.round(self.sig, 2)}')
        plt.hist(self.dff[self.parameter], bins='auto', histtype='stepfilled', alpha=0.2,
                 linewidth=2, color=color, linestyle='dotted', edgecolor=color)
        plt.plot(self.bin_centers, gauss(self.bin_centers, self.amp, self.mu, self.sig), color=color)

        if 'gain' in self.parameter:
            self.xx = f'{self.parameter} [e-/ADU]'
        elif self.parameter.startswith('turnoff'):
            self.xx = f'{self.parameter} [ADU]'
            if len(self.parameter) > 8:
                self.xx = f'{self.parameter} [pe]'
                

        plt.xlim(self.mu - 8*self.sig, self.mu + 8*self.sig)
        plt.xlabel(self.xx)

    def plot_hist(self):
        self.histogramme(fabr='ITL', color='red')
        self.histogramme(fabr='e2v', color='blue')

        plt.ylabel('number of entries')
        plt.legend()
        plt.show()

