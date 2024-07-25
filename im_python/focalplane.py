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
        """
        Returns
        -------
        pd.Dataframe()
            Load PTC adjustment result
        """
        Path_data = f'{self.path}{self.filename}{self.num_run}{self.suffix}.hdf5'
        print('Path data: ',Path_data)
        self.df = pd.read_hdf(Path_data)
        return self.df
  
class FocalPlanePlotter:
    def __init__(self, df,  parameter, sigma, faulty='no', 
                 min_value=None, max_value=None):
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
        
    def convertDataframe(self):
        """
        Transform the data :
            
        list, array -> str
        str (in column 'turnoff') -> 0

        """
        self.merge_columns = self.df.columns.tolist() 
        #Convert string into 0 for the 'turnoff' column
        self.df['turnoff'] = self.df['turnoff'].replace(to_replace=r'.*',
                                                        value=int(0), regex=True) 
  
        for col in self.merge_columns:
            #Convert array or list into string
            if isinstance(self.df[col].iloc[0], (np.ndarray, list)):
                self.df[col] = self.df[col].apply(lambda x: ','.join(map(str, x))) 
        return self.df
      
    def add_columns(self):
        """
        Add a column if the parameter is not a column of the dataframe
    
        Raises
        ------
        ValueError
            Impossible to create a new column, must change the name of the 
            parameter
        """
        if 'turnoff' in self.parameter:
            if 'lin' in self.parameter and 'quadra' in self.parameter:
                raise ValueError('Impossible: "lin" and "quadra" cannot be '
                                 'combined with "turnoff" simultaneously.')
                
            #Determines the gain-corrected turnoff calculated from the quadratic fit
            if 'quadra' in self.parameter:
                self.parameter = 'turnoff_quadra'
                self.df[self.parameter] = self.df['gain_quadra'] * self.df['turnoff']
                
            #Determines the gain-corrected turnoff calculated from the linear fit
            if 'lin' in self.parameter:
                self.parameter = 'turnoff_lin'
                self.df[self.parameter] = self.df['gain_lin'] * self.df['turnoff']
        
        elif ('lin' in self.parameter and 'quadra' in self.parameter) and (
                'gain' not in self.parameter):
            raise ValueError('If you want the difference in "gain",'
                             ' you must include the term "gain".')
        
        elif 'gain' in self.parameter:
            if self.parameter not in self.df.columns:
                # Change the name of the parameter if it does not match a column name
                if 'quadra' in self.parameter and 'lin' not in self.parameter:
                    self.parameter = 'gain_quadra'
                elif 'lin' in self.parameter and 'quadra' not in self.parameter:
                    self.parameter = 'gain_lin'
                #Create a new column : the difference between the two gains
                elif 'lin' in self.parameter and 'quadra' in self.parameter:
                    print('Difference between linear gain and quadratic gain')
                    self.parameter = 'gain_lin_quadra'
                    self.df[self.parameter] = self.df['gain_lin'] - self.df['gain_quadra']
        
        return self.df, self.parameter

        
    def scale_limit_max(self):
        """
        Define the maximum limit for the focal plane plot based on the parameter.
        """
        if 'turnoff' in self.parameter:
            if self.parameter == 'turnoff':
                self.max_value = 150000
            else:
                self.max_value = 180000
    
        elif 'gain' in self.parameter:
            if 'lin' in self.parameter and 'quadra' in self.parameter:
                self.max_value = 0.05
            else:
                self.max_value = 3
    
    def scale_limit_min(self):
        """
        Define the minimum limit for the focal plane plot based on the parameter.
        """
        if 'turnoff' in self.parameter:
            if self.parameter == 'turnoff':
                self.min_value = 50000
            else:
                self.min_value = 70000
    
        elif 'gain' in self.parameter:
            if 'lin' in self.parameter and 'quadra' in self.parameter:
                self.min_value = -0.05
            else:
                self.min_value = 0

   
    
    def border(self):
        """
        Update CCD names for preparation of the plot.

        """
        #CCDs for which we need to change the name
        data = {
            'raft': ['R00', 'R00', 'R44', 'R44', 'R40', 'R40', 'R04', 'R04'],
            'sensor': ['SG0', 'SG1', 'SG0', 'SG1', 'SG0', 'SG1', 'SG0', 'SG1'],
            'new_sensor': ['S21', 'S12', 'S10', 'S01', 'S01', 'S12', 'S21', 'S10'],
        }
        self.df_file = pd.DataFrame(data)
        
        
        # Create a new column 'new_raft' in both DataFrames by combining
        #'raft' and 'sensor' 
        self.df['new_raft'] = self.df['raft'] + '_' + self.df['sensor']
        self.df_file['new_raft'] = self.df_file['raft'] + '_' + self.df_file['sensor']
        
        #Update the name
        for i, cond in enumerate(self.df_file['new_raft']):
            condition = self.df['new_raft'] == cond
            self.df.loc[condition, 'sensor'] = self.df_file['new_sensor'][i]

    def map_raft(self):
        """
        Create a mapping for 'raft' by defining a grid

        """
        self.df['num_col_raft'], self.df['num_line_raft'] = zip(
            *[(int(raft[2]), int(raft[1])) for raft in self.df['raft']])

    def map_sensor(self):
        """
        Create a mapping for 'sensor' by defining a grid
        
        """
        self.df['num_col_sensor'] = [self.safe_convert(
            sensor[2], 3 * num_col) for sensor, num_col in zip(
                self.df['sensor'], self.df['num_col_raft'])]
                
        self.df['num_line_sensor'] = [self.safe_convert(
            sensor[1], 3 * num_line) for sensor, num_line in zip(
                self.df['sensor'], self.df['num_line_raft'])]

    def map_ampli(self):
        """
        Create a mapping for 'ampli' by defining a grid

        """
        self.df = self.df.reset_index(drop=True)
        condition = self.df['ampli'] <= 8
        self.df['num_col_ampli'] = np.where(condition,
                                            self.df['ampli'] + 8 * 
                                            self.df['num_col_sensor'],
                                            self.df['ampli'] - 8 + 8 *
                                            self.df['num_col_sensor'])
        self.df['num_line_ampli'] = np.where(condition, 
                                             2 * self.df['num_line_sensor'],
                                             2 * self.df['num_line_sensor'] + 1)

    def filter_data(self):
        """
        Filter the data that are out of bounds for the focal plane plot
        bounds : [min_value, max_value]
        

        """
        self.df_plot = self.df[(self.df[self.parameter] <= self.max_value) & (
            self.df[self.parameter] >= self.min_value)].copy()
    
    def bord(self, ax):
        """
        Draw the contours of the focal plane.
        The positions are determined by the grid created by the ampli.

        """
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
        """
        Highlight the amplifiers considered as faulty

        """
        #GainStudy determine the faulty CCD
        a = GainStudy(self.df, self.parameter, self.sigma)
        data = pd.DataFrame()
        for i in ['ITL', 'e2v']:
            dd = a.faultyCCDs(i)
            data = pd.concat([data, dd])
        
        self.df = self.df.merge(data, how='inner',on=self.merge_columns).copy()
               
        #A new column is added, if == 1 : faulty, else == 0 : good
        if f'{self.parameter}_ok' in data.columns:
            frame_condition = (data[f'{self.parameter}_ok'] == 1)
            for _, row in data[frame_condition].iterrows():
                rect = patches.Rectangle(
                    (row['num_col_ampli'] - 1.5, row['num_line_ampli'] - 0.5),
                    1, 1, linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)
        
    
    def focalplane(self, color='viridis'):
        """
        Plot the focal plane.
    
        Parameters
        ----------
        color : str, optional
            The colormap to be used for plotting
        """
        # Convert data to be usable
        self.convertDataframe()
        # Add columns if necessary
        self.add_columns()
        
        # Determine limits if not specified
        if self.min_value is None:
            self.scale_limit_min()
        if self.max_value is None:
            self.scale_limit_max()
        
        # Prepare the focal plane plot
        self.border()
        self.map_raft()
        self.map_sensor()
        self.map_ampli()
        self.filter_data()
        self.color = color
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(False)

        grid = self.df_plot.pivot_table(index='num_line_ampli',
                                        columns='num_col_ampli', 
                                        values=self.parameter)
        
        im = ax.imshow(grid, cmap=color, interpolation='none', aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(self.parameter)

        ax.invert_yaxis()
        ax.set_title('Focal Plane of the LSST Camera')
        
        # Highlight faulty CCDs if specified
        if self.Faulty != 'no':
            self.focalplane_faulty(ax)
            
        self.bord(ax)
        plt.tight_layout()
        plt.show()


def gauss(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

class GainStudy:
    def __init__(self, file, parameter, sigma):
        self.ITL = ['R01', 'R02', 'R03', 'R10', 'R20', 'R41', 'R42', 'R43']
        self.e2v = ['R11', 'R12', 'R13', 'R14', 'R21', 'R22', 'R23', 
                    'R24', 'R30', 'R31', 'R32', 'R33', 'R34']
        self.df = file
        self.parameter = parameter
        self.nb_sigma = sigma
    
    def faultyCCDs(self, fabr):
        """
        Determine the faulty ccd

        Parameters
        ----------
        fabr : str
            Manufacturer name ('ITL' or 'e2v')

        """
        new_col = f'{self.parameter}_ok'
        
        #Convert dataframe and add column if needed
        plotter = FocalPlanePlotter(self.df, self.parameter, self.nb_sigma)
        self.df = plotter.convertDataframe()
        if self.parameter not in self.df.columns:
            self.df, self.parameter = plotter.add_columns()
        
        self.fabr = self.ITL if fabr == 'ITL' else self.e2v
        self.dff = self.df[self.df['raft'].isin(self.fabr)].reset_index(drop=True)
        
        #Define the min and max threshold based on the parameter
        if 'gain' in self.parameter:
            if 'lin' in self.parameter and 'quadra' in self.parameter:
                mini = -0.05
                maxi = 0.05
            else :
                mini = 0
                maxi = 3
        if 'turnoff' in self.parameter:
            mini = 40000
            maxi = 200000
        
        #Report defective ccd within defined limits
        self.dff[new_col] = np.where(
                                (self.dff[self.parameter] < maxi) &
                                (self.dff[self.parameter] > mini), 0, 1).copy() 
        
        #Fit a gaussian on the distribution
        m = np.median(self.dff[self.parameter][self.dff[new_col] == 0])
        std = np.std(self.dff[self.parameter][self.dff[new_col] == 0])

        n, bins = np.histogram(self.dff[self.parameter], bins='auto')
        self.bin_centers = (bins[:-1] + bins[1:]) / 2

        popt, _ = curve_fit(gauss, self.bin_centers, n, p0=[np.max(n), m, std])
        self.amp, self.mu, self.sig = popt
        
        #Dtermine the faulty CCDs
        cond = self.nb_sigma * self.sig
        c = self.dff[self.parameter].between(self.mu - cond, self.mu + cond)
        self.dff.loc[c, new_col] = 0
        self.dff.loc[~c, new_col] = 1
        p = len(self.dff[self.dff[new_col] == 1]) / len(self.dff) * 100
        print(f'Number of faulty CCDs for {fabr} with {self.parameter}: '
        f'{len(self.dff[self.dff[new_col] == 1])} out of {len(self.dff)} or '
        f'{np.round(p, 2)}%')

        return self.dff
    
    def histogramme(self, fabr, color):
        """
        Plot a histogram of the parameter for a given manufacturer 
        and a gaussian fit.        
        
        Parameters
        ----------
        fabr : str
            Manufacturer name ('ITL' or 'e2v')
        color : str
            Color for the histogram

        """
        #Determined the faulty CCD
        self.faultyCCDs(fabr)
        
        #Plot the histogram
        plt.hist(self.dff[self.parameter], bins='auto', histtype='step', 
                 alpha=1, linewidth=2, color=color, linestyle='dotted',
                 edgecolor=color, label=f'{fabr} : {np.round(self.mu, 2)}'
                 ' ± {np.round(self.sig, 2)}')
        
        plt.hist(self.dff[self.parameter], bins='auto', histtype='stepfilled', 
                 alpha=0.2,linewidth=2, color=color, linestyle='dotted',
                 edgecolor=color)
        #Plot the fitted Gaussian curve
        ll = np.linspace(self.mu-10*self.sig, self.mu+10*self.sig, 300)
        plt.plot(ll, gauss(ll, self.amp, self.mu, self.sig), color=color)
        
        #Set x_label based on the parameter
        if 'gain' in self.parameter:
            self.xx = f'{self.parameter} [e-/ADU]'
        elif 'turnoff' in self.parameter:
            self.xx = f'{self.parameter} [ADU]'
            if len(self.parameter) > 8:
                self.xx = f'{self.parameter} [pe]'
            
        plt.xlim(self.mu - 8*self.sig, self.mu + 8*self.sig)
        plt.xlabel(self.xx)

    def plot_hist(self):
        """
        Plot the histograms for the parameter from the two manufacturers ('ITL' and 'e2v')
        
        """
        self.histogramme(fabr='ITL', color='red')
        self.histogramme(fabr='e2v', color='blue')

        plt.ylabel('number of entries')
        plt.legend()
        plt.show()
