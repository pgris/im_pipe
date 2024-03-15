#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:10:58 2024

@author: julie
"""

# Analyse: généralités

# Tasks to be performed on flat runs:
# * identify data path for run 13144
# * display the 16 amplifiers of a sensor/raft
# * overscan and bias correction
# * gain correction
# * display different illuminations

# *******************

# * Per set of data
#     * Identify the biasses, and the level of illumination
# * Per exposure / files / amps
#    * open the file
#    * correct for overscan and bias 
  
   

# From P. Antilogus:
# * /sps/lsst/users/antilog/web/bot/spot/Xtalk_Spot.ipynb
# * bot_frame_op.py

# From P. Astier:
# * /sps/lsst/users/astier/slac/13144/gains.list 
# * Cod.py

# # NP (from P. Antilogus): Pour les données à differents flux : /sps/lsst/groups/FocalPlane/SLAC/run5/13144  c’est un run PTC
# * le bas flux à des filtres neutres pour reduire le flux pour un temps de pose donnée … ci-joint la syntaxe des noms de directory :
#   * flat_ND_OD0.5_SDSSi_492.0_flat0_351
#       * ND_OD0.5: neutral density filter 0.5   ( l’autre solution : empty = pas de filtre )
#       * SDSSi   : filtre SDSS i
#       * 492.0   : flux de 492 e-   , le flux le plus bas est donné à 50e- …mais l’éclairement est non uniforme sur les bord c’est bien moins
#       * flat0   : première pose d’une série de 2 ( il y aussi donc le flat1 )
#       * 351     : 351 ieme pose du run
#       # Analyse: généralités

# Tasks to be performed on flat runs:
# * identify data path for run 13144
# * display the 16 amplifiers of a sensor/raft
# * overscan and bias correction
# * gain correction
# * display different illuminations


import astropy.io.fits as pyfits
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import linregress
import json
import bot_frame_op as bot
import pandas as pd


UnBias='1D'
#UnBias='2D'

path = "/home/julie/stage2/notebook/"


#chemin du dossier contenant les données
path = '/home/julie/stage2/data/CCD_test/'
path = '/home/julie/Téléchargements/CCD2-20240307T134109Z-002/CCD2/'
filelist = os.listdir(path) #liste des fichiers sur mon ordi






def merge_data(doss, refNames) -> pd.DataFrame:
    """
    Function to get (flat0,flat1) pairs

    Parameters
    ----------
    doss : list(str)
        List of files on disk.
    refNames : list(str)
        list of reference files

    Returns
    -------
    df_merge : pandas df
        Output data (columns='full_path_flat0',
                             'full_path_flat1', 
                             'fileName_flat0', 'fileName_flat1')

    """

    ref_flat = pd.DataFrame()

    for name in refNames:
        dd = pd.read_csv(name, comment='#')
        ref_flat = pd.concat((ref_flat, dd))
        
    df_data = pd.DataFrame(doss, columns=['full_path'])
    df_data['fileName'] = df_data['full_path'].str.split('/').str.get(-1)
    
    
    df_data['raft'] = df_data['fileName'].str.split('_').str.get(4)
  
    
    df_data['sensor'] = df_data['fileName'].str.split('_').str.get(5).str.split('.').str.get(0)
   
    
    df_merge = df_data.merge(ref_flat, left_on=['fileName'], right_on=[
                             'file_flat0'], suffixes=['', ''])
    
    df_merge = df_merge.merge(df_data, left_on=['file_flat1'], right_on=[
                              'fileName'], suffixes=['_flat0', '_flat1'])

    df_merge = df_merge[['full_path_flat0',
                         'full_path_flat1','fileName_flat0','fileName_flat1']]
    df_merge['raft']=df_data['raft']
    df_merge['sensor']=df_data['sensor']
    
    return df_merge



def parse_section(section_string) :
    # input : a fits-like section string (as in DATASEC '[y0:y1,x0:x1]')
    # output :  image section coordinate to be used in python table:  ymin , ymax ,xmin, xmax
    #
    r=section_string[1:-1].split(',')
    x=list(map(int,r[0].split(':')))
    y=list(map(int,r[1].split(':')))
    # put into pythonic way
    if y[0]<=y[1]:
        y[0] = y[0]-1
    else:
        y[1] = y[1]-1
    if x[0]<=x[1]:
        x[0] = x[0]-1
    else:
        x[1] = x[1]-1
    
    return y[0],y[1],x[0],x[1]

def SingleImageIR(actfile,gains=None):
        first_line,first_lover,first_col,first_cover=parse_section(actfile.Datasec[0])
        col_size=first_cover-first_col
        line_size=first_lover-first_line
        #
        spf=np.zeros((line_size*2,col_size*8))

        for i in range(16) :
            y1,y2,x1,x2=parse_section(actfile.Datasec[i])
            yd1,yd2,xd1,xd2=parse_section(actfile.Detsec[i])
            xdir,ydir=(1,1)
            if yd2<yd1:
                ydir=-1
                (yd2,yd1)=(yd1,yd2)
            if xd2<xd1:
                xdir=-1
                (xd2,xd1)=(xd1,xd2)
            if gains is not None:
                raft_ccd=actfile.raftbay+'_'+actfile.ccdslot
                amp='C'+actfile.Extname[i][-2:]
                spf[yd1:yd2,xd1:xd2]=actfile.Image[i][y1:y2,x1:x2][::ydir,::xdir] * gains[raft_ccd][amp]
            else:
                spf[yd1:yd2,xd1:xd2]=actfile.Image[i][y1:y2,x1:x2][::ydir,::xdir]
        return spf
                
        for i in range(16) :
            if i<8 :
                xx=i*col_size-1
                yy=0
                for x in range(first_col,first_cover) : 
                    spf[yy:yy+line_size,xx+col_size-(x-first_col)]=actfile.Image[i][first_line:first_lover,x]
            else :
                xx=(15-i)*col_size
                yy=-1
                for y in range(first_line,first_lover) :  
                    spf[yy+2*line_size-(y-first_line),xx:xx+col_size]=actfile.Image[i][y,first_col:first_cover]
                    
        return spf


def process(flat0, flat1):
    
    file_name_flat0 =flat0.split('/')[-1]
    file_name_flat1 =flat1.split('/')[-1]
    raft = file_name_flat0.split('_')[4]
    sensor = file_name_flat0.split('_')[5].split('.')[0]
    file_list = [flat0, flat1]
    print(file_list)
    FileUnBias=bot.InFile(dirall=file_list[:],Slow=False,verbose=False,
                          Bias=UnBias)
    print(FileUnBias.all_file)
    flat0_overscanned=SingleImageIR(FileUnBias.all_file[0])
    flat1_overscanned=SingleImageIR(FileUnBias.all_file[1])
    
    r = []
    
    mean =np.mean((flat0_overscanned+flat1_overscanned)/2)
    std =np.var(flat0_overscanned-flat1_overscanned)/2

    r.append((raft,sensor,file_name_flat0,file_name_flat1,99,mean,std))
    for p in range (16):
            
        if p <=7:
            ampli_flat0=flat0_overscanned[0:2002,512*(p):(513)*(p+1)]
            ampli_flat1=flat1_overscanned[0:2002,512*(p):(513)*(p+1)]
        else :
            ampli_flat0=flat0_overscanned[2002:-1,512*(p-8):(513)*(p+1-8)]
            ampli_flat1=flat1_overscanned[2002:-1,512*(p-8):(513)*(p+1-8)]

        std = np.var(ampli_flat0-ampli_flat1)/2
        mean = np.mean((ampli_flat0+ampli_flat1)/2)
        
        r.append((raft,sensor,file_name_flat0,file_name_flat1,p+1, 
                  mean, std))
 
    res = pd.DataFrame(r ,columns=['raft','sensor','flat0','flat1','ampli',
                                   'mean','var'])
    return res


def fit_lin (var,mean) :

    mean = mean.to_numpy()
    var = var.to_numpy()
    diff_var_=[]
    mean_=[]   
    for i in range (len(mean)):
        if  mean[i] < 5000:
            diff_var_.append(var[i])
            mean_.append(mean[i])
    a, b, r, p_value, std_err = linregress(mean_, diff_var_)
    return a,b


def fit_quadra (var,mean) :
    mean = mean.to_numpy()
    var = var.to_numpy()
    mean_max = 60000
    x_data=[]
    y_data=[]
    for i in range (len(mean)):
        if  mean[i] < mean_max:
            y_data.append(var[i])
            x_data.append(mean[i])
    y_params = np.polyfit(x_data, y_data, 2)  #équation de degré 2
    return y_params

def fit (tab) -> pd.DataFrame() : 
    """
    Make a quadratic fit of the data and calculate the turnoff

    Parameters
    ----------
    tab : pd.dataframe
        Mean and variance for each ADU/run/raft/sensor 
        
    Returns
    -------
    parameters : pd.dataframe
        output : 'raft','sensor','ampli','a','b','c','gain','turnoff'
                                              

    """
    c = []
    raft = tab['raft'].unique()
    sensor = tab['sensor'].unique()
    for r in raft:
        for s in sensor:
            for p in range(1,17):
                idx = tab[(tab['ampli']==p)&
                          (tab['raft']==r)&
                          (tab['sensor']==s)].index
                if not list(idx):
                    continue
                else:
                    mean = tab['mean'][idx].reset_index(drop=True)
                    var = tab['var'][idx].reset_index(drop=True)
                    a,b = fit_lin(var, mean)
                    y_params = fit_quadra(var, mean)
                    max_var = np.max(var)
                    turnoff_idx = var[var==max_var].index
                    turnoff = mean[turnoff_idx[0]]
                
                    c.append((r,s,p,
                              y_params[0],y_params[1],y_params[2],1/y_params[1],turnoff))
            
    parameters = pd.DataFrame(c ,columns=['raft','sensor','ampli',
                                          'a','b','c','gain','turnoff'])
                                          
    return parameters



def plot_ampli (data, parameters) -> plt.figure :
    """
    Plot the ptc for each run/raft/sensor for the 16 HDU

    Parameters
    ----------
    file_name : 
    data : pd.dataframe
        Mean and variance for each ADU/run/raft/sensor 
        
    parameters : pd.dataframe
        parameters of fit + turnoff
        
    Returns
    -------
    None.

    """
    l=np.linspace(0,60000,50)
    xx = 60000/2
    yy=100
    legend = "HDU {} \nGain = {}\nTurnoff = {}"
    raft = parameters['raft'].unique()
    sensor = parameters['sensor'].unique()
    for r in raft:
        for s in sensor:
            idxc = data[(data['raft']==r)&(data['sensor']==s)].index
            if not list(idxc):
                continue
            else:
                fig, axs = plt.subplots(4,4,sharex=True,figsize=(18,9))
                for i in range (16):
                    irow = int(i/4)
                    icol = i%4
                    idx = data[(data['ampli']==i+1)&
                               (data['raft']==r)&
                               (data['sensor']==s)].index
                    mean = data['mean'][idx] 
                    var = data['var'][idx]
                    idx = parameters[(parameters['ampli']==i+1)&
                                     (parameters['raft']==r)&
                                     (parameters['sensor']==s)].index
                    a = parameters['a'][idx].values[0]
                    b = parameters['b'][idx].values[0]
                    c = parameters['c'][idx].values[0]
                    turnoff = parameters['turnoff'][idx].values[0]
                    gain = parameters['gain'][idx].values[0]
                    axs[irow, icol].plot(mean,var,'.')
                    axs[irow, icol].plot(l,a*l**2+b*l+c)
                    axs[irow,icol].text(xx,yy,legend.
                                        format(i+1, np.round(gain,2),
                                               np.round(turnoff,4)),
                                        fontsize=12)
          
                fig.suptitle('PTC for '+r+s+ 'run13144 : '+str(len(raft)-1)+
                             ' illuminations (unbias '+str(UnBias)+')')
                fig.supxlabel('mean signal (ADU)')
                fig.supylabel('var ($ADU^{2}$)')
                fig.show()
    return 


# path = '/home/julie/stage2/data/CCD_test/'
# filelist = os.listdir(path)

# path = '/home/julie/fm_study/im_pipe/input/'
# flat_ND = 'flat_13144_flat_ND.csv'  
# flat_empty = 'flat_13144_flat_empty.csv'
# flat = [path+flat_empty,path+flat_ND]
# path = "/home/julie/Téléchargements/CCD2-20240307T134109Z-002/CCD2/"


# df = merge_data(filelist, flat)
# resa = pd.DataFrame()

# raft = df['raft'].unique()
# sensor = df['sensor'].unique()



# for i,row in df.iterrows():
#     flat0 = path+row['full_path_flat0']
#     flat1 = path+row['full_path_flat1']
#     res  = process(flat0,flat1)
#     resa = pd.concat([res,resa]).reset_index(drop=True)

# param_fit=fit(resa)
# plot_ampli(resa, param_fit)



# print(resa)
# print(param_fit)






































    
    


