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


import matplotlib.pyplot as plt
import numpy as np
import bot_frame_op as bot
import pandas as pd


UnBias='1D'
#UnBias='2D'


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




# def varianceCalculation (flat0, flat1, h):
    
#     var = np.var(flat0-flat1)/2
#     mean = np.mean((flat0+flat1)/2)
#     varA = np.var(flat0)
#     varB = np.var(flat1)
#     stdAB = -0.5 * (2*var-varA-varB)

#     if h > 0:
#         dvarA = np.var((flat0+h)-flat1)
#         dvarB = np.var(flat0-(flat1+h))
#         dA = (dvarA-2*var)/h
#         dB = (dvarB-2*var)/h

#         vv = (dA**2)*varA + (dB**2)*varB + 2*dA*dB*stdAB
#     else :
#         vv=0
        
#     return var, mean, varA, varB, dvarA, dvarB, stdAB, vv

# def varianceCalculation(flat0, flat1, h):
#     var = np.var(flat0-flat1)
#     mean = np.mean((flat0+flat1)/2)
    
#     varA = np.var(flat0)
#     varB = np.var(flat1)

#     covAB = np.cov(flat0, flat1)[0][1]  

#     vv = np.sqrt(((var)**2)/42)

#     return var, mean, varA, varB, varA,varB, covAB, vv




def varianceCalculation (flat0, flat1, h):
    mean = np.mean((flat0+flat1)/2)
    
    var = np.var(flat0-flat1)
    
    varA = np.var(flat0)
    varB = np.var(flat1)
    stdAB = -0.5 * (var-varA-varB)

   
    dvarA = np.var(flat0-flat1-h)
    dvarB = np.var(flat0-flat1+h)
    dA = (dvarA-dvarB)/(2*h)

    vv = (dA**2)*var

    return var, mean, varA, varB, dvarA, dvarB, stdAB, vv

def plot_hist_ampli (flat0, flat1, p, name):
    plt.figure()
    plt.hist(flat0.flatten(), bins = 100, color='red', alpha = 0.6, label='flat0')
    plt.hist(flat1.flatten(), bins = 100, alpha = 0.6, color = 'blue', label='flat1')
    plt.hist(flat0.flatten()-flat1.flatten(), bins = 100, alpha = 0.6, color = 'green', label='flat0-flat1')
    plt.title(str(name)+' - ampli '+str(p))
    #plt.xlim((40000,60000))
    plt.legend()
    plt.show()
    return

def process(flat0, flat1) -> pd.DataFrame() : 
    """
    Process the data
        - Calculate mean value for each HDU (16) for the two flat
        - Calculate variance of the difference of the 2 flat divided by 2 
        (mean to estimate the ptc)

    Parameters
    ----------
    flat0/flat1 : pd.dataframe
        .fits file of a CCD for a flat (0 or 1)
        
    Returns
    -------
    parameters : pd.dataframe
        output : ['raft','type','sensor','flat0','flat1','ampli',
                  'mean','var', 'var0', 'var1','dvar0',
                  'dvar1', 'std01', 'var(var)']

    """
    ITL = ['R00', 'R01', 'R02', 'R03', 'R04', 'R10',
           'R20', 'R40', 'R41', 'R42', 'R43', 'R44']
    e2v = ['R11', 'R12', 'R13', 'R14', 'R21', 'R22', 'R23', 
           'R24', 'R30', 'R31', 'R32', 'R33', 'R34']
    
    file_name_flat0 =flat0.split('/')[-1]
    file_name_flat1 =flat1.split('/')[-1]
    raft = file_name_flat0.split('_')[4]
    sensor = file_name_flat0.split('_')[5].split('.')[0]
    file_list = [flat0, flat1]
    FileUnBias=bot.InFile(dirall=file_list[:],Slow=False,
                          verbose=False,Bias=UnBias)
    flat0_overscanned=SingleImageIR(FileUnBias.all_file[0])
    flat1_overscanned=SingleImageIR(FileUnBias.all_file[1])
    
    r = []
    for p in range (16):
        print(p) 
        if p <=7:
            ampli_flat0=flat0_overscanned[0:2002,512*(p):(513)*(p+1)]
            ampli_flat1=flat1_overscanned[0:2002,512*(p):(513)*(p+1)]
            # if p == 5:
            #     plot_hist_ampli(ampli_flat0, ampli_flat1, p, file_name_flat0)
        else :
            ampli_flat0=flat0_overscanned[2002:-1,512*(p-8):(513)*(p+1-8)]
            ampli_flat1=flat1_overscanned[2002:-1,512*(p-8):(513)*(p+1-8)]


        var, mean, var0, var1,dvar0, dvar1, std01, vv= varianceCalculation(ampli_flat0,
                                                              ampli_flat1,
                                                              h=10**-6)
        
        if raft in ITL : 
            ccd = 'ITL'
        else:
            ccd = 'e2v'
            
        r.append((raft,ccd,sensor,file_name_flat0,file_name_flat1,p+1, 
                  mean, var/2, var0,var1,dvar0, dvar1,std01, vv))
        
    
    res = pd.DataFrame(r ,columns=['raft','type','sensor','flat0','flat1','ampli',
                                   'mean','var', 'var0', 'var1','dvar0',
                                   'dvar1', 'std01', 'var(var)'])
    return res

