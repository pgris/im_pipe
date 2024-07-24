import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# # Lire les données à partir du fichier HDF5
# #f = pd.read_hdf('/home/julie/fm_study/im_pipe/fit_resu/fit_resu_13144.hdf5')

# df_file = pd.read_csv('/home/julie/Téléchargements/Raft.csv')

# df = pd.read_hdf('/home/julie/fm_study/im_pipe/fit_resu/fit_resu_13144.hdf5')

    




def focalplane (df, param):
    data = {
        'raft': ['R00','R00','R44','R44', 'R40', 'R40', 'R04', 'R04'],
        'sensor': ['SG0', 'SG1','SG0', 'SG1','SG0', 'SG1','SG0', 'SG1'],
        'new_sensor': ['S21','S12','S10','S01','S01','S12','S21','S10'],
    }

    df_file = pd.DataFrame(data)

    df['new_raft'] = df['raft']+'_'+df['sensor']
    df_file['new_raft'] = df_file['raft']+'_'+df_file['sensor']
    
    for i, cond in enumerate (df_file['new_raft']):
        condition = df['new_raft'] == cond
        df.loc[condition, 'sensor'] = df_file['new_sensor'][i]
    
    # Décomposer le numéro de raft pour définir les colonnes/lignes
    df['num_col_raft'] = [int(raft[2]) for raft in df['raft']]
    df['num_line_raft'] = [int(raft[1]) for raft in df['raft']]
    
    
    def safe_convert(value, multiplier):
        try:
            return int(value) + multiplier
        except ValueError:
            return value
    
    
    
    df['num_col_sensor'] = [safe_convert(sensor[2], 3 * num_col) for sensor, num_col in zip(df['sensor'], df['num_col_raft'])]
    df['num_line_sensor'] = [safe_convert(sensor[1], 3 * num_line) for sensor, num_line in zip(df['sensor'], df['num_line_raft'])]
    
    
    df = df.reset_index(drop=True)
    
    condition = df['ampli']<=8
    df.loc[condition, 'num_col_ampli']= df['ampli'] + 8*df['num_col_sensor']
    df.loc[~condition, 'num_col_ampli']= df['ampli']-8 + 8*df['num_col_sensor']
    df.loc[condition, 'num_line_ampli'] = 2*df.num_line_sensor
    df.loc[~condition, 'num_line_ampli'] = 2*df.num_line_sensor+1

    
    if param == 'gain_lin':
        df = df[(df['gain_lin'] < 5)&(df['gain_lin']>0)]
    elif param == 'gain_quadra':
        df = df[(df['gain_quadra'] < 5)&(df['gain_quadra']>0)]
    elif param == 'turnoff':
        df = df[df['turnoff'] != 'undetermined']
        df['turnoff'] = df['turnoff'].astype(float)
    
    
    grid = df.pivot_table(index='num_line_ampli', columns='num_col_ampli', values=param)
    
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(False)
    
    im = ax.imshow(grid, cmap='viridis', interpolation='none', aspect='auto')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(param)
    
    
    
    #Inverser l'axe des y
    ax.invert_yaxis()
    
    ax.set_title('Plan Focal de la Caméra LSST')
    plt.tight_layout()
    plt.show()
    return
