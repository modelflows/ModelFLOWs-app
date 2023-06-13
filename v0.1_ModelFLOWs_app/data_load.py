import hdf5storage
import numpy as np
import pickle
import pandas as pd
import h5py
import os 
from sys import platform

def load_data(format, wantedfile):
    # LOAD MATLAB .mat FILES
    if format == 'mat':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        elif platform == "darwin":
            # OS X
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        return Tensor

#-----------------------------------------------------------------------

    # LOAD NUMPY .npy FILES
    elif format == 'npy':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor = np.load(f'{wantedfile}')
        elif platform == "darwin":
            # OS X
            Tensor = np.load(f'{wantedfile}')
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor = np.load(f'{wantedfile}')
        return Tensor

#-----------------------------------------------------------------------

    # LOAD .csv FILES
    elif format == 'csv':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        elif platform == "darwin":
            # OS X
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        return Tensor

#-----------------------------------------------------------------------
  
    # LOAD .h5 FILES
    elif format == 'h5':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        elif platform == "darwin":
            # OS X
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        elif platform in ["win32", "win64"]:
            # Windows
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        return Tensor

#-----------------------------------------------------------------------
   
    # LOAD .pkl FILES
    elif format == 'pkl':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        elif platform == "darwin":
            # OS X
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        elif platform in ["win32", "win64"]:
            # Windows
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        return Tensor
    
def main(filetype):
    while True:
        if filetype.strip().lower() in ['mat', '.mat']:
            wantedfile = input('Introduce name of input data file (as in: Desktop/Folder/data.mat): ')
            if os.path.exists(f'{wantedfile}'):
                Tensor = load_data('mat', wantedfile)
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')
        
        elif filetype.strip().lower() in ['.npy', 'npy']:
            wantedfile = input('Introduce name of input data file (as in: Desktop/Folder/data.npy): ')
            if os.path.exists(f'{wantedfile}'):
                Tensor = load_data('npy', wantedfile)
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')      

        elif filetype.strip().lower() in ['csv', '.csv']:
            wantedfile = input('Introduce name of input data file (as in: Desktop/Folder/data.csv): ')
            if os.path.exists(f'{wantedfile}'):
                Tensor = load_data('csv', wantedfile)
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')
        
        elif filetype.strip().lower() in ['pkl', '.pkl']:
            wantedfile = input('Introduce name of input data file (as in: Desktop/Folder/data.pkl): ')
            if os.path.exists(f'{wantedfile}'):
                Tensor = load_data('pkl', wantedfile)
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')

        elif filetype.strip().lower() in ['h5', '.h5']:
            wantedfile = input('Introduce name of input data file (as in: Desktop/Folder/data.h5): ')
            if os.path.exists(f'{wantedfile}'):
                Tensor = load_data('h5', wantedfile)
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')

    database = wantedfile.strip('.mat').strip('.npy').strip('.csv').strip('.pkl').strip('.h5')

    return Tensor, database

