import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")

import AEmodel
import DLsuperresolution
import FullDLmodel
import Superresolution
import GappyRepair
import mdHODMD
import mdHODMD_pred
import HODMD
import HODMD_pred
import mainHOSVD
import HybCNNpredmodel
import HybRNNpredmodel
import SVD

__version__ = '0.1'
__author__ = 'ModelFLOWs Research Group - E.T.S.I. Aeronautica y del Espacio - Universidad Politecnica de Madrid'

print('\n\n')
print('''
         __  __             _        _  _____  _      ___ __        __                                
        |  \/  |  ___    __| |  ___ | ||  ___|| |    / _ \\\ \      / /___          __ _  _ __   _ __  
        | |\/| | / _ \  / _` | / _ \| || |_   | |   | | | |\ \ /\ / // __| _____  / _` || '_ \ | '_ \ 
        | |  | || (_) || (_| ||  __/| ||  _|  | |___| |_| | \ V  V / \__ \|_____|| (_| || |_) || |_) |
        |_|  |_| \___/  \__,_| \___||_||_|    |_____|\___/   \_/\_/  |___/        \__,_|| .__/ | .__/ 
                                                                                        |_|    |_|   

                                        ModelFLOWs Application
                                        ----------------------
''')

print(f'''
Authors: {__author__}

Version: {__version__}


This data-driven application consists of two modules: 
- Modal Decomposition 
- Deep Learning

Both blocks consist of algorithms capable of:
- detecting patterns, 
- repairing and enhancing data, 
- and predicting data from complex flow databases

The databases can be in the following formats:
- MATLAB ".mat"
- Numpy ".npy"
- Pickle ".pkl"
- Pandas ".csv"
- h5 ".h5"

This application takes in databases with the following shapes:
- 1D Arrays -> (1, n)
- 2D Matrices -> (x, y)
- 3D Tensors -> (y, x, t)
- 4D Tensors -> (m, y, x, t)
- 5D Tensors -> (m, y, x, z, t)

IMPORTANT: The data that defines the spatial mesh (x, y, z) must be located in the specified positions

For more information please visit: https://modelflows.github.io/modelflowsapp/
''')

while True:
    print('''
----------------------MODULES----------------------

Modules:
1) Modal Decomposition
2) Deep Learning

    ''')
    while True:
        module = input('Select a MODULE (1/2): ')
        if module.isdigit():
            if int(module) == 1:
                break
            elif int(module) == 2:
                break
            else:
                print('\tError: Please select a valid module\n')
        else:
            print('\tError: Please introduce a valid input\n')

    print('''
----------------------ACTIONS----------------------

Operations:
1) Pattern detection
2) Data repairing
3) Prediction
0) Exit

    ''')

    while True:
        operation = input('Select an OPERATION (1/2/3): ')
        if operation.isdigit():
            if int(operation) == 1:
                break
            elif int(operation) == 2:
                break
            elif int(operation) == 3:
                break
            elif int(operation) == 0:
                break
            else:
                print('\tError: Please select a valid operation\n')
        else:
            print('\tError: Please introduce a valid input\n')

    if int(module) == 1 and int(operation) == 1:
        print('''
-----------------PATTERN DETECTION-----------------

Algorithms or mathematic methods for pattern detection
1) SVD
2) HOSVD
3) HODMD
0) Exit  

        ''')
        while True:
            option = input('Select an ALGORITHM (1/2/3): ')
            if option.isdigit():
                if int(option) == 3:
                    break
                elif int(option) == 2:
                    mainHOSVD.HOSVD()
                    break
                elif int(option) == 1:
                    SVD.SVD()
                    break
                elif int(option) == 0:
                    break
                else:
                    print('\tError: Please select a valid algorithm\n')
            else:
                print('\tError: Please introduce a valid input\n')
        
        if int(option) == 3:
            print('''
-----------------------HODMD-----------------------

Available HODMD algorithms
1) HODMD
2) Multi-dimensional HODMD 
0) Exit

        ''')
            while True:
                type = input('Select a HODMD ALGORITHM (1/2): ')
                if type.isdigit():
                    if int(type) == 1:
                        HODMD.HODMD()
                        break
                    elif int(type) == 2:
                        mdHODMD.mdHODMD()
                        break
                    elif int(type) == 0:
                        break
                    else:
                        print('\tError: Please select a valid HODMD algorithm\n')
                else:
                    print('\tError: Please introduce a valid input\n')
        
    if int(module) == 1 and int(operation) == 2:
        print('''
----------------DATA RECONSTRUCTION----------------

Gappy operations
1) Data repairing
2) Data enhancement (superresolution) 
0) Exit 

        ''')
        while True:
            option = input('Select an OPTION (1/2): ')
            if option.isdigit():
                if int(option) == 1:
                    GappyRepair.GappyRepair()
                    break
                elif int(option) == 2:
                    Superresolution.GappyResolution()
                    break
                elif int(option) == 0:
                    break
                else:
                    print('\tError: Please select a valid option\n')
            else:
                print('\tError: Please introduce a valid input\n')

    if int(module) == 1 and int(operation) == 3:
        print('''
---------------------PREDICTION--------------------

Algorithms or mathematic methods for prediction
1) Predictive HODMD
2) Predictive Multi-dimensional HODMD   
0) Exit

        ''')
        while True:
            type = input('Select a HODMD ALGORITHM (1/2): ')
            if type.isdigit():
                if int(type) == 1:
                    HODMD_pred.HODMDpred()
                    break
                elif int(type) == 2:
                    mdHODMD_pred.mdHODMDpred()
                    break
                elif int(type) == 0:
                    break
                else:
                    print('\tError: Please select a valid HODMD algorithm\n')
            else:
                print('\tError: Please introduce a valid input\n')

    if int(module) == 2 and int(operation) == 1:
        print('''
-----------------PATTERN DETECTION-----------------

        ''')
        AEmodel.autoencoders()

    if int(module) == 2 and int(operation) == 2:
        print('''
----------------DATA RECONSTRUCTION----------------

        ''')
        DLsuperresolution.DNNreconstruct()

    if int(module) == 2 and int(operation) == 3:
        print('''
---------------------PREDICTION--------------------

Model type:
1) Full Deep Learning model
2) Hybrid Deep Learning model (SVD + DL)
0) Exit

        ''')
        while True:
            option = input('Select a MODEL TYPE (1/2): ')
            if option.isdigit():
                if int(option) == 1:
                    break
                elif int(option) == 2:
                    break
                elif int(option) == 0:
                    break
                else:
                    print('\tError: Please select a valid model type\n')
            else:
                print('\tError: Please introduce a valid input\n')

        if int(option) == 1:
            print('''
--------------FULL DEEP LEARNING MODEL-------------

Model architecture:
1) CNN
2) RNN
0) Exit

            ''')
            while True:
                type = input('Select a MODEL ARCHITECTURE (1/2): ')
                if type.isdigit():
                    if int(type) == 1:
                        FullDLmodel.FullDL('cnn')
                        break
                    elif int(type) == 2:
                        FullDLmodel.FullDL('rnn')
                        break
                    elif int(type) == 0:
                        break
                    else:
                        print('\tError: Please select a valid model architecture\n')
                else:
                    print('\tError: Please introduce a valid input\n')

        elif int(option) == 2:
            print('''
-------------HYBRID DEEP LEARNING MODEL------------

Model architecture:
1) SVD + CNN
2) SVD + RNN
0) Exit

            ''')

            while True:
                type = input('Select a MODEL ARCHITECTURE (1/2): ')
                if type.isdigit():
                    if int(type) == 1:
                        HybCNNpredmodel.hybCNN()
                        break
                    elif int(type) == 2:
                        HybRNNpredmodel.hybRNN()
                        break
                    elif int(type) == 0:
                        break
                    else:
                        print('\tError: Please select a valid model architecture\n')
                else:
                    print('\tError: Please introduce a valid input\n')

    while True:
        cont = input('\nWould you like to perform another operation? (y/n): ')
        if cont.strip().lower() in ['y', 'yes']:
            print('\n')
            break
        if cont.strip().lower() in ['n', 'no']:
            print('\n\n')
            print('''
         __  __             _        _  _____  _      ___ __        __                                
        |  \/  |  ___    __| |  ___ | ||  ___|| |    / _ \\\ \      / /___          __ _  _ __   _ __  
        | |\/| | / _ \  / _` | / _ \| || |_   | |   | | | |\ \ /\ / // __| _____  / _` || '_ \ | '_ \ 
        | |  | || (_) || (_| ||  __/| ||  _|  | |___| |_| | \ V  V / \__ \|_____|| (_| || |_) || |_) |
        |_|  |_| \___/  \__,_| \___||_||_|    |_____|\___/   \_/\_/  |___/        \__,_|| .__/ | .__/ 
                                                                                        |_|    |_|   

                                        Exiting ModelFLOWs-app...
                                        -------------------------
            ''')
            exit()
        else: 
            print('\tError: Please select YES or NO (y/n)\n') 
    
    if cont.strip().lower() in ['y', 'yes']:
        continue 












