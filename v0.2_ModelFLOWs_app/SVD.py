import numpy as np
import data_load
import os
import matplotlib.pyplot as plt
path0 = os.getcwd()
import time
import hdf5storage
timestr = time.strftime("%Y-%m-%d_%H.%M.%S")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def svdtrunc(A):
    U, S, V = np.linalg.svd(A, full_matrices = False)
    return U, S, V

def SVD():

    print('\nSVD Algorithm')
    print('\n-----------------------------')
    print('Inputs:' + '\n')

    while True:
        filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    Tensor, _ = data_load.main(filetype)

    while True:
        n_modes = input('Select number of SVD modes. Continue with 18: ')
        if not n_modes:
            n_modes = 18
            break
        if n_modes.isdigit():
            n_modes = int(n_modes)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')
    
    print('\n-----------------------------')
    print('SVD summary:')
    print(f'Number of modes to retain: {n_modes}')

    print('\n-----------------------------')
    print('Outputs:\n')

    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_SVD_solution_modes_{n_modes}'
    else:
        filen = f'{filen}'

    while True:
        decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')
    
    print('')

    dims = Tensor.ndim
    shape = Tensor.shape

    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f"{path0}/{filen}")

    if dims > 2:
        dims_prod = np.prod(shape[:-1])
        Tensor0 = np.reshape(Tensor, (dims_prod, shape[-1]))

    U, S, V = svdtrunc(Tensor0) 

    S = np.diag(S)

    U = U[:,0:n_modes]
    S = S[0:n_modes,0:n_modes]
    V = V[0:n_modes,:]

    Reconst = (U @ S) @ V

    svd_modes = np.dot(U, S)

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - SVD modes')
    for i in range(S.shape[0]):
        ax.plot(i+1, S[i][i] / S[0][0], "k*") 
        
    ax.set_yscale('log')      # Logarithmic scale in y axis
    ax.set_xlabel('SVD modes')
    ax.set_ylabel('Singular values')
    ax.set_title('SVD modes vs. Singular values')
    plt.savefig(f'{path0}/{filen}/svd_modes_plot.png', bbox_inches='tight')
    plt.show()
    plt.close()

    if dims <= 2:
        print(f'SVD modes shape: {svd_modes.shape}')
        print(f'Reconstructed tensor shape: {Reconst.shape}')

        # Usado para verificar. Borrar despues

        Norm2V = np.linalg.norm(Tensor.flatten(), 2)
        diff = (Tensor - Reconst).flatten()
        Norm2diff = np.linalg.norm(diff, ord=2)
        RelativeErrorRMS = Norm2diff/Norm2V
        print('\n' + f'The relative error (RMS) is: ' +  format(RelativeErrorRMS,'e'))
        
        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')
        ax.contourf(svd_modes)
        ax.set_title('SVD mode')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    elif dims > 2:
        newshape = []
        newshape.append(shape[:-1])
        newshape.append(svd_modes.shape[-1])
        newshape = list(newshape[0]) + [newshape[1]]
        svd_modes = np.reshape(svd_modes, np.array(newshape))
        print(f'SVD modes shape: {svd_modes.shape}')
        Reconst = np.reshape(Reconst, shape)
        print(f'Reconstructed tensor: {Reconst.shape}')

        # Usado para verificar. Borrar despues

        RRMSE = np.linalg.norm(np.reshape(Tensor-Reconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
        print(f'\nRelative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')

    if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
        np.save(f'{path0}/{filen}/Reconstruction.npy', Reconst)
        np.save(f'{path0}/{filen}/svd_modes.npy', svd_modes)

    if decision_2.strip().lower() in ['.mat', 'mat']:
        mdic0 = {"Reconst": Reconst}
        mdic1 = {"svd_modes": svd_modes}

        file_mat0 = str(f'{path0}/{filen}/Reconstruction.mat')
        file_mat1 = str(f'{path0}/{filen}/svd_modes.mat')

        hdf5storage.savemat(file_mat0, mdic0, appendmat=True, format='7.3')
        hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')


    print('Please CLOSE all figures to continue the run\n')

    while True:
        while True:
            ModeNum = input(f'Introduce the mode number to plot (default mode 1). Maximum number of modes is {svd_modes.shape[-1]}: ')
            if not ModeNum:
                ModeNum = 0
                break
            elif ModeNum.isdigit():
                if int(ModeNum) <= svd_modes.shape[-1]:
                    ModeNum = int(ModeNum)-1  
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')
        if dims > 3:
            while True:
                ModComp = input(f'Introduce the component to plot (default component 1). Maximum number of components is {svd_modes.shape[0]}: ')
                if not ModComp:
                    ModComp = 0
                    break
                elif ModComp.isdigit():
                    if int(ModComp) <= svd_modes.shape[0]:
                        ModComp = int(ModComp)-1
                        break
                    else:
                        print('\tError: Selected value is out of bounds\n')
                else:
                    print('\tError: Select a valid number format (must be integer)\n')

        elif dims==3:
            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')
            ax.contourf(svd_modes[:,:,ModeNum])
            ax.set_title(f'SVD modes - Mode Number {ModeNum+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()
        
        if dims==4:
            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')
            ax.contourf(svd_modes[ModComp,:,:,ModeNum])
            ax.set_title(f'SVD modes - Component {ModComp+1} - Mode Number {ModeNum+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        elif dims==5:
            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode XY plane')
            ax.contourf(svd_modes[ModComp,:,:,0,ModeNum])
            ax.set_title(f'SVD modes XY plane - Component {ModComp+1} - Mode Number {ModeNum+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        while True:
            Resp = input('Do you want to plot another mode? Yes or No (y/n). Press enter to continue with No: ')
            if not Resp or Resp.strip().lower() in ['n', 'no']:
                Resp = 0
                break
            elif Resp.strip().lower() in ['y', 'yes']:
                Resp = 1
                break
            else:
                print('\tError: Select yes or no (y/n)\n')
        
        if Resp == 0:
            break

        if Resp == 1:
            continue





