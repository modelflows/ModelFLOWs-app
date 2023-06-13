def GappyResolution():
    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    import data_load
    import os
    import time
    import hosvd
    import hdf5storage
    timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

    ## INPUTS
    path0 = os.getcwd()

    print('\nData Enhancement')
    print('\n-----------------------------')
    print('Inputs: \n')
    while True:
        filetype = input('Select the downsampled input file format (.mat, .npy, .csv, .pkl, .h5): ')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    A_down, _ = data_load.main(filetype)

    if A_down.ndim == 2:
        method = 'svd'
        m = [0, 1]
    else:
        method = 'hosvd'
        if A_down.ndim == 3:
            m = [0, 1]
        elif A_down.ndim == 4:
            m = [1, 2]
        elif A_down.ndim == 5:
            m = [1, 2, 3]
        
    while True:
        aug = input('Introduce the enhancement factor (2^factor): ')
        if aug.isdigit():
            enhance = int(aug)
            break
        else:
            print('\tError: Please introduce a number (must be integer)\n')


    # Output
    print('\n-----------------------------')
    print('Data enhancement summary:')
    print('\n' + f'Method used: {method.upper()}')
    print(f'Enhacement scale: {2**enhance}')
    print('\n-----------------------------')
    print('Outputs:' + '\n')

    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_Gappy_data_enhancement_factor_{enhance}'
    else:
        filen = f'{filen}'

    while True:
        decision = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not decision or decision.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')

    print('')

    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f"{path0}/{filen}")

    A_d = A_down

    for ii in range(enhance):
        if method =='svd':
            print('Performing SVD. Please wait...')
            [U,S,V]=LA.svd(A_d, full_matrices=False)
            S = np.diag(S)
            print('SVD complete!\n')

            if ii  == 0:
                nm = S.shape[0]
            
            x = np.linspace(0, 1, A_d.shape[0]*2)
            y = np.linspace(0, 1, A_d.shape[1]*2)
            U_dens = np.zeros((x.shape[0],U.shape[1]))
            V_dens = np.zeros((V.shape[0],y.shape[0]))

            for j in range(S.shape[0]):   
                Udenscolumn = U[:,j]
                U_dens[:,j] = np.interp(x, x[0:x.shape[0]:2], Udenscolumn)
                Vdenscolumn = V[j,:]
                V_dens[j,:] = np.interp(y, y[0:y.shape[0]:2], Vdenscolumn)

            A_reconst = U_dens @ S @ V_dens

            print('Performing SVD. Please wait...')
            [Ur,Sr,Vr] = LA.svd(A_reconst)
            print('SVD complete!\n')
            Sr = np.diag(Sr)
            A_d = Ur[:,:nm] @ Sr[:nm,:nm] @ Vr[:nm,:]
            
        elif method =='hosvd':
            
            if ii == 0:
                n = A_d.shape
                
            _ , S, U, _ , _ = hosvd.HOSVD_function(A_d,n)
            Udens = U
            
            for dim in m:
                x = np.linspace(0, 1, U[0][dim].shape[0]*2)
                U_dens = np.zeros((x.shape[0],S.shape[dim]))
                for j in range(S.shape[dim]):
                    Udenscolumn = U[0][dim][:,j]
                    U_dens[:,j] = np.interp(x, x[0:x.shape[0]:2], Udenscolumn) 
                Udens[0][dim] = U_dens

            print('Performing HOSVD. Please wait...')  
            A_d = hosvd.tprod(S, Udens)
            A_d = hosvd.HOSVD_function(A_d,n)[0]
            print('HOSVD complete!\n')

    A_reconst = A_d

    print(f'''
Reconstruction complete!

Reconstruction summary:
Original data shape: {A_down.shape}
Enhanced data shape: {A_reconst.shape}

    ''')

    if not decision or decision.strip().lower() in ['npy', '.npy']:
        np.save(f"{path0}/{filen}/enhanced_data.npy", A_reconst)

    if decision.strip().lower() in ['.mat', 'mat']:
        mdic = {"Enhancement": A_reconst}
        file_mat= str(f'{path0}/{filen}/enhanced_data.mat')
        hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

    dims = A_reconst.ndim

    print('Please CLOSE all figures to continue the run\n')

    if dims == 2:
        fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Original data vs. Enhanced data')
        plt.suptitle('Original data vs. Enhanced data')
        ax[0].contourf(A_down)
        ax[0].set_title('Original data')
        ax[0].xaxis.grid(True, zorder = 0)
        ax[0].yaxis.grid(True, zorder = 0)
        ax[1].contourf(A_reconst)
        ax[1].set_title('Enhanced data')
        ax[1].xaxis.grid(True, zorder = 0)
        ax[1].yaxis.grid(True, zorder = 0)
        plt.show()
    
    if dims > 2:
        while True:
            if dims > 3:
                while True:
                    v = input(f'Select a component to plot (max. is {A_down.shape[0]}). Continue with 1: ')
                    if not v:
                        v = 0
                        break
                    elif v.isdigit():
                        if int(v) <= A_down.shape[0]:
                            v = int(v) - 1
                            break
                        else:
                            print('\tError: Selected value is out of range')
                    else:
                        print('\tError: Select a valid number format (must be integer)\n')
            while True:    
                t = input(f'Select a snapshot to plot (max. is {A_down.shape[-1]}). Continue with 1: ')
                if not t:
                    t = 0
                    break
                elif t.isdigit():
                    if int(t) <= A_down.shape[-1]:
                        t = int(t) - 1
                        break
                    else:
                        print('\tError: Selected value is out of range')
                else:
                    print('\tError: Select a valid number format (must be integer)\n')

            fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Original data vs. Enhanced data')
            plt.suptitle('Original data vs. Enhanced data')
            
            if dims == 3:
                ax[0].contourf(A_down[:, :, t])
                ax[0].set_title('Original data')
                ax[0].xaxis.grid(True, zorder = 0)
                ax[0].yaxis.grid(True, zorder = 0)
                ax[1].contourf(A_reconst[:, :, t])
                ax[1].set_title('Enhanced data')
                ax[1].xaxis.grid(True, zorder = 0)
                ax[1].yaxis.grid(True, zorder = 0)
                plt.show()

            if dims == 4:
                ax[0].contourf(A_down[v, :,  :, t])
                ax[0].set_title('Original data')
                ax[0].xaxis.grid(True, zorder = 0)
                ax[0].yaxis.grid(True, zorder = 0)
                ax[1].contourf(A_reconst[v, :,  :, t])
                ax[1].set_title('Enhanced data')
                ax[1].xaxis.grid(True, zorder = 0)
                ax[1].yaxis.grid(True, zorder = 0)
                plt.show()

            if dims == 5:
                nz = int(A_down.shape[3] / 2)
                ax[0].contourf(A_down[v, :, :, nz, t])
                ax[0].set_title('Original data - XY Plane')
                ax[0].xaxis.grid(True, zorder = 0)
                ax[0].yaxis.grid(True, zorder = 0)
                ax[1].contourf(A_reconst[v, :, :, nz, t])
                ax[1].set_title('Enhanced data - XY Plane')
                ax[1].xaxis.grid(True, zorder = 0)
                ax[1].yaxis.grid(True, zorder = 0)
                plt.show()

            while True:
                Resp = input('Do you want to plot another figure? Yes or No (y/n). Continue with No: ')
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















