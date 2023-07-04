def GappyRepair():
    import numpy as np
    import hosvd 
    import os
    from scipy.interpolate import griddata
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    import data_load
    import hdf5storage
    import time
    timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

    print('\nGappy SVD')

    # Load data
    path0 = os.getcwd()

    print('\n-----------------------------')
    print('Inputs: \n')
    while True:
        filetype = input('Select the gappy input file format (.mat, .npy, .csv, .pkl, .h5): ')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    A_gappy, _ = data_load.main(filetype)

    while True:
        Truth = input('Would you like to select ground truth data? (y/n). Continue with No: ')
        if not Truth or Truth.strip().lower() in ['n', 'no']:
            break
        elif Truth.strip().lower() in ['y', 'yes']:
            while True:
                filetype = input('Select the ground truth input file format (.mat, .npy, .csv, .pkl, .h5): ')
                if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
                    break
                else: 
                    print('\tError: The selected input file format is not supported\n')

            Tensor, _ = data_load.main(filetype)
            break
        else:
            print('\tError: Please select yes or no (y/n)\n')

    ## INPUTS
    if A_gappy.ndim <= 2:
        method = 'svd'
    else:
        method = 'hosvd'
        
    while True:
        m = input('Number of modes to retain during reconstruction: ') # Number of modes to retain on the reconstruction
        if m.isdigit():
            m = int(m)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')

    print(f'''
The selected file has {np.sum(np.isnan(A_gappy))} NaN values...

This data must be filled in with one of the following options:
Zeros: NaN values are substituted with zeros
Mean: NaN values are substituted with the mean value of the array or tensor
Interp_1d: NaN values are substituted by interpolation (thought for arrays)
Interp_2d: NaN values are substituted by interpolation (thought for matrices)
Tensor_interp: NaN values are substituted by interpolation (thought for tensors)

The interpolation options are:
Linear interpolation
Nearest interpolation
        ''')

    while True:
        decision_1 = input('How would you like to complete the missing data? (zeros, mean, interp_1d, interp_2d, tensor_interp). Continue with zeros: ')
        if not decision_1 or decision_1.strip().lower() == 'zeros':
            decision_1 = 'zeros'
            break
        elif decision_1.strip().lower() == 'mean':
            decision_1 = 'mean'
            break
        elif decision_1.strip().lower() == 'interp_1d':
            decision_1 = 'interp_1d'
            break
        elif decision_1.strip().lower() == 'interp_2d':
            decision_1 = 'interp_2d'
            break 
        elif decision_1.strip().lower() == 'tensor_interp':
            decision_1 = 'tensor_interp'
            break 
        else: 
            print('\tError: Please select a valid option\n')

    if decision_1 in ['interp_2d', 'tensor_interp']:
        method_ = input('Select an interpolation method (linear/nearest): ')
        if method_.strip().lower() == 'linear':
            method_ = 'linear'
        elif method_.strip().lower() == 'nearest':
            method_ = 'nearest'
        else:
            print('\tError: Select a valid interpolation method\n')
    else:
        method_ = None

    ## OUTPUTS
    print('\n-----------------------------')
    print('Gappy SVD summary:')
    print('\n' + f'Method used: {method.upper()}')
    print(f'Number of retained modes during reconstruction: {m}')
    print(f'Data completed using: {decision_1}')
    print('\n-----------------------------')
    print('Outputs:' + '\n')

    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_Gappy_SVD_solution_{decision_1}_nmodes_{m}'
    else:
        filen = f'{filen}'

    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f"{path0}/{filen}")

    while True:
        output_1 = input('Plot singular values decay before and after? (y/n). Continue with Yes: ')
        if not output_1 or output_1.strip().lower() in ['y', 'yes']:
            output_1 = 'yes'
            break
        elif output_1.strip().lower() in ['n', 'no']:
            output_1 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output_2 = input('Plot surface comparison? (y/n). Continue with Yes: ')
        if not output_2 or output_2.strip().lower() in ['y', 'yes']:
            output_2 = 'yes'
            break    
        elif output_2.strip().lower() in ['n', 'no']:
            output_2 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')

    N = sum(np.isnan(A_gappy.flatten()))

    # Initial reconstruction
    print('\nCompleting data. Please wait...')
    if decision_1 == 'zeros':
        A0_1 = np.nan_to_num(A_gappy, nan = 0)
    elif decision_1 == 'mean':
        A0_1 = np.nan_to_num(A_gappy, nan = 0)
        A0_1 = np.nan_to_num(A_gappy,nan=sum(A0_1.flatten())/(A0_1.size-N))
    elif decision_1 == 'interp_1d':
        A0_1 = np.zeros(A_gappy.shape)
        y = np.linspace(0, 1, A_gappy.shape[1])
        for j in range(np.size(A_gappy,1)):   
            A_gappycolumn = A_gappy[:,j]
            A0_1[:,j] = np.interp(y, y[np.isfinite(A_gappy[:,j])], A_gappycolumn[np.isfinite(A_gappy[:,j])])
    elif decision_1 == 'interp_2d':
        if A_gappy.ndim == 2:
            x = np.linspace(0, 1, A_gappy.shape[0])
            y = np.linspace(0, 1, A_gappy.shape[1])
            xv, yv = np.meshgrid(x, y)
            xnumber = xv[np.isfinite(A_gappy)]
            ynumber = yv[np.isfinite(A_gappy)]
            A0_1 = griddata(np.transpose(np.array([xnumber, ynumber])), A_gappy[np.isfinite(A_gappy)] , (xv, yv), method=method_)
    
    elif decision_1 == 'tensor_interp':
        if A_gappy.ndim == 3:
            shape = A_gappy.shape
            A_gappy_re = np.reshape(A_gappy, (A_gappy.shape[0] * A_gappy.shape[1], A_gappy.shape[2]))
            for j in range(A_gappy_re.shape[-1]):
                velocity_values = A_gappy_re[:, j]
                nan_mask = np.isnan(velocity_values)
                
                non_nan_indices = np.where(~nan_mask)[0]
                
                interpolated_values = griddata(non_nan_indices, velocity_values[~nan_mask], np.arange(A_gappy_re.shape[0]), method=method_)
                
                A_gappy_re[nan_mask, j] = interpolated_values[nan_mask]

            A0_1 = np.reshape(A_gappy_re, shape)

        if A_gappy.ndim == 4:
            shape = A_gappy.shape
            A_gappy_re = np.reshape(A_gappy, (A_gappy.shape[0], A_gappy.shape[1] * A_gappy.shape[2], A_gappy.shape[3]))
            for i in range(A_gappy_re.shape[0]):
                for j in range(A_gappy_re.shape[-1]):
                    velocity_values = A_gappy_re[i, :, j]
                    nan_mask = np.isnan(velocity_values)
                    
                    non_nan_indices = np.where(~nan_mask)[0]
                    
                    interpolated_values = griddata(non_nan_indices, velocity_values[~nan_mask], np.arange(A_gappy_re.shape[1]), method=method_)
                    
                    A_gappy_re[i, nan_mask, j] = interpolated_values[nan_mask]

            A0_1 = np.reshape(A_gappy_re, shape)
        
        elif A_gappy.ndim == 5:
            shape = A_gappy.shape
            reshaped = np.reshape(A_gappy, (A_gappy.shape[0], A_gappy.shape[1] * A_gappy.shape[2] * A_gappy.shape[3], A_gappy.shape[4]))

            x, y, z = np.meshgrid(np.arange(A_gappy.shape[1]), np.arange(A_gappy.shape[2]), np.arange(A_gappy.shape[3]), indexing='ij')
            coordinates = np.stack((x, y, z), axis=-1)

            for i in range(A_gappy.shape[0]):
                for j in range(A_gappy.shape[-1]):
                    velocity_values = reshaped[i, :, j]
                    
                    nan_mask = np.isnan(velocity_values)
                    
                    non_nan_coords = coordinates[~nan_mask]
                    
                    non_nan_values = velocity_values[~nan_mask]
                    
                    interpolated_values = griddata(non_nan_coords, non_nan_values, coordinates, method=method_)
                    
                    reshaped[i, nan_mask, j] = interpolated_values[nan_mask]

            A0_1 = np.reshape(reshaped, shape)
            
    print('Initial reconstruction complete!')    
    A_s = A0_1.copy()
    MSE_gaps = np.zeros(500)

    for ii in range(500):
        print(f'\nIteration number: {ii+1}')

        if method == 'svd':
            print('\nPerforming SVD. Please wait...')
            [U,S,V]=LA.svd(A_s)
            print('SVD complete!')
            S = np.diag(S)
            A_reconst = U[:,0:m] @ S[0:m,0:m] @ V[0:m,:]
        elif method == 'hosvd':
            n = m*np.ones(np.shape(A_s.shape))
            print('\nPerforming HOSVD. Please wait...')
            A_reconst = hosvd.HOSVD_function(A_s,n)[0]
            print('HOSVD complete!')
            
        MSE_gaps[ii] = LA.norm(A_reconst[np.isnan(A_gappy)]-A_s[np.isnan(A_gappy)])/N
        
        if ii>3 and MSE_gaps[ii]>=MSE_gaps[ii-1]:
            break
        else:
            A_s[np.isnan(A_gappy)] = A_reconst[np.isnan(A_gappy)]

    if output_1 or output_2 == 'yes':
        print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
        print('Please CLOSE all figures to continue the run')

    if output_1 == 'yes':
        print('\nPlotting singular values decay')
        if method =='svd':
            print('\nPerforming SVD. Please wait...')
            [U0,S0,V0]=LA.svd(A0_1)
            [U,S,V]=LA.svd(A_s)
            print('SVD complete!\n')
            plt.semilogy(S0/S0[0],'kx')
            plt.semilogy(S/S[0],'rx')
            plt.ylabel('S(i,i)/S(1,1)')
            plt.legend(['Initial Reconstruction','Final Reconstruction'])
            plt.tight_layout()
            plt.savefig(f'{path0}/{filen}/GappyMatrix_Reconstr.png')
            plt.show()
        elif method == 'hosvd':
            print('\nPerforming HOSVD. Please wait...')
            sv0 = hosvd.HOSVD_function(A0_1,n)[3]
            sv = hosvd.HOSVD_function(A_s,n)[3]
            print('HOSVD complete!\n')
            cmap = plt.cm.get_cmap('jet')
            rgba = cmap(np.linspace(0,1,A_s.ndim))
            for i in range(A_s.ndim):
                plt.semilogy(sv0[0,i]/sv0[0,i][0], linestyle = 'none', marker = 'x',color = rgba[i])
                plt.semilogy(sv[0,i]/sv[0,i][0], linestyle = 'none', marker = '+', color = rgba[i])
            plt.legend(['Original Reconstruction','Final Reconstruction'])
            plt.tight_layout()
            plt.savefig(f'{path0}/{filen}/GappyTensor_Reconstr.png')
            plt.show()

    if Truth.strip().lower() in ['y', 'yes']:
        if Tensor.ndim == 2:
            Tensor0 = Tensor[:A_s.shape[0], :A_s.shape[1]].copy()
            RRMSE = np.linalg.norm(np.reshape(Tensor0 - A_s ,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
            print(f'\nError made during reconstruction: {np.round(RRMSE*100, 3)}%')

        if Tensor.ndim == 3:
            Tensor0 = Tensor[:A_s.shape[0], :A_s.shape[1], :A_s.shape[2]].copy()
            RRMSE = np.linalg.norm(np.reshape(Tensor0 - A_s ,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
            print(f'\nError made during reconstruction: {np.round(RRMSE*100, 3)}%')

        if Tensor.ndim == 4:
            Tensor0 = Tensor[:A_s.shape[0], :A_s.shape[1], :A_s.shape[2], :A_s.shape[3]].copy()
            RRMSE = np.linalg.norm(np.reshape(Tensor0 - A_s ,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
            print(f'\nError made during reconstruction: {np.round(RRMSE*100, 3)}%')

        if Tensor.ndim == 5:
            Tensor0 = Tensor[:A_s.shape[0], :A_s.shape[1], :A_s.shape[2], :A_s.shape[3], :A_s.shape[4]].copy()
            RRMSE = np.linalg.norm(np.reshape(Tensor0 - A_s ,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
            print(f'\nError made during reconstruction: {np.round(RRMSE*100, 3)}%')

    if output_2 == 'yes':
        if method == 'hosvd':
            if A_gappy.ndim == 5:
                nz = int(A_gappy.shape[3] / 2)
                for var in range(A_gappy.shape[0]):
                    fig, ax = plt.subplots(figsize=(10, 10), num = f'CLOSE TO CONTINUE RUN - XY plane initial data for component {var+1}')
                    heatmap = ax.imshow(A_gappy[var, ..., nz, 0], cmap='coolwarm')
                    ax.set_title(f'XY Plane initial gappy data - Component {var+1}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    heatmap.set_clim(np.nanmin(A_gappy[var, ..., nz, 0]), np.nanmax(A_gappy[var, ..., nz, 0]))
                    plt.show()
            if A_gappy.ndim == 4:
                for var in range(A_gappy.shape[0]):
                    fig, ax = plt.subplots(figsize=(10, 10), num = f'CLOSE TO CONTINUE RUN - Initial data for component {var+1}')
                    heatmap = ax.imshow(A_gappy[var, ..., 0], cmap='coolwarm')
                    ax.set_title(f'Initial gappy data - Component {var+1}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    heatmap.set_clim(np.nanmin(A_gappy[var, ..., 0]), np.nanmax(A_gappy[var, ..., 0]))
                    plt.show()
            elif A_gappy.ndim == 3:
                fig, ax = plt.subplots(figsize=(10, 10), num = 'CLOSE TO CONTINUE RUN - Initial data')
                heatmap = ax.imshow(A_gappy[..., 0], cmap='coolwarm')
                ax.set_title('Initial gappy data')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                heatmap.set_clim(np.nanmin(A_gappy[..., 0]), np.nanmax(A_gappy[..., 0]))
                plt.show()

        elif method == 'svd':
            fig, ax = plt.subplots(figsize=(10, 10), num = 'CLOSE TO CONTINUE RUN - Initial data')
            heatmap = ax.imshow(A_gappy, cmap='coolwarm')
            ax.set_title('Initial gappy data')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            heatmap.set_clim(np.nanmin(A_gappy), np.nanmax(A_gappy))
            plt.show()

        # Initial and final reconstruction vs ground truth   
        if Truth.strip().lower() in ['y', 'yes']:
            ncols = 3
        else:
            ncols = 2

        if method == 'svd':
            fig, ax = plt.subplots(1, ncols, num = 'CLOSE TO CONTINUE RUN - Data comparison')
            plt.suptitle('Data comparison')
            ax[0].contourf(A_gappy)
            ax[0].set_title('Initial data')
            ax[1].contourf(A_s)
            ax[1].set_title('Reconstructed data')
            if ncols == 3:
                ax[2].contourf(Tensor)
                ax[2].set_title('Ground truth')
            plt.savefig(f'{path0}/{filen}/data_comparison.png')
            plt.show()

        if method == 'hosvd':
            if A_gappy.ndim == 4:
                fig, ax = plt.subplots(1, ncols, num = 'CLOSE TO CONTINUE RUN - Data comparison')
                plt.suptitle('Data comparison')
                im0 = ax[0].contourf(A_gappy[0, ..., 0])
                ax[0].set_title('Initial data')
                im1 = ax[1].contourf(A_s[0, ..., 0])
                ax[1].set_title('Reconstructed data')
                if ncols == 3:
                    im2 = ax[2].contourf(Tensor[0, ..., 0])
                    ax[2].set_title('Ground truth')
                    fig.colorbar(im2, ax=ax[2])
                else:
                    fig.colorbar(im1, ax=ax[1])
                plt.savefig(f'{path0}/{filen}/data_comparison.png')
                plt.show()

            if A_gappy.ndim == 5:
                nz = int(A_gappy.shape[3] / 2)
                fig, ax = plt.subplots(1, ncols, num = 'CLOSE TO CONTINUE RUN - Data comparison')
                plt.suptitle('XY plane data comparison')
                ax[0].contourf(A_gappy[0, ..., nz, 0])
                ax[0].set_title('Initial data')
                ax[1].contourf(A_s[0, ..., nz, 0])
                ax[1].set_title('Reconstructed data')
                if ncols == 3:
                    ax[2].contourf(Tensor[0, ..., nz, 0])
                    ax[2].set_title('Ground truth')
                plt.savefig(f'{path0}/{filen}/data_comparison.png')
                plt.show()

            if A_gappy.ndim == 3:
                fig, ax = plt.subplots(1, ncols, num = 'CLOSE TO CONTINUE RUN - Data comparison')
                plt.suptitle('Data comparison')
                ax[0].contourf(A_gappy[..., 0])
                ax[0].set_title('Initial data')
                ax[1].contourf(A_s[..., 0])
                ax[1].set_title('Reconstructed data')
                if ncols == 3:
                    ax[2].contourf(Tensor[..., 0])
                    ax[2].set_title('Ground truth')
                plt.savefig(f'{path0}/{filen}/data_comparison.png')
                plt.show()
        
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/reconstructed.npy', A_s)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic1 = {"reconstructed": A_s}
            file_mat1 = str(f'{path0}/{filen}/reconstructed.mat')
            hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')








