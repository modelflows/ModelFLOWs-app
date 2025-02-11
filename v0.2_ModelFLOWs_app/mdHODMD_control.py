def HODMDsens():
    import DMDd
    import hosvd
    import warnings
    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

    import os               
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    import numpy as np
    import matplotlib.pyplot as plt
    import data_load
    import hdf5storage
    import time
    timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    print('HODMD for control' + '\n')
    print('-----------------------------')

    print('''
The HODMD algorithm for flow control needs a prior calibration of HODMD.
    ''')

    # Load data
    path0 = os.getcwd()
    print('Inputs:' + '\n')

    while True:
        filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    Tensor, _ = data_load.main(filetype)

  
    ##Number of snapshots SNAP:
    while True:
        SNAP = input(f'Introduce number of snapshots. Continue with {Tensor.shape[-1]} snapshots: ')
        if not SNAP:
            SNAP = int(Tensor.shape[-1])
            break
        elif SNAP.isdigit():
            SNAP = int(SNAP)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')

    ## d Parameter:
    while True:
        d = input(f'Introduce number of HODMD windows (d): ')
        if d.isdigit():
            d = int(d)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')

                        
    # SVD Tolerance
    while True:
        varepsilon1 = input('Introduce SVD tolerance. Continue with 1e-3: ')
        if not varepsilon1:
            varepsilon1 = 1e-3   
            break
        elif is_float(varepsilon1):
            varepsilon1 = float(varepsilon1)
            break
        else:
            print('\tError:Please introduce a number\n')
        
    varepsilon2 = varepsilon1
  
    
    ##Time:
    while True:
        deltaT = input('Introduce time step (deltaT). Continue with 1: ')
        if not deltaT:
            deltaT = 1
            break
        elif is_float(deltaT):
            deltaT = float(deltaT)
            break
        else:
            print('\tError: Please introduce a number\n')


    Time = np.linspace(0,SNAP-1,num=SNAP)*deltaT
        
    ## Position of temporal variable:
    TimePos = Tensor.ndim

    #--------------------------------------------------------------------------------------------------------------------------------------
    ################### Output ###################

    print('\n-----------------------------')
    print('HODMD for flow control summary:')
    print('\n' + f'Number of snapshots set at: {SNAP}')
    print(f'd Parameter(s) set at: {d}')
    print(f'SVD tolerance(s) {varepsilon1}')
    print(f'HODMD tolerance(s): {varepsilon2}')
    print(f'Time gradient set at deltaT: {deltaT}')

    print('\n-----------------------------')
    print('Outputs:\n')

    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_HODMDcontrol_solution'
    else:
        filen = f'{filen}'

    while True:
        decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')
    
    print('')

    # LOAD MESH in format (3, nx, ny ,nz) or (2, nx, ny)
    while True:
        decision_3 = input('Do you want to plot the non linear sensitivity? (y/n)')
        if not decision_3 or decision_3.strip().lower() in ['yes', 'y']:
            while True:
                filetype = input('Select the input file format for the mesh (.mat, .npy, .csv, .pkl, .h5): ')
                if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
                    break
                else: 
                    print('\tError: The selected input file format is not supported\n')
            Malla, _ = data_load.main(filetype)
            break
        elif decision_3.strip().lower() in ['no', 'n']:
            break
        else:
            print('\tError: Please select a valid answer\n')
            
    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f"{path0}/{filen}")

    ## Save results in folder:

    Tensor0 = Tensor
    shapeTens = list(np.shape(Tensor))
    shapeTens[-1] = SNAP
    Tensor = np.zeros(shapeTens)

    Tensor[..., :] = Tensor0[..., 0:SNAP]
    TensorR = Tensor.copy()

    ## ALGORITHM:

    ## ITERATIVE:
    nn0 = np.shape(Tensor)
    nn = np.array(nn0)
    nn[1:np.size(nn)] = 0

    Frequencies = []
    Amplitudes = []
    GrowthRates = []
    d_val = []
    tol_val = []
    DMDmode_list = []
    Tensor_rec_list = []
    
    print(f'\nRunning HODMD for control for d = {d} and tol = {varepsilon1}')
    
    for iter in range(3):
        
        
        if not os.path.exists(f'{path0}/{filen}/Step_{iter}'):
            os.mkdir(f"{path0}/{filen}/Step_{iter}")
        if not os.path.exists(f'{path0}/{filen}/Step_{iter}/DMDmodes'):
            os.mkdir(f"{path0}/{filen}/Step_{iter}/DMDmodes")
        
        if iter == 1:
            Frequencies = []
            Amplitudes = []
            GrowthRates = []
            d_val = []
            tol_val = []
            DMDmode_list = []
            Tensor_rec_list = []
            TensorR = []
            if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
                TensorR = data_load.load_data('npy',f'{path0}/{filen}/Step_0/TensorReconst.npy')
            elif decision_2.strip().lower() in ['.mat', 'mat']:
                TensorR = data_load.load_data('mat',f'{path0}/{filen}/Step_0/TensorReconst.mat')
        elif iter == 2:
            Frequencies = []
            Amplitudes = []
            GrowthRates = []
            d_val = []
            tol_val = []
            DMDmode_list = []
            Tensor_rec_list = []
            TensorR = []
            TensorReconst = []
            
            if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
                Tensor1 = data_load.load_data('npy',f'{path0}/{filen}/Step_0/TensorReconst.npy')
            elif decision_2.strip().lower() in ['.mat', 'mat']:
                Tensor1 = data_load.load_data('mat',f'{path0}/{filen}/Step_0/TensorReconst.mat')
            Media = np.mean(Tensor1,axis=Tensor1.ndim-1)  
            TensorR = np.zeros(Tensor1.shape)

            for nsnap in range(SNAP):
                TensorR[...,nsnap] = Tensor1[...,(SNAP-1)-nsnap] - Media       

            Media = []
        
        ## Perform HOSVD decomposition to calculate the reduced temporal matrix: hatT
        nnin = nn
        nnin = tuple(nnin)
        print('\nPerforming HOSVD. Please wait...\n')
        [hatT,U,S,sv,nn1,n,TT] = hosvd.HOSVD(TensorR,varepsilon1,nn,nn0,TimePos)
        print('\nHOSVD complete!\n')
        # hatT: Reduced temporal matrix
        
        ## Perform HODMD to the reduced temporal matrix hatT:
        print('Performing HODMD. Please wait...\n')
        [hatMode,Amplitude,Eigval,GrowthRate,Frequency] = DMDd.hodmd_IT(hatT,d,Time,varepsilon1,varepsilon2)
        print('\nHODMD complete!\n')
            
        ## Reconstruct the original Tensor using the DMD expansion:
        TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)
                
        RRMSE = np.linalg.norm(np.reshape(Tensor-TensorReconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
        print(f'Relative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')
                
        GRFreqAmp = np.zeros((np.size(GrowthRate),3))
        for ind in range (0,np.size(GrowthRate)):
            GRFreqAmp[ind,0] = GrowthRate[ind]
            GRFreqAmp[ind,1] = Frequency[ind]
            GRFreqAmp[ind,2] = Amplitude[ind]
                    
        print('GrowthRate, Frequency and Amplitude:')
        print(GRFreqAmp)
        
        Frequencies.append(Frequency)
        Amplitudes.append(Amplitude)
        GrowthRates.append(GrowthRate)
        d_val.append(d)
        tol_val.append(varepsilon1)
        
                    
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/Step_{iter}/GRFreqAmp.npy', GRFreqAmp)
                        
        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"GRFreqAmp": GRFreqAmp}
            file_mat = str(f'{path0}/{filen}/Step_{iter}/GRFreqAmp.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        # Tensor reconstruction
        if iter == 0:
            TensorReconst = DMDd.reconst_IT(hatMode[:,:3],Time,U,S,sv,nn1,TimePos,GrowthRate[:3],Frequency[:3])
        else:
            TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)
        Tensor_rec_list.append(TensorReconst)

        ## Save the reconstruction of the tensor and the Growth rates, frequencies and amplitudes:
            
        # Reconstruction:
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/Step_{iter}/TensorReconst.npy', TensorReconst)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"TensorReconst": TensorReconst}
            file_mat = str(f'{path0}/{filen}/Step_{iter}/TensorReconst.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        ## Calculate DMD modes:
        print('Calculating DMD modes...')
        N = np.shape(hatT)[0]
        DMDmode = DMDd.modes_IT(N,hatMode,Amplitude,U,S,nn1,TimePos)
        DMDmode_list.append(DMDmode)

        # Save DMD modes:
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/Step_{iter}/DMDmode.npy',DMDmode)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"DMDmode": DMDmode}
            file_mat = str(f'{path0}/{filen}/Step_{iter}/DMDmode.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        print(f'\nSaving first 2 DMDmode plots to {path0}/{filen}/Step_{iter}/DMDmodes\n')

        if TimePos == 3:
            for ModeNum in range(2):
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'DMDmode - Mode Number {ModeNum+1}')
                ax[0].contourf(DMDmode[:,:,ModeNum].real)
                ax[0].set_title('Real part')
                ax[0].set_xlabel('X')
                ax[0].set_ylabel('Y')

                ax[1].contourf(DMDmode[:,:,ModeNum].imag)
                ax[1].set_title('Imaginary part')
                ax[1].set_xlabel('X')
                ax[1].set_ylabel('Y')
                plt.savefig(f'{path0}/{filen}/Step_{iter}/DMDmodes/ModeNum_{ModeNum+1}.png')
                plt.close(fig)
                
        for ModComp in range(DMDmode.shape[0]):
            for ModeNum in range(2):
                if TimePos==4:
                    fig, ax = plt.subplots(1, 2)
                    fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}')
                    ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
                    ax[0].set_title('Real part')
                    ax[0].set_xlabel('X')
                    ax[0].set_ylabel('Y')
                    
                    ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
                    ax[1].set_title('Imaginary part')
                    ax[1].set_xlabel('X')
                    ax[1].set_ylabel('Y')
                    plt.savefig(f'{path0}/{filen}/Step_{iter}/DMDmodes/DMDmodeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                    plt.close(fig)

                elif TimePos==5:
                    nz = int(Tensor.shape[3] / 2)
                    fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode XY plane')
                    fig.suptitle(f'DMDmode XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
                    ax[0].contourf(DMDmode[ModComp,:,:,nz,ModeNum].real)
                    ax[0].set_title('Real part - XY Plane')
                    ax[0].set_xlabel('X')
                    ax[0].set_ylabel('Y')
                    
                    ax[1].contourf(DMDmode[ModComp,:,:,nz,ModeNum].imag)
                    ax[1].set_title('Imaginary part - XY Plane')
                    ax[1].set_xlabel('X')
                    ax[1].set_ylabel('Y')
                    plt.savefig(f'{path0}/{filen}/Step_{iter}/DMDmodes/DMDmode_XY_Comp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                    plt.close(fig)
                else:
                    pass
                
    if not decision_3 or decision_3.strip().lower() in ['yes', 'y']:
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            ModesD = data_load.load_data('npy',f'{path0}/{filen}/Step_1/DMDmode.npy')
            ModesA = data_load.load_data('npy',f'{path0}/{filen}/Step_2/DMDmode.npy')
            GRFA = data_load.load_data('npy',f'{path0}/{filen}/Step_1/GRFreqAmp.npy')
        elif decision_2.strip().lower() in ['.mat', 'mat']:
            ModesD = data_load.load_data('mat',f'{path0}/{filen}/Step_1/DMDmode.mat')
            ModesA = data_load.load_data('mat',f'{path0}/{filen}/Step_2/DMDmode.mat')
            GRFA = data_load.load_data('mat',f'{path0}/{filen}/Step_1/GRFreqAmp.mat')
            
        mode0 = ModesD[...,0]
        moded = ModesD[...,1]
        modea = ModesA[...,0]
        
        if TimePos == 4:
            vector_x = Malla[0,0,:]
            vector_y = Malla[1,:,0]
            if vector_x[0] == vector_x[1]:
                vector_x = Malla[0,:,0]
                vector_y = Malla[1,0,:]
            
            grad0 = np.gradient(mode0, vector_x,  vector_y, axis=(1,2))
            grad0 = np.array(grad0).transpose((1,0,2,3))
        
            Part1 = np.zeros(Malla.shape,dtype = np.complex128)
            Part1[0,:,:] = mode0[0,:,:]*grad0[0,0,:,:]+mode0[1,:,:]*grad0[0,1,:,:]
            Part1[1,:,:] = mode0[0,:,:]*grad0[1,0,:,:]+mode0[1,:,:]*grad0[1,1,:,:]
        
            ff1 = np.linalg.norm(Part1,axis = 0)
           
            gradd = np.gradient(moded, vector_x, vector_y, axis=(1,2))
            gradd = np.array(gradd).transpose((1,0,2,3))
        
            PartD = np.zeros(Malla.shape,dtype = np.complex128)
            PartD[0,:,:] = moded[0,:,:]*gradd[0,0,:,:]+moded[1,:,:]*gradd[0,1,:,:]
            PartD[1,:,:] = moded[0,:,:]*gradd[1,0,:,:]+moded[1,:,:]*gradd[1,1,:,:]
        
       
            grada = np.gradient(modea, vector_x, vector_y, axis=(1,2))
            grada = np.array(grada).transpose((1,0,2,3))
            
            PartA = np.zeros(Malla.shape,dtype = np.complex128)
            PartA[0,:,:] = modea[0,:,:]*grada[0,0,:,:]+modea[1,:,:]*grada[0,1,:,:]
            PartA[1,:,:] = modea[0,:,:]*grada[1,0,:,:]+modea[1,:,:]*grada[1,1,:,:]
        
            ff2= np.sqrt((abs(PartD[0,:,:])**2+abs(PartD[1,:,:])**2)*(abs(PartA[0,:,:])**2+abs(PartA[1,:,:])**2))
        
            NLSens = np.abs(GRFA[1,1])* np.real(ff1)/np.amax(np.real(ff1)) + 2*ff2/np.amax(ff2)
        
            fig = plt.figure()
            plt.contourf(Malla[0,:,:],Malla[1,:,:],np.real(ff1)/np.amax(np.real(ff1)))
            plt.title('||M||*||M||')
            plt.savefig(f'{path0}/{filen}/MM.png')
            plt.close(fig)   
            
            fig = plt.figure()
            plt.contourf(Malla[0,:,:],Malla[1,:,:],ff2/np.amax(ff2))
            plt.title('||N||*||N*||')
            plt.savefig(f'{path0}/{filen}/NN.png')
            plt.close(fig)   
        
            fig = plt.figure()
            plt.contourf(Malla[0,:,:],Malla[1,:,:],NLSens)
            plt.savefig(f'{path0}/{filen}/NLSens.png')
            plt.close(fig)
            
            index = np.argwhere(NLSens == np.amax(NLSens))
            X_index = Malla[0,index[0,0],index[0,1]]
            Y_index = Malla[1,index[0,0],index[0,1]]
            print(f'The maximum value of the sensititvuity locates at: X  = {X_index} Y = {Y_index} ')
            
        elif TimePos == 5:
            
            vector_x = Malla[0,0,0,:]
            if vector_x[0] == vector_x[1]:
                vector_x = Malla[0,0,:,0]
                if vector_x[0] == vector_x[1]:
                    vector_x = Malla[0,:,0,0]
                    vector_y = Malla[1,0,0,:]
                    vector_z = Malla[2,0,:,0]
                    if vector_y[0] == vector_y[1]:
                        vector_y = Malla[1,0,:,0]
                        vector_z = Malla[2,0,0,:]
                else:
                    vector_y = Malla[1,0,0,:]
                    vector_z = Malla[2,:,0,0]
                    if vector_y[0] == vector_y[1]:
                        vector_y = Malla[1,:,0,0]
                        vector_z = Malla[2,0,0,:]
            else:
                vector_y = Malla[1,0,:,0]
                vector_z = Malla[2,:,0,0]
                if vector_y[0] == vector_y[1]:
                    vector_y = Malla[1,:,0,0]
                    vector_z = Malla[2,0,:,0]
            
            grad0 = np.gradient(mode0, vector_x, vector_y, vector_z, axis=(1,2,3))
            grad0 = np.array(grad0).transpose((1,0,2,3,4))
        
            Part1 = np.zeros(Malla.shape,dtype = np.complex128)
            Part1[0,:,:] = mode0[0,:,:]*grad0[0,0,:,:]+mode0[1,:,:]*grad0[0,1,:,:]+mode0[2,:,:]*grad0[0,2,:,:]
            Part1[1,:,:] = mode0[0,:,:]*grad0[1,0,:,:]+mode0[1,:,:]*grad0[1,1,:,:]+mode0[2,:,:]*grad0[1,2,:,:]
            Part1[2,:,:] = mode0[0,:,:]*grad0[2,0,:,:]+mode0[1,:,:]*grad0[2,1,:,:]+mode0[2,:,:]*grad0[2,2,:,:]
            ff1 = np.linalg.norm(Part1,axis=0)
           
            gradd = np.gradient(moded, vector_x, vector_y, vector_z, axis=(1,2,3))
            gradd = np.array(gradd).transpose((1,0,2,3,4))
        
            PartD = np.zeros(Malla.shape,dtype = np.complex128)
            PartD[0,:,:] = moded[0,:,:]*gradd[0,0,:,:]+moded[1,:,:]*gradd[0,1,:,:]+moded[2,:,:]*gradd[0,2,:,:]
            PartD[1,:,:] = moded[0,:,:]*gradd[1,0,:,:]+moded[1,:,:]*gradd[1,1,:,:]+moded[2,:,:]*gradd[1,2,:,:]
            PartD[2,:,:] = moded[0,:,:]*gradd[2,0,:,:]+moded[1,:,:]*gradd[2,1,:,:]+moded[2,:,:]*gradd[2,2,:,:]
       
            grada = np.gradient(modea, vector_x, vector_y, vector_z, axis=(1,2,3))
            grada = np.array(grada).transpose((1,0,2,3,4))
            
            PartA = np.zeros(Malla.shape,dtype = np.complex128)
            PartA[0,:,:] = modea[0,:,:]*grada[0,0,:,:]+modea[1,:,:]*grada[0,1,:,:]+modea[2,:,:]*grada[0,2,:,:]
            PartA[1,:,:] = modea[0,:,:]*grada[1,0,:,:]+modea[1,:,:]*grada[1,1,:,:]+modea[2,:,:]*grada[1,2,:,:]
            PartA[2,:,:] = modea[0,:,:]*grada[2,0,:,:]+modea[1,:,:]*grada[2,1,:,:]+modea[2,:,:]*grada[2,2,:,:]
        
            ff2= np.sqrt((abs(PartD[0,:,:,:])**2+abs(PartD[1,:,:,:])**2+abs(PartD[2,:,:,:])**2)*(abs(PartA[0,:,:,:])**2+abs(PartA[1,:,:,:])**2+abs(PartA[2,:,:,:])**2))
            NLSens =  np.abs(GRFA[1,1])* np.real(ff1)/np.amax(np.real(ff1)) + 2*ff2/np.amax(ff2)
        
        
            fig = plt.figure()
            plt.contourf(Malla[0,:,:,int(Malla.shape[3] / 2)],Malla[1,:,:,int(Tensor.shape[3] / 2)],NLSens[:,:,int(Tensor.shape[3] / 2)])
            plt.savefig(f'{path0}/{filen}/NLSens.png')
            plt.close(fig)
            
            index = np.argwhere(NLSens == np.amax(NLSens))
            X_index = Malla[0,index[0,0],index[0,1],index[0,2]]
            Y_index = Malla[1,index[0,0],index[0,1],index[0,2]]
            Z_index = Malla[2,index[0,0],index[0,1],index[0,2]]
            print(f'The maximum value of the sensititvuity locates at: X  = {X_index} Y = {Y_index} Z = {Z_index}')
            
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/NLSens.npy',NLSens)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"NLSens": NLSens}
            file_mat = str(f'{path0}/{filen}/NLSens.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')