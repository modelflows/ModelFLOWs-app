# Import functions
import data_fetch
import numpy as np
import DMDd
import os
import sys
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import contour_anim
import hosvd

def HODMD_func(Tensor, deltaT, varepsilon1, varepsilon2, d, SNAP, path0, decision, decision1):
    # Create new folder:
    if not os.path.exists(f'{path0}/HODMD_solution'):
        os.mkdir(f'{path0}/HODMD_solution')
    if not os.path.exists(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}'):
        os.mkdir(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}')
    if not os.path.exists(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes'):
        os.mkdir(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes')
        
    Time = np.linspace(0,SNAP-1,num=SNAP)*deltaT
    Tensor = Tensor[..., :SNAP]
    Tensor0 = Tensor.copy()
    dims = Tensor.ndim
    shape = Tensor.shape

    dims_prod = np.prod(shape[:-1])
    Tensor = np.reshape(Tensor, (dims_prod, shape[-1]))
    
    notone=0
    for i in range(0,np.size(np.shape(Tensor))):
        if np.shape(Tensor)[i] != 1:
            notone=notone+1

    if notone<=2:
        if d==1:
            st.write('Performing DMD')
            st.write("")
            [u,Amplitude,Eigval,GrowthRate,Frequency,DMDmode] = DMDd.dmd1(Tensor, Time, varepsilon1, varepsilon2)
            st.write("")
            st.write('DMD complete')
            dt=Time[1]-Time[0]
            icomp=complex(0,1)
            mu=np.zeros(np.size(GrowthRate),dtype=np.complex128)
            for iii in range(0,np.size(GrowthRate)):
                mu[iii] = np.exp(np.dot(dt,GrowthRate[iii]+np.dot(icomp,Frequency[iii])))
            Reconst=DMDd.remake(u,Time,mu)
        else:
            st.write('Performing HODMD')
            st.write("")
            [u,Amplitude,Eigval,GrowthRate,Frequency,DMDmode] = DMDd.hodmd(Tensor, d, Time, varepsilon1, varepsilon2)
            st.write("")
            st.write('HODMD complete')
            dt=Time[1]-Time[0]
            icomp=complex(0,1)
            mu=np.zeros(np.size(GrowthRate),dtype=np.complex128)
            for iii in range(0,np.size(GrowthRate)):
                mu[iii] = np.exp(np.dot(dt,GrowthRate[iii]+np.dot(icomp,Frequency[iii])))
            Reconst=DMDd.remake(u,Time,mu)

    
    newshape = []
    newshape.append(shape[:-1])
    newshape.append(DMDmode.shape[-1])
    newshape = list(newshape[0]) + [newshape[1]]
    DMDmode = np.reshape(DMDmode, np.array(newshape))
    Reconst = np.reshape(Reconst, shape)
    RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
    st.write(f'\n###### Relative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%')
        

    GRFreqAmp = np.zeros((np.size(GrowthRate),3))
    GRFreqAmp[:,0] = GrowthRate[:]
    GRFreqAmp[:,1] = Frequency[:]
    GRFreqAmp[:,2] = Amplitude[:]

 
    st.write("")
    st.table(pd.DataFrame(GRFreqAmp, columns=['GrowthRate', 'Frequency', 'Amplitude']))
    st.write("")

    # Reconstruction:
    np.save(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/TensorReconst',Reconst)

    # GrowthRate, Frequency and Amplitude:
    np.save(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/GrowthRateFrequencyAmplitude',GRFreqAmp)

    ## Calculate DMD modes:
    np.save(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmode',DMDmode)

    fig, ax = plt.subplots()
    ax.plot(Frequency,GrowthRate, 'k+')
    plt.yscale('log') 
    ax.set_xlabel('Frequency ($\omega_{n}$)')
    ax.set_ylabel('GrowthRate ($\delta_{n}$)')
    plt.savefig(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/FrequencyGrowthRate.png')
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.plot(Frequency, Amplitude/np.amax(Amplitude),'r+')
    ax.set_yscale('log')           # Logarithmic scale in Tensor0.shape[1]/3 axis
    ax.set_xlabel('Frequency ($\omega_{n}$)')
    ax.set_ylabel('Amplitude divided by max. amplitude ($a_{n}$)')
    plt.savefig(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/FrequencyAmplitude.png')
    st.pyplot(fig)
        
    st.info(f'Saving DMDmodes plots to {path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes\n')
    for ModComp in range(DMDmode.shape[0]):
        for ModeNum in range(DMDmode.shape[-1]):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}', fontsize = 16)
            ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
            ax[0].set_title('Real part', fontsize = 14)
            ax[0].set_xlabel('X', fontsize = 10)
            ax[0].set_ylabel('Y', fontsize = 10)

            ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
            ax[1].set_title('Imaginary part', fontsize = 14)
            ax[1].set_xlabel('X', fontsize = 10)
            ax[1].set_ylabel('Y', fontsize = 10)
            fig.tight_layout()
            plt.savefig(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmodeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
            plt.close(fig)

    st.info('Plotting first 3 DMD modes for all velocity components')
    for ModComp in range(DMDmode.shape[0]):
        for ModeNum in range(3):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}', fontsize = 16)
            ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
            ax[0].set_title('Real part', fontsize = 14)
            ax[0].axis('off')

            ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
            ax[1].set_title('Imaginary part', fontsize = 14)
            ax[1].axis('off')
            fig.tight_layout()
            st.pyplot(fig)

    st.info('Plotting Real data vs Reconstruction')
    for i in range(Tensor0.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize = (20, 7))
        fig.suptitle(f'Real Data vs Reconstruction - Vel. comparison - Component {i+1}')
        ax[0].contourf(Tensor0[i, :, :, 0])
        ax[0].scatter(Tensor0.shape[2]/4, Tensor0.shape[1]/3, c ='black', s=50)
        ax[1].plot(Time[:], Reconst[i, int(Tensor0.shape[1]/3), int(Tensor0.shape[2]/4), :], 'k-x', label = 'Reconstructed Data')
        ax[1].plot(Time[:], Tensor0[i, int(Tensor0.shape[1]/3), int(Tensor0.shape[2]/4), :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        fig.tight_layout()
        plt.savefig(f'{path0}/HODMD_solution/d_{d}_tol_{varepsilon1}/OrigReconst_{i}.png')
        st.pyplot(fig)


    if decision == 'Yes':
        st.info('Comparison of Real Data vs Reconstruction')
        if decision1 != 'V':
            contour_anim.animated_plot(path0, Tensor0, vel=0, Title = 'Real Data U velocity')
            contour_anim.animated_plot(path0, Reconst, vel=0, Title = 'Reconstruction U velocity')
        if decision1 != 'U':
            contour_anim.animated_plot(path0, Tensor0, vel=1, Title = 'Real Data V velocity')
            contour_anim.animated_plot(path0, Reconst, vel=1, Title = 'Reconstruction V velocity')

def HODMD_IT(Tensor, SNAP, varepsilon1, varepsilon2, deltaT, TimePos, d, path0, decision, decision1, type_, iters):
    # Create new folder:
    if not os.path.exists(f'{path0}/{type_}_HODMD_solution'):
        os.mkdir(f"{path0}/{type_}_HODMD_solution")
    if not os.path.exists(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}'):
        os.mkdir(f"{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}")

    ## Tensor dimension - number of snapshots:

    Tensor = Tensor[..., 0:SNAP]

    TensorR = Tensor.copy()

    Time = np.linspace(0, SNAP-1, num = SNAP) * deltaT

    path0 = os.getcwd()

    dims = Tensor.ndim

    ## ALGORITHM:

    ## ITERATIVE:
    nn0 = np.shape(Tensor)
    nn = np.array(nn0)
    nn[1:np.size(nn)] = 0
    st.write("")
    for zz in range(0,iters):
        if iters > 1:
            st.write(f'##### Iteration number: {zz+1}')
        
        if zz != 0:
            del S,U,Frequency,GrowthRate,hatT,hatMode,sv,TensorR,nnin
            TensorR = TensorReconst
        
        ## Perform HOSVD decomposition to calculate the reduced temporal matrix: hatT
        nnin = nn
        nnin = tuple(nnin)
        st.write('Performing HOSVD')
        [hatT,U,S,sv,nn1,n,TT] = hosvd.HOSVD(TensorR,varepsilon1,nn,nn0,TimePos)
        st.write('HOSVD complete')
        st.write("")

        ## Perform HODMD to the reduced temporal matrix hatT:
        st.write('Performing HODMD')
        st.write("")
        [hatMode,Amplitude,Eigval,GrowthRate,Frequency] = DMDd.hodmd_IT(hatT,d,Time,varepsilon1,varepsilon2)
        st.write("")
        st.write('HODMD complete')
        
        ## Reconstruct the original Tensor using the DMD expansion:
        TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)
        
        ## Print outcome:

        RRMSE = np.linalg.norm(np.reshape(Tensor-TensorReconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
        st.write(f'\n###### Relative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%')
        st.write("""
                    
                    """)

        GRFreqAmp = np.zeros((np.size(GrowthRate),3))
        GRFreqAmp[:,0] = GrowthRate[:]
        GRFreqAmp[:,1] = Frequency[:]
        GRFreqAmp[:,2] = Amplitude[:]

        ## Break the loop when the number of singular values is the same in two consecutive iterations:

        num = 0
        for ii in range(1,np.size(nn1)):
            if nnin[ii]==nn1[ii]:
                num = num+1
        
        if num==np.size(nn1)-1:
            break
        nn = nn1
        print('\n')

    ## Save the reconstruction of the tensor and the Growth rates, frequencies and amplitudes:

    # Reconstruction:
    np.save(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/TensorReconst',TensorReconst)


    st.table(pd.DataFrame(GRFreqAmp, columns=['GrowthRate', 'Frequency', 'Amplitude']))
    st.write("")

    # GrowthRate, Frequency and Amplitude:
    np.save(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/GrowthRateFrequencyAmplitude',GRFreqAmp)


    ## Calculate DMD modes:
    N = np.shape(hatT)[0]
    DMDmode = DMDd.modes_IT(N,hatMode,Amplitude,U,S,nn1,TimePos)
    np.save(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmode',DMDmode)

    # Frequency vs absolute value of GrowthRate:

    fig1, ax1 = plt.subplots()
    ax1.plot(Frequency,np.abs(GrowthRate), 'k+')
    ax1.set_yscale('log')           # Logarithmic scale in y axis
    ax1.set_xlabel('Frequency ($\omega_{m}$)')
    ax1.set_title('Frequency vs. GrowthRate')
    ax1.set_ylabel('Absolute value of GrowthRate (|$\delta_{m}$|)')
    plt.savefig(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/FrequencyGrowthRate.png')
            
    st.pyplot(fig1)

    # Frequency vs amplitudes:
    fig2, ax2 = plt.subplots()
    ax2.plot(Frequency, Amplitude,'r+')
    ax2.set_yscale('log')           # Logarithmic scale in y axis
    ax2.set_xlabel('Frequency ($\omega_{m}$)')
    ax2.set_ylabel('Amplitude ($a_{m}$)')
    ax2.set_title('Frequency vs. Amplitude')
    plt.savefig(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/FrequencyAmplitude.png')

    st.pyplot(fig2)

    if not os.path.exists(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes'):
        os.mkdir(f"{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes")

    st.info(f'Saving DMD mode plots to {path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes\n')
    for ModComp in range(DMDmode.shape[0]):
        for ModeNum in range(DMDmode.shape[-1]):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}')
            ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
            ax[0].set_title('Real part')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')

            ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
            ax[1].set_title('Imaginary part')
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            fig.tight_layout()
            plt.savefig(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmodeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
            plt.close(fig)

    st.info('Plotting first 3 DMD modes for all velocity components')
    for ModComp in range(DMDmode.shape[0]):
        for ModeNum in range(3):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}', fontsize = 16)
            ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
            ax[0].set_title('Real part', fontsize = 14)
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')
            ax[0].axis('off')

            ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
            ax[1].set_title('Imaginary part', fontsize = 14)
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            ax[1].axis('off')
            fig.tight_layout()
            st.pyplot(fig)

    st.info('Plotting Real data vs Reconstruction')
    for i in range(Tensor.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize = (20, 7))
        fig.suptitle(f'Real Data vs Reconstruction - Vel. comparison - Component {i+1}')
        ax[0].contourf(Tensor[i, :, :, 0])
        ax[0].scatter(Tensor.shape[2]/4, Tensor.shape[1]/3, c ='black', s=50)
        ax[1].plot(Time[:], TensorReconst[i, int(Tensor.shape[1]/3), int(Tensor.shape[2]/4), :], 'k-x', label = 'Reconstructed Data')
        ax[1].plot(Time[:], Tensor[i, int(Tensor.shape[1]/3), int(Tensor.shape[2]/4), :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        fig.tight_layout()
        plt.savefig(f'{path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}/OrigReconst_{i}.png')
        st.pyplot(fig)
        
    if decision == 'Yes':
        st.info('Comparison of Real Data vs Reconstruction')
        if decision1 != 'V':
            contour_anim.animated_plot(path0, Tensor, vel=0, Title = 'Real Data U velocity')
            contour_anim.animated_plot(path0, TensorReconst, vel=0, Title = 'Reconstruction U velocity')
        if decision1 != 'U':
            contour_anim.animated_plot(path0, Tensor, vel=1, Title = 'Real Data V velocity')
            contour_anim.animated_plot(path0, TensorReconst, vel=1, Title = 'Reconstruction V velocity')


def menu():
    st.title("Higher Order Dynamic Mode Decomposition, HODMD")
    st.write('''
HODMD is a data-driven method generally used in fluid dynamics and in the analysis of complex non-linear dynamical systems modeling several complex industrial applications.
There are two available option: HODMD and Multi-dimensional HODMD. The difference between these is that the first algorithm applies SVD on the input tensor and, therefore,
is suitable for matrices, while Multi-dimensional HODMD applies HOSVD on the input data, making it ideal for tensors.
''')

    path0 = os.getcwd()

    hodmd_type = st.selectbox("Which type of HODMD would you like to perform", ("HODMD", "Multi-dimensional HODMD"))

### CLASSIC HODMD ##
    if hodmd_type == "HODMD":
        st.write(" ## HODMD - Parameter Configuration")

        # 1. Fetch data matrix or tensor
        selected_data = 'Tensor_cylinder_Re100.mat'
        Tensor = data_fetch.fetch_data(path0, selected_data)

        varepsilon1 = st.number_input(f'Introduce SVD tolerance value', min_value = 0.0, max_value = 0.5, value = float(1e-10), step = 0.001, format="%.10f")
        varepsilon1 = float(varepsilon1)

        # 2. Select tolerance. DMDd tolerance = SVD tolerance
        varepsilon2 = st.number_input(f'Introduce HODMD tolerance value', min_value = 0.0, max_value = 0.5, value = float(1e-3), step = 0.001, format="%.10f")
        varepsilon2 = float(varepsilon2)

        max_snaps = Tensor.shape[-1]
        
        # 4. Select number of snapshots
        SNAP = st.number_input(f'Introduce number of snapshots (must be lower than {max_snaps})', max_value = max_snaps, value = max_snaps, step = 1)
        SNAP = int(SNAP)

        # 3. d parameter
        d = st.number_input('Introduce number of HODMD windows (d)', min_value = 1, max_value = None, step = 1, value = int(np.round(SNAP/10)))
        st.info(f'Interval of recommended number of HODMD windows (d): [{int(np.round(SNAP/10))}, {int(np.round(SNAP/2))}]. Other values are accepted')
        d = int(d)

        # 5. Time gradient (time step can be equidistant or not)
        deltaT = st.number_input('Introduce time gradient (deltaT)', min_value = 0.01, max_value = None, step = 0.01, value = 1., format = "%.2f")
        deltaT = float(deltaT)
                 
        dims = Tensor.ndim

        if dims > 2:
            decision = st.radio('Represent real data and reconstruction videos', ('Yes', 'No'))
            if decision == 'Yes':
                decision1 = st.radio('For U, V or both velocities', ('U', 'V', 'Both'))
            else:
                decision1 = None

        go = st.button('Calculate')

        if go:
            with st.spinner('Please wait until the run is complete'):
        
                HODMD_func(Tensor, deltaT, varepsilon1, varepsilon2, d, SNAP, path0, decision, decision1)

                st.success("Run complete!")

            st.warning(f"All files have been saved to HODMD_solution/d_{d}_tol_{varepsilon1}")

            st.info("Press 'Refresh' to run a new case")
            Refresh = st.button('Refresh')
            if Refresh:
                st.stop()


### ITERATIVE HODMD ###
    if hodmd_type == "Multi-dimensional HODMD":
        st.write('The Multi-dimensional HODMD has two variants: Non-iterative and iterative.')
        type = st.selectbox('Select a version', ('Non-iterative', 'Iterative'))
        if type == 'Iterative':
            st.write(" ## Iterative Multi-dimensional HODMD - Parameter Configuration")
            iters = 1000
            type_ = 'it'
        if type == 'Non-iterative':
            st.write(" ## Non-iterative Multi-dimensional HODMD - Parameter Configuration")
            iters = 1
            type_ = 'non-it'

        # 1. Select data matrix/tensor
        selected_file = 'Tensor_cylinder_Re100.mat'
        Tensor = data_fetch.fetch_data(path0, selected_file)

        varepsilon1 = st.number_input(f'Introduce SVD tolerance value', min_value = 0.0, max_value = 0.5, value = float(1e-10), step = 0.001, format="%.10f")
        varepsilon1 = float(varepsilon1)

        # 2. Tolerances. DMDd tolerance = SVD tolerance
        varepsilon2 = st.number_input(f'Introduce HODMD tolerance value', min_value = 0.0, max_value = 0.5, value = float(1e-3), step = 0.001, format="%.10f")
        varepsilon2 = float(varepsilon2)

        max_snaps = Tensor.shape[-1]
            
        # 4. Number of snapshots
        SNAP = st.number_input(f'Introduce number of snapshots (must be lower than {max_snaps})', max_value = max_snaps, value = max_snaps, step = 1)
        
        SNAP = int(SNAP)

        # 3. d parameter
        d = st.number_input('Introduce number of HODMD windows (d)', min_value = 1, max_value = None, step = 1, value = int(np.round(SNAP/10)))
        st.info(f'Interval of recommended number of HODMD windows (d): [{int(np.round(SNAP/10))}, {int(np.round(SNAP/2))}]. Other values are accepted')
        d = int(d)

        # 5. Time gradient (time step can be equidistant or not)
        deltaT = st.number_input('Introduce time gradient (deltaT)', min_value = 0.01, max_value = None, step = 0.01, value = 1., format = "%.2f")
        deltaT = float(deltaT)

        # 6. Position of temporal variable (only if data is in tensor form)

        TimePos = int(Tensor.ndim)

        decision = st.radio('Represent real data and reconstruction videos', ('Yes', 'No'))
        if decision == 'Yes':
            decision1 = st.radio('For U, V or both velocities', ('U', 'V', 'Both'))
        else:
            decision1 = None

        go = st.button('Calculate')

        if go:
            with st.spinner('Please wait until the run is complete'):
                
                HODMD_IT(Tensor, SNAP, varepsilon1, varepsilon2, deltaT, TimePos, d, path0, decision, decision1, type_, iters)

                st.success('Run complete!')

            st.warning(f"All files have been saved to {path0}/{type_}_HODMD_solution/d_{d}_tol_{varepsilon1}")
    
            st.info("Press 'Refresh' to run a new case")
            Refresh = st.button('Refresh')
            if Refresh:
                st.stop()
