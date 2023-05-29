import os
import data_fetch
import numpy as np
from scipy.interpolate import griddata
from numpy import linalg as LA
import matplotlib.pyplot as plt
import streamlit as st
import hosvd 

def Gappy_SVD(A_gappy, Tensor, m, method, decision_1, output_1, output_2, path0, method_ = None):
    N = sum(np.isnan(A_gappy.flatten()))

    st.write('Making initial reconstruction')
    # Initial Repairing
    if decision_1 == 'zeros':
        A0_1 = np.nan_to_num(A_gappy, nan = 0)
    elif decision_1 == 'mean':
        A0_1 = np.nan_to_num(A_gappy, nan = 0)
        A0_1 = np.nan_to_num(A_gappy,nan=sum(A0_1.flatten())/(A0_1.size-N))
    elif decision_1 == 'tensor_interp':
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
        
    st.write('Initial reconstruction complete')
               
    A_s = A0_1.copy()
    MSE_gaps = np.zeros(500)

    st.write('Performing HOSVD. Please wait')
    for ii in range(500):

        if method == 'svd':
            [U,S,V]=LA.svd(A_s)
            S = np.diag(S)
            A_reconst = U[:,0:m] @ S[0:m,0:m] @ V[0:m,:]
        elif method == 'hosvd':
            n = m*np.ones(np.shape(A_s.shape))
            A_reconst = hosvd.HOSVD_function(A_s,n)[0]
            
            
        MSE_gaps[ii] = LA.norm(A_reconst[np.isnan(A_gappy)]-A_s[np.isnan(A_gappy)])/N
        
        if ii>3 and MSE_gaps[ii]>=MSE_gaps[ii-1]:
            break
        else:
            A_s[np.isnan(A_gappy)] = A_reconst[np.isnan(A_gappy)]

    st.write('HOSVD complete')

    if output_1 == 'yes':
        sv0 = hosvd.HOSVD_function(A0_1,n)[3]
        sv = hosvd.HOSVD_function(A_s,n)[3]
        cmap = plt.cm.get_cmap('jet')
        rgba = cmap(np.linspace(0,1,A_s.ndim))
        fig, ax = plt.subplots()
        for i in range(A_s.ndim):
            ax.semilogy(sv0[0,i]/sv0[0,i][0], linestyle = 'none', marker = 'x',color = rgba[i])
            ax.semilogy(sv[0,i]/sv[0,i][0], linestyle = 'none', marker = '+', color = rgba[i])
        plt.legend(['Initial Reconstruction','Final Reconstruction'])
        plt.savefig(f'{path0}/Data_repair_solution_nmodes_{m}_fill_{decision_1}/svd_decay.png')
        st.pyplot(fig)

    Tensor0 = Tensor[:A_s.shape[0], :A_s.shape[1], :A_s.shape[2], :A_s.shape[3]].copy()
    RRMSE = np.linalg.norm(np.reshape(Tensor0 - A_s ,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
    st.write(f'###### Error made during reconstruction: {np.round(RRMSE*100, 3)}%\n')

    if output_2 == 'yes':
        for var in range(A_gappy.shape[0]):
            fig, ax = plt.subplots(figsize=(20, 7), num = f'CLOSE TO CONTINUE RUN - Initial data for component {var+1}')
            heatmap = ax.imshow(A_gappy[var, ..., 0], cmap='coolwarm')
            ax.set_title(f'Initial gappy data - Component {var+1}', fontsize = 14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.axis('off')
            heatmap.set_clim(np.nanmin(A_gappy[var, ..., 0]), np.nanmax(A_gappy[var, ..., 0]))
            st.pyplot(fig)

        # Initial and final reconstruction vs ground truth   
        fig, ax = plt.subplots(1, 3, figsize = (20, 7))
        plt.suptitle('Data comparison for vel. component 1 and snapshot 1', fontsize = 16)
        ax[0].contourf(A_gappy[0, ..., 0])
        ax[0].set_title('Initial data', fontsize = 14)
        ax[0].axis('off')

        ax[1].contourf(A_reconst[0, ..., 0])
        ax[1].set_title('Reconstructed data', fontsize = 14)
        ax[1].axis('off')

        ax[2].contourf(Tensor[0, ..., 0])
        ax[2].set_title('Ground truth', fontsize = 14)
        ax[2].axis('off')
        fig.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots(1, 3, figsize = (20, 7))
        plt.suptitle('Data comparison for vel. component 2 and snapshot 1', fontsize = 16)
        ax[0].contourf(A_gappy[1, ..., 0])
        ax[0].set_title('Initial data', fontsize = 14)
        ax[0].axis('off')

        ax[1].contourf(A_reconst[1, ..., 0])
        ax[1].set_title('Reconstructed data', fontsize = 14)
        ax[1].axis('off')

        ax[2].contourf(Tensor[1, ..., 0])
        ax[2].set_title('Ground truth', fontsize = 14)
        ax[2].axis('off')
        fig.tight_layout()
        st.pyplot(fig)
        
    np.save(f'{path0}/Data_repair_solution_nmodes_{m}_fill_{decision_1}/Repaired_data.npy', A_reconst)


def Gappy_augment(A_down, enhance, selected_database, path0):
    m = [1, 2]

    enhance_ = 2**enhance
    
    A_d = A_down

    st.write('Enhancing data resolution') 

    for ii in range(enhance):   
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
 
        A_d = hosvd.tprod(S, Udens)
        A_d = hosvd.HOSVD_function(A_d,n)[0]
    
    A_reconst = A_d

    st.write('Data resolution enhanced') 

    st.write(f'Initial data resolution: {A_down.shape}')
    st.write(f'New data resolution: {A_reconst.shape}')

    st.info('Plotting first snapshot for both velocity components')

    for v in range(A_down.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize = (20, 7))
        plt.suptitle(f'Superresolution for vel. component {v+1} and snapshot 1', fontsize = 16)
        ax[0].contourf(A_down[v, :,  :, 0])
        ax[0].set_title(f'Original data', fontsize = 14)
        ax[0].xaxis.grid(True, zorder = 0)
        ax[0].yaxis.grid(True, zorder = 0)
        
        ax[1].contourf(A_reconst[v, :,  :, 0])
        ax[1].set_title(f'Enhanced data', fontsize = 14)
        ax[1].xaxis.grid(True, zorder = 0)
        ax[1].yaxis.grid(True, zorder = 0)
        fig.tight_layout()
        st.pyplot(fig)

    np.save(f"{path0}/Superresolution_{selected_database}_enhancement_{enhance_}/Enhanced_data.npy", A_reconst)

def menu():
    st.title("Data Reconstruction")
    st.write("""
This module uses modal decomposition techniques to reconstruct datasets with missing data or enhance the resolution of a dataset.
This is acheived by combining interpolation techniques with HOSVD.
    """)

    path0 = os.getcwd()

    option = st.selectbox('Select an option', ('Repair', 'Superresolution'))

    if option == 'Repair':

        st.write(" ## Data Repairing")

        # 1. Fetch data matrix or tensor
        selected_database = 'Gappy_Tensor_cylinder_Re100.mat'
        A_gappy = data_fetch.fetch_data(path0, selected_database)
        selected_database1 = 'Tensor_cylinder_Re100.mat'
        Tensor = data_fetch.fetch_data(path0, selected_database1)

        # 2. Number of retained modes
        m = st.number_input(f'Number of modes to retain during repair', min_value = 0, max_value = None, value = 18, step = 1)
        m = int(m)

        if A_gappy.ndim > 2:
            method = 'hosvd'
        elif A_gappy.ndim <= 2:
            method = 'svd'

        decision = ('interpolation', 'zeros', 'mean')

        decision_1 = st.selectbox('Select a decision to fill missing data', decision)

        if decision_1 == 'interpolation':
            decision_1 = 'tensor_interp'
            method_ = st.radio('Select an interpolation method', ('nearest', 'linear', 'cubic'))
        else:
            method_ = None
        st.write('### Output Configuration')

        output_1 = st.radio('Visualize decay of the singular values before/after', ('yes', 'no'))
        output_2 = st.radio('Visualize comparison of the surfaces', ('yes', 'no'))

        go = st.button('Calculate')

        if not os.path.exists(f'{path0}/Data_repair_solution_nmodes_{m}_fill_{decision_1}'):
            os.mkdir(f"{path0}/Data_repair_solution_nmodes_{m}_fill_{decision_1}")

        if go: 
            with st.spinner('Please wait for the run to complete'):
            
                Gappy_SVD(A_gappy, Tensor, m, method, decision_1, output_1, output_2, path0, method_)

            st.success('Run complete!')
            st.warning(f"All files have been saved to {path0}/Data_repair_solution_nmodes_{m}_fill_{decision_1}")
        
            st.info("Press 'Refresh' to run a new case")
            Refresh = st.button('Refresh')
            if Refresh:
                st.stop()

    elif option == 'Superresolution':
        st.write(" ## Data Superresolution")
        selected_database = 'DS_30_Tensor_cylinder_Re100.mat'
        A_down = data_fetch.fetch_data(path0, selected_database)

        # 2. Number of retained modes
        augment = st.number_input(f'Select the enhancement scale (2^factor)', min_value = 0, max_value = None, value = 4, step = 1)
        augment = int(augment)

        augment_ = 2**augment

        go = st.button('Calculate')

        selected_database = selected_database.replace(".mat", "")

        if not os.path.exists(f'{path0}/Superresolution_{selected_database}_enhancement_{augment_}'):
            os.mkdir(f"{path0}/Superresolution_{selected_database}_enhancement_{augment_}")

        if go: 
            with st.spinner('Please wait for the run to complete'):
            
                Gappy_augment(A_down, augment, selected_database, path0)

            st.success('Run complete!')
            st.warning(f"All files have been saved to {path0}/Superresolution_{selected_database}_enhancement_{augment_}")
        
            st.info("Press 'Refresh' to run a new case")
            Refresh = st.button('Refresh')
            if Refresh:
                st.stop()