import data_fetch
import os
import numpy as np
import hosvd
import matplotlib.pyplot as plt
import streamlit as st
import contour_anim


def hosvd_fun(SNAP, varepsilon1, Tensor, TimePos, path0, decision, decision1):
    # Create new folder:
    if not os.path.exists(f'{path0}/HOSVD_solution'):
        os.mkdir(f"{path0}/HOSVD_solution")
    if not os.path.exists(f'{path0}/HOSVD_solution/tol_{varepsilon1}'):
        os.mkdir(f"{path0}/HOSVD_solution/tol_{varepsilon1}")
    if not os.path.exists(f'{path0}/HOSVD_solution/tol_{varepsilon1}/SVD_modes'):
        os.mkdir(f"{path0}/HOSVD_solution/tol_{varepsilon1}/SVD_modes")
    
    Tensor0 = Tensor.copy()
    shapeTens = list(np.shape(Tensor))
    shapeTens[-1] = SNAP
    Tensor = np.zeros(shapeTens)

    Tensor[..., :] = Tensor0[..., 0:SNAP]

    nn0 = np.array(Tensor.shape)
    nn = np.array(nn0)
    nn[1:np.size(nn)] = 0 

    hatT, U, S, sv, nn1, n, TT = hosvd.HOSVD(Tensor, varepsilon1, nn, nn0, TimePos)
    st.write("")

    # Graph to visualize sigular values vs modes

    fig, ax = plt.subplots()
    markers = ['yo', 'gx', 'r*', 'bv', 'y+']
    labels = ["Variable Singular Values",
              "X Space Singular Values",
              "Y Space Singular Values",
              "Z Space Singular Values",
              "Time Singular Values"]

    sub_axes = plt.axes([.6, .6, .25, .2]) 

    if np.array(n).size == 4:
        labels.remove("Z Space Singular Values")

    for i in range(np.array(n).size):
        ax.plot(sv[0, i] / sv[0, i][0], markers[i])
        sub_axes.plot(sv[0, i][:nn1[i]] / sv[0, i][0], markers[i])
        
    ax.hlines(y=varepsilon1, xmin = 0, xmax=np.array(n).max(), linewidth=2, color='black', label = f'SVD tolerance: {varepsilon1}')   

    ax.set_yscale('log')           # Logarithmic scale in y axis
    sub_axes.set_yscale('log')
    ax.set_xlabel('SVD modes')
    ax.set_ylabel('Singular values')
        
    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('SVD modes vs. Singular values', fontsize = 14)
    sub_axes.set_title('Retained singular values', fontsize = 8)
    plt.savefig(f'{path0}/HOSVD_solution/tol_{varepsilon1}/SingularValues.png')
    st.pyplot(fig)

    U = U[0].copy()

    st.info('Calculating SVD modes')
    SVD_modes = np.einsum('ijkl, ai, bj, ck -> abcl', S, U[0], U[1], U[2])

    st.info(f'All SVD mode plots will be saved to {path0}/HOSVD_solution/tol_{varepsilon1}/SVD_modes')
    for ModComp in range(SVD_modes.shape[0]):
        for ModeNum in range(SVD_modes.shape[-1]):
            fig, ax = plt.subplots()
            ax.contourf(SVD_modes[ModComp,:,:,ModeNum])
            ax.set_title(f'Component {ModComp+1} - SVD mode {ModeNum+1}', fontsize = 14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            fig.tight_layout()
            plt.savefig(f'{path0}/HOSVD_solution/tol_{varepsilon1}/SVD_modes/Comp_{ModComp+1}_SVDmodes_{ModeNum+1}.png')
            plt.close(fig)

    st.info(f'Showing first 3 SVD modes for {SVD_modes.shape[0]} components')

    for ModComp in range(SVD_modes.shape[0]):
        for ModeNum in range(0, 3):    
            fig, ax = plt.subplots()
            ax.contourf(SVD_modes[ModComp,:,:,ModeNum])
            ax.set_title(f'Component {ModComp+1} - SVD mode {ModeNum+1}', fontsize = 14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.axis('off')
            fig.tight_layout()
            st.pyplot(fig)
            
    np.save(f'{path0}/HOSVD_solution/tol_{varepsilon1}/TensorReconst', TT)
    np.save(f'{path0}/HOSVD_solution/tol_{varepsilon1}/SVD_modes', SVD_modes)

    if decision == 'Yes':
        st.write('Comparison of Real Data vs Reconstruction')
        if decision1 != 'V':
            contour_anim.animated_plot(path0, Tensor, vel=0, Title = 'Real Data U velocity')
            contour_anim.animated_plot(path0, TT, vel=0, Title = 'Reconstruction U velocity')
        if decision1 != 'U':
            contour_anim.animated_plot(path0, Tensor, vel=1, Title = 'Real Data V velocity')
            contour_anim.animated_plot(path0, TT, vel=1, Title = 'Reconstruction V velocity') 


def menu():
    st.title("Higher-Order Singular Value Decomposition, HOSVD")
    st.write("""#
In multilinear algebra, the Higher-Order Singular Value Decomposition (HOSVD) of a tensor is a specific orthogonal Tucker decomposition. 
It may be regarded as one generalization of the matrix singular value decomposition. 
It has applications in computer vision, computer graphics, machine learning, scientific computing, and signal processing. 
    """)
    st.write(" ## HOSVD - Parameter Configuration")

    path0 = os.getcwd()

    # 1. Data selection
    selected_file = 'Tensor_cylinder_Re100.mat'
    Tensor = data_fetch.fetch_data(path0, selected_file)

    # 2. Tolerances. DMDd tolerance = SVD tolerance
    varepsilon1 = st.number_input(f'Introduce SVD tolerance', min_value = 0.000, max_value = 0.5, value = 0.0001, step = 0.0001, format="%.10f")
    varepsilon1 = float(varepsilon1)

    SNAP = st.number_input(f'Introduce number of snapshots (must be lower than {Tensor.shape[-1]})', max_value = int(Tensor.shape[-1]), value = int(Tensor.shape[-1]), step = 1)
    SNAP = int(SNAP)

    decision = st.radio('Represent real data and reconstruction videos', ('Yes', 'No'))
    if decision == 'Yes':
        decision1 = st.radio('For U, V or both velocities', ('U', 'V', 'Both'))
    else:
        decision1 = None

    TimePos = int(Tensor.ndim)

    go = st.button('Calculate')

    if go:
        with st.spinner('Please wait for the run to complete'):
        
            hosvd_fun(SNAP, varepsilon1, Tensor, TimePos, path0, decision, decision1)
            st.success('Run complete!')
        
        st.warning(f"All files have been saved to {path0}/HOSVD_solution/tol_{varepsilon1}")
        st.info("Press 'Refresh' to run a new case")
        Refresh = st.button('Refresh')
        if Refresh:
            st.stop()