import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st
import data_fetch

def svdtrunc(A):
    U, S, V = np.linalg.svd(A, full_matrices = False)
    return U, S, V

def menu():
    st.title("Singular Value Decomposition, SVD")
    st.write("""#
SVD (Singular Value Decomposition) is a mathematical technique used to analyze and reduce the dimensionality of databases. 
It decomposes a given database into a set of orthogonal modes, capturing the most important patterns and variability in the data. 
This allows for efficient storage, compression, and analysis of the information, enabling tasks such as visualization, model reduction, and optimization.
    """)
    st.write(" ## SVD - Parameter Configuration")
    path0 = os.getcwd()

    selected_file = 'Tensor_cylinder_Re100.mat'
    Tensor = data_fetch.fetch_data(path0, selected_file)

    n_modes = st.number_input('Select number of modes to retain during SVD', min_value = 0, max_value = None, value = 18, step = 1)

    go = st.button('Calculate')
    if go:
        with st.spinner('Please wait until the run is complete'):
            shape = Tensor.shape
            if not os.path.exists(f'{path0}/SVD_solution'):
                os.mkdir(f'{path0}/SVD_solution')
            if not os.path.exists(f'{path0}/SVD_solution/n_modes_{n_modes}'):
                os.mkdir(f'{path0}/SVD_solution/n_modes_{n_modes}')
            if not os.path.exists(f'{path0}/SVD_solution/n_modes_{n_modes}/svd_modes'):
                os.mkdir(f'{path0}/SVD_solution/n_modes_{n_modes}/svd_modes')

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
                
            ax.set_yscale('log')       # Logarithmic scale in y axis
            ax.set_xlabel('SVD modes')
            ax.set_ylabel('Singular values')
            ax.set_title('SVD modes vs. Singular values')
            plt.savefig(f'{path0}/SVD_solution/svd_modes_plot.png', bbox_inches='tight')
            st.pyplot(fig)

            newshape = []
            newshape.append(shape[:-1])
            newshape.append(svd_modes.shape[-1])
            newshape = list(newshape[0]) + [newshape[1]]
            svd_modes = np.reshape(svd_modes, np.array(newshape))
            Reconst = np.reshape(Reconst, shape)

            RRMSE = np.linalg.norm(np.reshape(Tensor-Reconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
            st.write(f'\n###### Relative mean square error made during reconstruction: {np.round(RRMSE*100, 3)}%\n')
        
            st.info(f'All SVD mode plots will be saved to {path0}/SVD_solution/n_modes_{n_modes}/svd_modes')
            st.info(f'Showing first 3 SVD modes for {svd_modes.shape[0]} components')
            
            for ModComp in range(svd_modes.shape[0]):
                for ModeNum in range(svd_modes.shape[-1]):
                    fig, ax = plt.subplots()
                    ax.contourf(svd_modes[ModComp,:,:,ModeNum].real)
                    ax.set_title(f'SVD modes - Component {ModComp+1} Mode Number {ModeNum+1}', fontsize = 14)
                    ax.set_xlabel('X', fontsize = 10)
                    ax.set_ylabel('Y', fontsize = 10)
                    ax.axis('off')
                    fig.tight_layout()
                    plt.savefig(f'{path0}/SVD_solution/n_modes_{n_modes}/svd_modes/Comp_{ModComp+1}_svdmode_{ModeNum+1}.png')
                    if ModeNum <= 2:
                        st.pyplot(fig)
                    

            st.success('Run complete!')
        
        np.save(f'{path0}/SVD_solution/n_modes_{n_modes}/svd_modes.npy', svd_modes)
        np.save(f'{path0}/SVD_solution/n_modes_{n_modes}/Reconst.npy', Reconst)

        st.warning(f"All files have been saved to {path0}/SVD_solution/n_modes_{n_modes}")

        st.info("Press 'Refresh' to run a new case")
        Refresh = st.button('Refresh')
        if Refresh:
            st.stop()



