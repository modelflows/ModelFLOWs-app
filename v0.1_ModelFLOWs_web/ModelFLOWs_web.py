import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu


# Load Pages
import SVD_page
import hodmd_page
import hosvd_page
import gappy_page
import autoencoders_page
import hybRNN_model_page
import hybCNN_model_page
import hodmd_pred_page
import DLsuperresolution_page
import FullDL_page
import hdf5storage
import os
import tensorflow as tf
import pickle
path0 = os.getcwd()

# Page configuration
st.set_page_config(page_title='ModelFLOWs-app', page_icon = 'thumbnail.png', layout = 'wide', initial_sidebar_state = 'auto')


# Sidebar menu layout
with st.sidebar:
    selected = option_menu("ModelFLOWs-app", ["About", 'Models'], 
        icons=['info-square', 'diagram-3'], menu_icon="house", default_index=0)

if selected == 'About':
    col1, col2, col3 = st.columns(3)
    image = Image.open('modelflows.jpeg')
    col2.image(image, width=200)
    st.title('About the ModelFLOWs application')
    st.write('This application has been developed by the ModelFLOWs research group')

    st.write("""
ModelFLOWs is a research group whose main promoter and tutor is Soledad Le Clainche and was formed at the School of Aeronautics Engineering at Universidad Politécnica de Madrid (UPM). 
This team uses different data-driven methods, i.e. reduced order models (ROMs) or neural networks (NN), to generate, study and predict databases related to complex flows (turbulent, reactive, etc.).    
    """)

    st.write("""
This web-browser version of ModelFLOWs-app is a demo, used to show the structure of the application and how each algorithm works. The datasets used are 'Tensor_cylinder_Re100.mat',
a 2-Dimensional CFD simulation of flow passing a cylinder at Reynolds 100, and Tensor.pkl, a 3-Dimensional CFD simulation.
            """)
    selection = 'Tensor_cylinder_Re100.mat'
    Tensor_ = hdf5storage.loadmat(f'{path0}/{selection}')
    data = list(Tensor_.values())[-1]
    st.write(f"""
The Tensor_cylinder_Re100.mat has the following shape:

nv, ny, nx, nt: {data.shape}, where:
            """)
    st.markdown("""
- nv is the number of velocity components, also known as variables
- nx is the number of points in the Y axis that form the mesh
- ny is the number of points in the X axis that form the mesh
- nt is the number of temporal components, also known as snapshots
    """)

    st.write('Here is a brief example of what this data looks like')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Tensor_cylinder_Re100.mat')
    ax.contourf(data[0, :, :, 0])
    ax.set_title('First velocity component of the first snapshot')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Tensor_cylinder_Re100.mat')
    ax.contourf(data[1, :, :, 0])
    ax.set_title('Second velocity component of the first snapshot')
    st.pyplot(fig)

    st.write("Other variants of this database are also provided such as:")
    st.markdown('''
- DS_30_Tensor_cylinder_Re100.mat, a downscaled version of the database, which means that the spatial mesh is smaller and, therefore, contains less data
- Gappy_Tensor_cylinder_Re100.mat, a version with the same shape as the original database, but with missing data which is replaced with NaN (Not A Number) values
    ''')

    st.write('Here is what these databases look like')

    selection = 'DS_30_Tensor_cylinder_Re100.mat'
    Tensor_ = hdf5storage.loadmat(f'{path0}/{selection}')
    ds = list(Tensor_.values())[-1]

    selection = 'Gappy_Tensor_cylinder_Re100.mat'
    Tensor_ = hdf5storage.loadmat(f'{path0}/{selection}')
    gappy = list(Tensor_.values())[-1]

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: DS_30_Tensor_cylinder_Re100.mat')
    ax.contourf(ds[0, :, :, 0])
    ax.set_title('First velocity component of the first snapshot')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: DS_30_Tensor_cylinder_Re100.mat')
    ax.contourf(ds[1, :, :, 0])
    ax.set_title('Second velocity component of the first snapshot')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Gappy_Tensor_cylinder_Re100.mat')
    ax.contourf(gappy[0, :, :, 0])
    ax.set_title('First velocity component of the first snapshot')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Gappy_Tensor_cylinder_Re100.mat')
    ax.contourf(gappy[1, :, :, 0])
    ax.set_title('Second velocity component of the first snapshot')
    st.pyplot(fig)

    with open(f'{path0}/Tensor.pkl', 'rb') as file:
            Tensor=pickle.load(file)
    
    Tensor = tf.transpose(Tensor, (0, 2, 1, 3, 4))

    st.write(f'''
The Tensor.pkl database is used exclusively in the Deep Learning Reconstruction module. This tensor has the following shape:

nv, nx, ny, nz, nt: {Tensor.shape}, where:
            ''')
    
    st.markdown("""
- nv is the number of velocity components, also known as variables
- nx is the number of points in the Y axis that form the mesh
- ny is the number of points in the X axis that form the mesh
- nz is the number of points in the Z axis that form the mesh
- nt is the number of temporal components, also known as snapshots
    """)

    st.write('Here is a brief example of what this data looks like')

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Tensor.pkl')
    ax.contourf(Tensor[0, :, :, 0, 0])
    ax.set_title('First velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Tensor.pkl')
    ax.contourf(Tensor[1, :, :, 0, 0])
    ax.set_title('Second velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: Tensor.pkl')
    ax.contourf(Tensor[2, :, :, 0, 0])
    ax.set_title('Third velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    st.write('''A variant to this database is also provided. In this case, a downscaled version: DS_30_Tensor.pkl''')

    st.write('Here is what this database looks like')

    with open(f'{path0}/DS_30_Tensor.pkl', 'rb') as file:
        Tensords=pickle.load(file)

    Tensords = tf.transpose(Tensords, (0, 2, 1, 3, 4))

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: DS_30_Tensor.pkl')
    ax.contourf(Tensords[0, :, :, 0, 0])
    ax.set_title('First velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: DS_30_Tensor.pkl')
    ax.contourf(Tensords[1, :, :, 0, 0])
    ax.set_title('Second velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.suptitle('Dataset: DS_30_Tensor.pkl')
    ax.contourf(Tensords[2, :, :, 0, 0])
    ax.set_title('Third velocity component of the first snapshot - XY plane')
    st.pyplot(fig)

    st.write("##### Check out the application tutorials at https://modelflows.github.io/modelflowsapp/tutorials/")

if selected == 'Models':
    # Sidebar with options
    page = st.sidebar.selectbox("Select an option", ("Modal Decomposition", "Deep Learning"))
    if page == 'Modal Decomposition':
        action = st.sidebar.selectbox("Select an action", ("Pattern detection", "Reconstruction", "Prediction"))
        if action == "Pattern detection":
            option = st.sidebar.radio("Select an algorithm", ("SVD", "HOSVD", "HODMD"))

        elif action == "Reconstruction":
            option = "Gappy SVD"

        elif action == "Prediction":
            option = 'pred'

    if page == 'Deep Learning':
        mdl = st.sidebar.selectbox("Select an option", ("Pattern detection", "Reconstruction", "Prediction"))

        if mdl == "Prediction":
            option = st.sidebar.selectbox("Select a model type", ("Full DL model", "Hybrid DL Model"))
            if option == 'Full DL model':
                model_type = st.sidebar.radio('Select an architecture', ('CNN', 'RNN'))
                model_type = model_type.lower()
            elif option == 'Hybrid DL Model':
                model_type1 = st.sidebar.radio('Select an architecture', ('CNN', 'RNN'))
                
                if model_type1 == 'RNN':
                    hybRNN_model_page.menu()

                if model_type1 == 'CNN':
                    hybCNN_model_page.menu()

        elif mdl == "Pattern detection":
            option = "Autoencoder DNN Model"
            
        elif mdl == "Reconstruction":
            option = "DNN Reconstruction Model"

    # Load the selected algorithm or model menu
    if option == "SVD":
        SVD_page.menu()

    if option == "HODMD":
        hodmd_page.menu()

    if option == "HOSVD":
        hosvd_page.menu()

    if option == "Gappy SVD":
        gappy_page.menu()

    if option == "Autoencoder DNN Model":
        autoencoders_page.menu()

    if option == "DNN Reconstruction Model":
       DLsuperresolution_page.menu()

    if option == 'Full DL model':
        FullDL_page.menu(model_type) 

    if option == 'pred':
        hodmd_pred_page.menu()

