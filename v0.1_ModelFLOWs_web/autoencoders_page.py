import streamlit as st
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
import math

import os
import data_fetch

# Load scikit-learn 
from sklearn.model_selection import train_test_split

# Load TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from IPython.display import HTML

def animated_plot(path0, Tensor, vel, Title):
    '''
    Function that creates an animated contourf plot
    
    Args:
        path0 - path where the graph is to be saved to later be loaded on the streamlit app
        Tensor - data file
        vel - velocity variable: 0 for x velocity; 1 for y velocity
        Title - Title for the graph (i.e. original data, reconstructed data...)
    '''
    if Tensor.shape[-1] > Tensor.shape[0]:
        Tensor = tf.transpose(Tensor, perm = [3, 1, 2, 0])

    frames = Tensor.shape[0]

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.contourf(Tensor[i, :, :, vel]) 
        ax.set_title(Title)

    interval = 2     
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval*1e+2, blit = False)

    plt.show()

    with open(f"{path0}/animation.html","w") as f:
        print(anim.to_html5_video(), file = f)

    HtmlFile = open(f"{path0}/animation.html", "r")

    source_code = HtmlFile.read()

    components.html(source_code, height = 700, width=900)

def mean_absolute_percentage_error(y_true, y_pred): 
    epsilon = 1e-10 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

def smape(A, F):
    return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

def RRMSE(Tensor0, Reconst):
    RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
    return(RRMSE)

def menu():
    st.title("Autoencoders Model")
    st.write("""
An autoencoder is an unsupervised learning technique for neural networks that learns efficient data representations (encoding) by training the network to ignore signal “noise.” 
""")
    st.write(" ## Autoencoders DNN Model - Parameter Configuration")

    path0 = os.getcwd()

    if not os.path.exists(f'{path0}/Autoencoders_model_solution'):
        os.mkdir(f"{path0}/Autoencoders_model_solution")

    # Menu parameters

    Tensor = 'Tensor_cylinder_Re100.mat'

    tensor = data_fetch.fetch_data(path0, Tensor)

    _, ny, nx, nt = tensor.shape

    # AEs parameters
    hyp_batch = st.slider("Select Batch Size", 2, 256, value = 64, step=2)
    hyp_epoch = st.slider("Select Epochs", 0, 250, value = 200, step=1)
    encoding_dim = st.number_input("Select autoencoder dimensions", 0, None, value = 10, step=1)

    # True: plot reconstruction vs original data
    dec2 = st.radio(f"Plot autoencoder modes", ("Yes", "No"))
    if dec2 == "Yes":
        decision2 = True
        AEs_toplot = encoding_dim 
    else:
        decision2 = False

    dec1 = st.radio("Original data vs Reconstruction video", ("Yes", "No"))
    if dec1 == "Yes":
        decision1 = True
    else:
        decision1 = False  

    go = st.button("Calculate")

    if go:

        with st.spinner("Please wait while the model is being trained"):
            RedStep=1

            tensor = tensor[:,0::RedStep,0::RedStep,0::RedStep]
            ntt = tensor.shape[3]    # Nt
            ncomp = tensor.shape[0]
            ny = tensor.shape[1]
            nx = tensor.shape[2]

            print(tensor.shape)

            min_val = np.array(2)

            min_val = np.zeros(ncomp,)
            max_val=np.zeros(ncomp,)
            range_val=np.zeros(ncomp,)
            std_val=np.zeros(ncomp,)

            tensor_norm = np.zeros(tensor.shape)

            for j in range(ncomp):
                min_val   [j] = np.amin(tensor[j,:,:,:])
                max_val   [j] = np.amax(tensor[j,:,:,:])
                range_val [j] = np.ptp(tensor[j,:,:,:])
                std_val   [j] = np.std(tensor[j,:,:,:])
                tensor_norm[j,:,:,:] = (tensor[j,:,:,:]-min_val[j])/range_val[j]
            
            keras.backend.clear_session()
            nxy2 = ny * nx * ncomp
            dim = nt

            TT=tensor_norm.transpose((3,1,2,0))
            ntt, ny, nx, ncomp= TT.shape

            X_scale = np.reshape(TT,(dim,nxy2),order='F')
            X_scale = X_scale.transpose((1,0))

            input_img = Input(shape=(dim,))
            encoded = Dense(encoding_dim, activation='linear')(input_img)
            decoded = Dense(dim, activation='linear')(encoded)
            autoencoder = Model(input_img, decoded)
            encoder = Model(input_img, encoded)
            decoder = Model(encoded, decoded)

            autoencoder.compile(optimizer='adam', loss='mse')
            # Get a summary
            

            x_train, x_test, y_train, y_test = train_test_split(X_scale, 
                                                    X_scale, 
                                                    test_size=0.1) 

            # CALLBACK : Early Stoping
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta = 0.001)

            autoencoder.summary(print_fn=lambda x: st.text(x))

            History = autoencoder.fit(x_train, x_train,
                            epochs=hyp_epoch,
                            batch_size=hyp_batch,
                            callbacks = [callback],
                            shuffle=True,
                            validation_data=(x_test, x_test))
            
            st.write(History.history)
            st.write("")

            st.success("The model has been trained!")

            # get convergence history
            loss_linlin = History.history['loss']
            loss_v = History.history['val_loss']

            # Prediction of the encoding/decoding
            z = encoder.predict(X_scale)
            x_tilde = autoencoder.predict(X_scale)

            #Check error
            Err=np.linalg.norm(x_tilde-X_scale)/np.linalg.norm(X_scale)
            
            st.write(f'###### Neural Network RRMSE with all modes: {Err*100:.2f}%')
            
            rrmse_ = np.zeros((z.shape[1],))
            contrib = np.zeros((z.shape[1],))

            for nm in range(0,encoding_dim):
                z1=np.zeros(z.shape);
                z1[:,0:nm] = z[:,0:nm];
                xx = decoder.predict(z1)
                rrmse = RRMSE(X_scale,xx)
                st.write(f'Adding mode number: {nm+1} - Updated RRMSE: {rrmse*100:.3f}%')
                rrmse_[nm]=rrmse

            fig, ax = plt.subplots(figsize=(6, 4)) # This creates the figure
            ax.plot(np.arange(0, encoding_dim)+1,rrmse_*100)
            ax.scatter(np.arange(0, encoding_dim)+1,rrmse_*100)
            ax.set_title("RRMSE value per added mode")
            ax.set_xlabel('Mode added')
            ax.set_ylabel('RRMSE')
            plt.savefig(f'{path0}/Autoencoders_model_solution/RRMSE.png')
            st.pyplot(fig)

            incr_rr_=np.zeros((encoding_dim,))
            for idel in range(0,encoding_dim):
                array=np.arange(0,encoding_dim)
                array=np.delete(array,idel)

                z1=np.zeros(z.shape)
                z1[:,array]=z[:,array]
                xx = decoder.predict(z1)
                rrmse = RRMSE(X_scale,xx)
                
                incr_rr = rrmse- RRMSE(X_scale,x_tilde)
                incr_rr_[idel]=incr_rr
                st.write(f'Deleted mode: {idel+1} - Updated RRMSE: {rrmse*100:.3f}% - RRMSE increase (compared to all modes RRMSE): {incr_rr*100:.3f}%')

            fig, ax = plt.subplots(figsize=(6, 4)) # This creates the figure
            ax.plot(np.arange(1,encoding_dim+1),incr_rr_*100)
            ax.scatter(np.arange(1,encoding_dim+1),incr_rr_*100)
            ax.set_title("RRMSE when mode 'n' is deleted")
            ax.set_xlabel('Mode deleted')
            ax.set_ylabel('RRMSE')
            plt.savefig(f'{path0}/Autoencoders_model_solution/relative_RRMSE.png')
            st.pyplot(fig)

            ## indexes for the sorting

            I = np.argsort(incr_rr_)
            modes_sorted =  np.flip(I)

            ## modes 
            z_sort = z[:,modes_sorted];

            SP_z_sort = z_sort

            if decision1 == True:
                
                lim = int(dim/ncomp)
                RR = x_tilde[:,0:dim:1]

                ntt = x_tilde.shape[1]

                RR = np.transpose(RR,(1,0))
                ZZT = np.reshape(RR,(ntt,ny,nx,ncomp),order='F')
                ZZT = np.transpose(ZZT,(3,1,2,0))

                np.save(f'{path0}/Autoencoders_model_solution/SM_Reconst', ZZT)
                                # CONTOUR AUTOENCODER -- CHECK RECONSTRUCTION

                plot_titles = {0: 'U velocity',
                               1: 'V Velocity'}

                velocity = 0

                tensor_copy = tensor.copy()
                ZZT_copy = ZZT.copy()

                animated_plot(path0, tensor_copy, vel = velocity, Title = f'Real data - {plot_titles[velocity]}')
                animated_plot(path0, ZZT_copy, vel = velocity, Title = f'Sp. Modes recontructed data - {plot_titles[velocity]}')

                velocity = 1

                animated_plot(path0, tensor_copy, vel = velocity, Title = f'Real data - {plot_titles[velocity]}')
                animated_plot(path0, ZZT_copy, vel = velocity, Title = f'Sp. Modes recontructed data - {plot_titles[velocity]}')

            if decision2 == True : # PLOT AUTOENCODER NUMBER XX
                if not os.path.exists(f'{path0}/Autoencoders_model_solution/Autoencoders'):
                    os.mkdir(f"{path0}/Autoencoders_model_solution/Autoencoders")
                st.info(f'Showing all autoencoder components')
                st.info(f'All generated plots will be saved to {path0}/Autoencoders_model_solution/Autoencoders')  
    
                for AEnum in range(min(AEs_toplot,encoding_dim)):

                    MODE=np.transpose(z_sort[:,AEnum])

                    AE=MODE[0:int(nx*ny*ncomp)]

                    Rec_AE=np.reshape(AE,(ny,nx,ncomp),order='F')
                    
                    fig, ax = plt.subplots(1, Rec_AE.shape[-1], figsize = (20, 7))
                    plt.suptitle(f'Autoencoder mode {AEnum+1}')
                    for i in range(Rec_AE.shape[-1]):
                        ax[i].contourf(Rec_AE[..., i])
                        ax[i].set_title(f"Component {i+1}")
                    
                    plt.savefig(f"{path0}/Autoencoders_model_solution/Autoencoders/AE_{i+1}.png")
                    st.pyplot(fig)

                np.save(f'{path0}/Autoencoders_model_solution/RecAE', Rec_AE)

        st.success('Run complete!')
        st.warning(f'All files have been saved to {path0}/Autoencoders_model_solution')
        st.info("Press 'Refresh' to run a new case")
        Refresh = st.button('Refresh')
        if Refresh:
            st.stop()

                    
