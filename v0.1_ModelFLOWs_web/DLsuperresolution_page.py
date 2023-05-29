import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten
import scipy
import scipy.io
import os 
from math import floor
import streamlit as st
import data_fetch


def menu(): 
    tf.keras.backend.set_floatx('float64')

    st.title("DNN Superresolution Model")
    st.write("""
This model is a hybrid, since it combines modal decomposition (precisely, SVD) with deep learning. The decomposed (factorized) tensor is 
used to train this model to enhance the resolution of downsampled (lower scaled) data to the same resolution as the ground truth data.
 """)
    st.write(" ## DNN Superresolution Model - Parameter Configuration")

    path0 = os.getcwd()

    def downsampling():
        wantedfile = st.selectbox('Please select a data file', ('DS_30_Tensor_cylinder_Re100.mat', 'DS_30_Tensor.pkl'))
        st.info('"DS_30_Tensor_cylinder_Re100.mat" is a downsampled 2D database, while "DS_30_Tensor.pkl" is a downsampled 3D database')

        if wantedfile == 'DS_30_Tensor_cylinder_Re100.mat':
            Tensor = data_fetch.fetch_data(path0, wantedfile)
            var_sel = np.transpose(Tensor,(3,2,1,0))
            _, ny_sel, nx_sel, _ = var_sel.shape
            

        elif wantedfile == 'DS_30_Tensor.pkl':
            with open(f'{path0}/{wantedfile}', 'rb') as file:
                Tensor=pickle.load(file)
            var_sel = np.transpose(Tensor,(4,1,2,3,0))
            _, ny_sel, nx_sel, _, _ = var_sel.shape
            
        
        return var_sel, ny_sel, nx_sel

    def scale_val(x, min_val, max_val):
        return((x - min_val)/(max_val - min_val))


    def descale_val(x, min_val, max_val):
        return(x * (max_val - min_val) + min_val)


    def custom_scale(tensor):
        min_val = np.amin(tensor)
        max_val = sum(np.amax(np.abs(tensor),axis=1))
        med_val = np.mean(tensor)
        range_val = np.ptp(tensor)
        std_val =np.std(tensor)
        tensor_norm = XUds / max_val_XUds
        
        return tensor_norm

    def mean_absolute_percentage_error(y_true, y_pred): 
        epsilon = 1e-10 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

    def smape(A, F):
        return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

    def RRMSE(Tensor0, Reconst):
        RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
        return(RRMSE)

    def error_tables(Xr_test, Xr_hat, Xr_sel,tabname1,tabname2, path0, filename, nvar):
        results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE', 'MAPE'],columns=['All'])
        for i in range(1):
            results_table.iloc[0,i] = mean_squared_error( Xr_test.flatten(), Xr_hat.flatten())
            results_table.iloc[1,i] = mean_absolute_error(Xr_test.flatten(), Xr_hat.flatten())
            results_table.iloc[2,i] = median_absolute_error( Xr_test.flatten(), Xr_hat.flatten())
            results_table.iloc[3,i] = r2_score(Xr_test.flatten(), Xr_hat.flatten())
            results_table.iloc[4,i] = smape( Xr_test.flatten(), Xr_hat.flatten())
            results_table.iloc[5,i] = RRMSE( np.reshape(Xr_test,(-1,1)), np.reshape(Xr_hat,(-1,1)))*100
            results_table.iloc[6,i] = mean_absolute_percentage_error( Xr_test.flatten(), Xr_hat.flatten())
        df1 = pd.DataFrame(results_table)
        df1.to_csv(f'{path0}/{filename}/{tabname1}.csv') 

        num_layers = Xr_sel.shape[3]

        cols = []
        for i in range(1, nvar + 1):
            cols.append(f'var{i}')
        
        results_table2 = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE','MAPE'],columns=cols)
        for i in range(num_layers):
            results_table2.iloc[0,i] = mean_squared_error( Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
            results_table2.iloc[1,i] = mean_absolute_error(Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
            results_table2.iloc[2,i] = median_absolute_error( Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
            results_table2.iloc[3,i] = r2_score(Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
            results_table2.iloc[4,i] = smape( Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
            results_table2.iloc[5,i] = RRMSE( np.reshape(Xr_test[...,i],(-1,1)), np.reshape(Xr_hat[...,i],(-1,1)))*100
            results_table2.iloc[6,i] = mean_absolute_percentage_error( Xr_test[...,i].flatten(), Xr_hat[...,i].flatten())
        df2 = pd.DataFrame(results_table2)
        df2.to_csv(f'{path0}/{filename}/{tabname2}.csv') 


        return results_table, results_table2

    def figures_results_ylim(n_snap,var, yg, xg, y, x, Xr_test, Xr_sel, Xr_hat, figname):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))
        
        fig.tight_layout()

        im1 = ax[0].contourf(yg,xg,np.transpose(Xr_test[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[0].set_ylim(bottom=0,top=16)
        ax[0].set_axis_off()
        ax[0].set_title('Ground truth', fontsize = 14)

        im2 = ax[1].contourf(y,x,np.transpose(Xr_sel[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[1].set_ylim(bottom=0,top=16)
        ax[1].set_axis_off()
        ax[1].set_title('Initial data', fontsize = 14)

        im3 = ax[2].contourf(yg,xg,np.transpose(Xr_hat[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[2].set_ylim(bottom=0,top=16)
        ax[2].set_axis_off()
        ax[2].set_title('Enhanced data', fontsize = 14)
    
        fig.tight_layout()
        plt.savefig(figname) 
        if n_snap == 0: 
            st.pyplot(fig)
 
    def videos_results(videoname, var, n_train, test_ind,xg, yg, x, y, Xr_test, Xr_hat, Xr_sel):
        from matplotlib import animation
        figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))
        figure.tight_layout()

        def animation_function(i):
            figure.suptitle(f'Snapshot: {n_train+i}', fontsize=20)
            im1 = ax[0].contourf(yg,xg,np.transpose(Xr_test[i,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[0,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[0,...,var],(1,0))))
            ax[0].set_axis_off()
            ax[0].set_title('Ground truth', fontsize = 14)
            
            im2 = ax[1].contourf(y,x,np.transpose(Xr_sel[i,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[0,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[0,...,var],(1,0))))
            ax[1].set_axis_off()
            ax[1].set_title('Original data', fontsize = 14)
            
            im3 = ax[2].contourf(yg,xg,np.transpose(Xr_hat[i,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[0,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[0,...,var],(1,0))))
            ax[2].set_axis_off()
            ax[2].set_title('Enhanced data', fontsize = 14)
                    
            figure.subplots_adjust(right=0.8, top=0.9)
            cbar_ax = figure.add_axes([0.85, 0.09, 0.01, 0.85])
            cbar = figure.colorbar(im1, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=20)


        anim = animation.FuncAnimation(figure, animation_function, frames=test_ind.size, interval=1000, blit=False)
        writergif = animation.PillowWriter(fps=5) 
        anim.save(videoname, writer=writergif)

        # Showing the video slows down the performance

    # Inputs
    # Load database


    [Xr_sel, ny_sel, nx_sel]=downsampling()

    if Xr_sel.ndim == 4:
        Xr = data_fetch.fetch_data(path0, "Tensor_cylinder_Re100.mat")
        Xr=np.transpose(Xr,(3,2,1,0))
        nt, ny, nx, nvar = Xr.shape
        dims = 'Data2D'

    elif Xr_sel.ndim == 5:
        with open(f'{path0}/Tensor.pkl', 'rb') as file:
            Tensor=pickle.load(file)
        Xr=np.transpose(Tensor,(4,1,2,3,0))
        nt, ny, nx, nz, nvar = Xr.shape
        dims = 'Data3D'

    decision1 = st.radio('Apply a first scaling to the input data?', ('No', 'Yes'))
    if not decision1 or decision1.strip().lower() in ['n', 'no']:
        decision1='no-scaling'
        
    elif decision1.strip().lower() in ['y', 'yes']:
        decision1='scaling'

    if decision1 == 'scaling':
        decision2 = st.radio('Output scaling graphs?', ('No', 'Yes'))
        if not decision2 or decision2.strip().lower() in ['n', 'no']:
            decision2 = 'no'
        elif decision2.strip().lower() in ['y', 'yes']:
            decision2 = 'yes'
            
    else:
        decision2 = 'no'

    decision3 = st.radio('Apply a second scaling to the input data?', ('No', 'Yes'))
    if not decision3 or decision3.strip().lower() in ['n', 'no']:
        decision3='no'
        
    elif decision3.strip().lower() in ['y', 'yes']:
        decision3='yes'
            
    decision4 = st.radio('Shuffle train and test data?', ('No', 'Yes'))
    if not decision4 or decision4.strip().lower() in ['n', 'no']:
        decision4='no-shuffle'
        
    elif decision4.strip().lower() in ['y', 'yes']:
        decision4 = 'shuffle'

    
    bs = st.slider('Select the data batch size', min_value = 0, max_value = 32, value = 16, step = 1)
    bs = int(bs)

    neurons = st.slider('Select the number of neurons per layer', min_value = 1, max_value = 100, value = 50, step = 1)
    neurons = int(neurons)         
        
    act_func = st.selectbox('Select an activation function for the hidden layers', ("elu", "relu", "tanh", "elu", "sigmoid", "linear"))
    act_fun2 = st.selectbox('Select an activation function for the output layers', ("tanh", "elu", "relu", "linear", "elu", "sigmoid"))
    
    learn_rate = st.number_input('Select the model learning rate', min_value = 0.0001, max_value = 0.1, value = 0.005, step = 0.0001, format = '%.4f')
    learn_rate = float(learn_rate)

    loss_function='mse'

    n_train = st.number_input(f'Select the number of samples for the training data. Continue with {round(int(nt)*0.8)} samples (80% aprox. RECOMMENDED)', min_value = 1, max_value = nt, value = round(int(nt)*0.8), step = 1)
    n_train = int(n_train)

    epoch = st.slider('Select the number of training epoch', min_value = 1, max_value = 200, value = 150, step = 10)
    epoch = int(epoch)

    decision5 = st.radio('Would you like to save the results?', ('Yes', 'No'))
    if not decision5 or decision5.strip().lower() in ['y', 'yes']:
        decision5='yes'

    elif decision5.strip().lower() in ['n', 'no']:
        decision5 = 'no'

    decision7 = st.radio('Data comparison plot?', ('Yes', 'No'))
    if not decision7 or decision7.strip().lower() in ['y', 'yes']:
        decision7='yes'
        
    elif decision7.strip().lower() in ['n', 'no']:
        decision7 = 'no'

    decisionx = st.radio('Create comparison videos of the ground truth vs. the predicted data?', ('No', 'Yes'))
    if not decisionx or decisionx.strip().lower() in ['n', 'no']:
        decisionx='no'
        
    elif decisionx.strip().lower() in ['y', 'yes']:
        decisionx = 'yes'
            

    filename0 = f'DNN_superresolution_solution'

    if not os.path.exists(f'{path0}/{filename0}'):
        os.mkdir(f'{path0}/{filename0}')

    if Xr_sel.ndim == 4:
        dat = '2DcylinderRe100'

    elif Xr_sel.ndim == 5:
        dat = '3Dcylinder'

    filename1 = f'{dat}_{decision1}_{decision4}'

    if not os.path.exists(f'{path0}/{filename0}/{filename1}'):
        os.mkdir(f'{path0}/{filename0}/{filename1}')

    filename = f'{filename0}/{filename1}'

    go = st.button('Calculate')

    if go:
        with st.spinner('Please wait for the run to complete'):
            st.write("")
            st.write('Downsampled data summary')
            st.write('Xr_sel: ', Xr_sel.shape) 
            st.write('ny_sel =', ny_sel) 
            st.write('nx_sel =', nx_sel)
            st.write("")

            st.write('Ground truth data summary')
            st.write('Xr: ', Xr.shape) 
            st.write('ny =', ny) 
            st.write('nx =', nx)
            st.write("")

            if Xr.ndim==3: 
                nt=100
                Xr= np.repeat(Xr[np.newaxis,...],nt, axis=0) 
                Xr_sel= np.repeat(Xr_sel[np.newaxis,...],nt, axis=0) 


            if dims=='Data3D':
                Xr_sel=np.reshape(Xr_sel,(nt,ny_sel,nx_sel*nz,nvar),order='F')
                Xr=np.reshape(Xr,(nt,ny,nx*nz,nvar),order='F')

            # SCALING

            if decision1=='scaling': 
                var_sel_min = []
                var_sel_max = []
                st.write('Scaling data')
                for i in range(nvar):
                    var_sel_min.append(np.min(Xr_sel[...,i]))
                    var_sel_max.append(np.max(Xr_sel[...,i]))
                    st.write(f'var{i+1}_sel_min: {var_sel_min[i]}, var{i+1}_sel_max: {var_sel_max[i]}')
                
                var_sel_min = np.array(var_sel_min)
                var_sel_max = np.array(var_sel_max)
                
                var_sel_sc = []
                for i in range(nvar):
                    var_sel_sc.append(np.array(scale_val(Xr_sel[...,i], var_sel_min[i], var_sel_max[i])))
                    st.write(f'var{i+1}_sel_sc shape: {np.array(var_sel_sc[i]).shape}')

                Xr_sel_sc = np.stack(var_sel_sc, axis=3)
                st.write('var_sel_sc shape: ', Xr_sel_sc.shape)
                
                var_min = []
                var_max = []
                for i in range(nvar):
                    var_min.append(np.min(Xr[...,i]))
                    var_max.append(np.max(Xr[...,i]))
                    st.write(f'var{i+1}_min: {var_min[i]}, var{i+1}_max: {var_max[i]}')
                
                var_min = np.array(var_min)
                var_max = np.array(var_max)

                var_sc = []
                for i in range(nvar):
                    var_sc.append(scale_val(Xr[...,i], var_min[i], var_max[i]))
                    st.write(f'var{i+1}_sel_sc shape: ',var_sc[i].shape)
                
                var_sc = np.array(var_sc)
                Xr_sc = np.stack(var_sc, axis=3)
                st.write('var_sel_sc shape: ', Xr_sc.shape)

            else:
                None

            if decision2 == 'yes':
                fig, ax = plt.subplots()
                
                ax.hist(Xr.flatten(), bins='auto')  
                ax.set_title('Ground truth data distribution')
                ax.set_ylim(0,2e5)
                ax.set_xlim(-5,5)
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.set_title('Scaled downsampled data distribution')
                ax.hist(Xr_sc.flatten(), bins='auto')  
                st.pyplot(fig)

            # SVD

            if 'nz' in locals():
                XUds=np.zeros([nt,ny_sel,ny_sel,nvar])
                XSds=np.zeros([nt,ny_sel,ny_sel,nvar])
                XVds=np.zeros([nt,ny_sel,nx_sel*nz,nvar])
            else:
                XUds=np.zeros([nt,ny_sel,nx_sel,nvar])
                XSds=np.zeros([nt,nx_sel,nx_sel,nvar])
                XVds=np.zeros([nt,nx_sel,nx_sel,nvar])

            if 'Xr_sel_sc' in locals():
                for i in range(np.size(Xr_sel_sc,0)):
                    U_var_sel_sc = []
                    S_var_sel_sc = []
                    V_var_sel_sc = []
                    for j in range(nvar):

                        U, S, V = np.linalg.svd(Xr_sel_sc[i,...,j], full_matrices=False)
                        U_var_sel_sc.append(U)
                        S_var_sel_sc.append(np.diag(S))
                        V_var_sel_sc.append(V)
                
                    XUds_= np.stack(U_var_sel_sc, axis=-1) 
                    XSds_= np.stack(S_var_sel_sc, axis=-1)
                    XVds_= np.stack(V_var_sel_sc, axis=-1)

                    XUds[i,...]=XUds_
                    XSds[i,...]=XSds_
                    XVds[i,...]=XVds_
                st.write("")
                st.write('SVD Summary')
                st.write('XUds: ',XUds.shape)
                st.write('XSds: ',XSds.shape)
                st.write('XVds: ',XVds.shape)

            else:
                for i in range(np.size(Xr_sel,0)):
                    U_var_sel = []
                    S_var_sel = []
                    V_var_sel = []
                    for j in range(nvar):

                        U, S, V = np.linalg.svd(Xr_sel[i,...,j], full_matrices=False)
                        U_var_sel.append(U)
                        S_var_sel.append(np.diag(S))
                        V_var_sel.append(V)
                
                    XUds_= np.stack(U_var_sel, axis=-1) 
                    XSds_= np.stack(S_var_sel, axis=-1)
                    XVds_= np.stack(V_var_sel, axis=-1)

                    XUds[i,...]=XUds_
                    XSds[i,...]=XSds_
                    XVds[i,...]=XVds_
                st.write("")
                st.write('SVD Summary')
                st.write('XUds: ',XUds.shape)
                st.write('XSds: ',XSds.shape)
                st.write('XVds: ',XVds.shape)

            fig, ax = plt.subplots()
            ax.set_title('Left singular vectors data distribution')
            ax.hist(XUds.flatten(), bins='auto')  
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.set_title('Right singular vectors data distribution')
            ax.hist(XVds.flatten(), bins='auto') 
            st.pyplot(fig)

            ind=np.linspace(0,nt-1,nt,dtype=int)

            if decision4=='shuffle':
                np.random.shuffle(ind)
                ind

            train_ind = ind[0:n_train]
            test_ind = ind[n_train:]

            if 'Xr_sel_sc' in locals():
                Xr_train = Xr_sc[train_ind]
                Xr_test =  Xr_sc[test_ind]
            else:
                Xr_train = Xr[train_ind]
                Xr_test =  Xr[test_ind]

            XUds_train = XUds[train_ind]
            XUds_test = XUds[test_ind]
            XSds_train = XSds[train_ind]
            XSds_test = XSds[test_ind]
            XVds_train = XVds[train_ind]
            XVds_test = XVds[test_ind]

            np.product(Xr_train.shape)+np.product(Xr_test.shape)

            if not os.path.exists(f"{path0}/{filename}/weights"):
                os.makedirs(f"{path0}/{filename}/weights")

            file_name = f"{path0}/{filename}/weights/Interp_dense_NN_v1.0"

            save_best_weights = file_name + '_best.h5'
            save_last_weights = file_name + '_last.h5'
            save_summary_stats = file_name + '.csv'

            # Model inputs
            in_U_dim = XUds.shape[1:]
            in_S_dim = XSds.shape[1:]
            in_V_dim = XVds.shape[1:]
            out_dim = Xr_train.shape[1:]

            # Neural Network construction

            def create_model_1(in_U_dim, in_S_dim, in_V_dim, out_dim):
                in_U = Input(shape=(*in_U_dim,),name='in_u')  
                in_S = Input(shape=(*in_S_dim,),name='in_s') 
                in_V = Input(shape=(*in_V_dim,),name='in_v') 
                
                u = Flatten(name='u_1')(in_U) 
                u = Dense (neurons, activation=act_func, name='u_2')(u)
                XUus = Dense(out_dim[0]*in_U_dim[1]*in_U_dim[2],activation=act_fun2,name='u_3')(u)
                
                v = Flatten(name='v_1')(in_V)
                v = Dense (neurons, activation=act_func, name='v_2')(v)
                XVus = Dense(in_V_dim[0]*out_dim[1]*in_V_dim[2],activation=act_fun2,name='v_3')(v)
                
                XUus_reshape = Reshape((out_dim[0],in_U_dim[1],in_U_dim[2]),name='reshape_u')(XUus)
                XVus_reshape = Reshape((in_V_dim[0],out_dim[1],in_V_dim[2]),name='reshape_v')(XVus)
                
                X_hat = tf.einsum('ijkl,iknl,inpl->ijpl', XUus_reshape, in_S, XVus_reshape)

                m = Model(inputs=[in_U,in_S,in_V],outputs= X_hat)
                m_upsampled_matrices = Model(inputs=[in_U,in_S,in_V],outputs= [XUus_reshape,XVus_reshape]) 

                m.compile(loss=loss_function, optimizer=Adam(learn_rate), metrics=['mse'])
                
                return(m, m_upsampled_matrices)

            model, model_upsampled_matrices = create_model_1(in_U_dim, in_S_dim, in_V_dim, out_dim)

            t0 = time.time()
            callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', min_delta = 0.0001)]

            model.summary(print_fn=lambda x: st.text(x))

            history = model.fit([XUds_train,XSds_train,XVds_train],Xr_train,
                                batch_size=bs,
                                epochs=epoch,
                                validation_split=0.15,
                                verbose=1,
                                shuffle=True,
                                initial_epoch = 0,
                                callbacks=callbacks)

            st.success('The model has been trained!')

        # Training stats

        summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                    'train_acc': history.history['mse'],
                                    'valid_acc': history.history['val_mse'],
                                    'train_loss': history.history['loss'],
                                    'valid_loss': history.history['val_loss']})

        summary_stats.to_csv(save_summary_stats)

        fig, ax = plt.subplots()
        
        ax.set_yscale("log")
        ax.set_title('Training vs. Validation loss', fontsize = 14)
        ax.plot(summary_stats.train_loss, 'b', label = 'Train loss') 
        ax.plot(summary_stats.valid_loss, 'g', label = 'Valid. loss') 
        ax.legend(loc = 'upper right')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        
        ax.set_title('Training vs. Validation accuracy', fontsize = 14)
        ax.plot(summary_stats.train_acc, 'b', label = 'Train accuracy')
        ax.plot(summary_stats.valid_acc, 'g', label = 'Valid. accuracy')
        ax.legend(loc = 'upper right')
        st.pyplot(fig)

        min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
        st.write('###### Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))
        min_loss = round(min_loss, 4)

        # Load the best model epoch

        model.load_weights(save_best_weights)

        # Model prediction on training data
        Xr_hat = model.predict([XUds_test, XSds_test, XVds_test]) 

        if decision1=='scaling':
            Xr_test =  Xr[test_ind]

            for i in range(nvar):
                Xr_hat[...,i] = descale_val(Xr_hat[...,i], var_min[i], var_max[i])
        else:
            None

        if dims=='Data3D':
            Xr_test=np.reshape(Xr_test,(nt-n_train,ny,nx,nz,nvar),order='F')
            Xr_hat=np.reshape(Xr_hat,(nt-n_train,ny,nx,nz,nvar),order='F')
            Xr_sel=np.reshape(Xr_sel,(nt,ny_sel,nx_sel,nz,nvar),order='F')


        xg = np.linspace(0,nx,nx)
        yg = np.linspace(0,ny,ny)

        x = np.linspace(0,nx,nx_sel)
        y = np.linspace(0,ny,ny_sel)
        xx, yy = np.meshgrid(x, y)

        # Saving results
        if decision5 == 'yes':
            if not os.path.exists(f"{path0}/{filename}/tables"):
                os.makedirs(f"{path0}/{filename}/tables")
            if not os.path.exists(f"{path0}/{filename}/videos"):
                os.makedirs(f"{path0}/{filename}/videos")

        st.info(f'Plotting first 3 snapshots for all {nvar} variables')
        for nv in range(nvar):
            for nt in range(3):
                if 'nz' in locals():
                    fig, ax = plt.subplots()
                    n5 = int(Xr_test.shape[3] / 2)
                    fig.suptitle(f'XY plane - Component {nv+1} Snapshot {nt+1}', fontsize = 16)
                    ax.contourf(yg,xg,np.transpose(Xr_hat[nt,..., n5, nv], (1,0)))
                    ax.set_title('Predicted data - XY Plane', fontsize = 14)
                    ax.set_xlabel('X', fontsize = 12)
                    ax.set_ylabel('Y', fontsize = 12)
                    fig.tight_layout()
                    st.pyplot(fig)

                else:
                    fig, ax = plt.subplots()
                    fig.suptitle(f'Component {nv+1} Snapshot {nt+1}', fontsize = 16)
                    ax.contourf(yg,xg,np.transpose(Xr_hat[nt,...,nv], (1,0)))
                    ax.set_title('Predicted data', fontsize = 14)
                    ax.set_xlabel('X', fontsize = 12)
                    ax.set_ylabel('Y', fontsize = 12)
                    fig.tight_layout()
                    st.pyplot(fig)

        if decision5 == 'yes':
            tabname1 = "tables/NN_table1"
            tabname2 = "tables/NN_table2"

            if 'nz' in locals():
                Xr_test_p0=Xr_test[...,0,:]
                Xr_hat_p0=Xr_hat[...,0,:]
                Xr_sel_p0=Xr_sel[...,0,:]

                [results_table, results_table2] = error_tables(Xr_test_p0, Xr_hat_p0, Xr_sel_p0,tabname1,tabname2, path0, filename, nvar)
            else:
                [results_table, results_table2] = error_tables(Xr_test, Xr_hat, Xr_sel, tabname1, tabname2, path0, filename, nvar)

        if decision5 == 'yes':
            st.write('Performance measures on Test data, for all measures')
            st.table(results_table)
            

            st.write('Performance measures on Test data, for one specific layer of measures')  
            st.table(results_table2)
            
        # Video creation
        if decisionx == 'yes':
            with st.spinner(f'Generating videos for all {nvar} variables. Video will be saved to {path0}/{filename}/videos. This may take some time'):
                for var in range(nvar):
                    if 'nz' in locals ():
                        n5 = int(Xr_test.shape[3] / 2)
                        videoname=f"{path0}/{filename}/videos/var{var}_p{n5}.gif"
                        Xr_test_sel=Xr_test[...,n5,:]
                        Xr_hat_sel=Xr_hat[...,n5,:]
                        Xr_sel_sel=Xr_sel[...,n5,:]
                        videos_results(videoname, var, n_train, test_ind, xg, yg, x, y, Xr_test_sel, Xr_hat_sel, Xr_sel_sel)
                    else:
                        videoname=f"{path0}/{filename}/videos/var{var}.gif"
                        videos_results(videoname, var, n_train, test_ind, xg, yg, x, y, Xr_test, Xr_hat, Xr_sel)
                st.success('All snapshots have been saved!')

        if decision7=='yes':
            st.info('Plotting comparison of enhanced data, original data, and ground truth for first 3 snapshots')
            for var in range(nvar):
                os.makedirs(f"{path0}/{filename}/compare_plots", exist_ok=True)
                os.makedirs(f"{path0}/{filename}/compare_plots/var{var}", exist_ok=True)
                figname = f"{path0}/{filename}/compare_plots/var{var}/var{var}_scatter.pdf"
                files3 = os.listdir(f'{path0}/{filename}/compare_plots')
                
                os.makedirs(f"{path0}/{filename}/compare_plots/var{var}",exist_ok=True)
                
                for n_snap in range(0, 3):
                    if 'nz' in locals():
                        n5 = int(Xr_test.shape[3] / 2)
                        figname = f"{path0}/{filename}/compare_plots/var{var}/var{var}_snap{n_train+n_snap}_p{n5}.pdf"
                        
                        Xr_test_sel=Xr_test[...,n5,:]
                        Xr_hat_sel=Xr_hat[...,n5,:]
                        Xr_sel_sel=Xr_sel[...,n5,:]
                        figures_results_ylim(n_snap, var, yg, xg, y, x, Xr_test_sel, Xr_sel_sel, Xr_hat_sel, figname)
                        
                    else:
                        figname = f"{path0}/{filename}/compare_plots/var{var}/var{var}_{n_train+n_snap}.pdf"
                        figures_results_ylim(n_snap, var, yg, xg, y, x, Xr_test, Xr_sel, Xr_hat, figname)

        np.save(f'{path0}/{filename}/EnhancedData.npy', Xr_hat)
                        
        st.info("Press 'Refresh' to run a new case")
        Refresh = st.button('Refresh')
        if Refresh:
            st.stop()