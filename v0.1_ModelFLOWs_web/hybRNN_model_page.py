import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import hdf5storage


from numpy import linalg as LA

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Reshape, TimeDistributed, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM



import streamlit as st

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

def menu():
    st.title("Hybrid RNN Model")
    st.write("""
This algorithm proposes a hybrid predictive Reduced Order Model to solve reacting flow problems. 
This algorithm is based on a dimensionality reduction using Proper Orthogonal Decomposition (POD) combined with deep learning architectures to predict the temporal coefficients of the POD modes. 
The deep learning architecture implemented is: recursive neural network (RNN).
""")
    st.write(" ## RNN Model - Parameter Configuration")

    def mean_absolute_percentage_error(y_true, y_pred): 
        epsilon = 1e-10 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

    def smape(A, F):
        return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

    def RRMSE(Tensor0, Reconst):
        RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
        return(RRMSE)

    def custom_loss(y_actual,y_pred):
        if decision2 == 'no-scaling':
            Ten = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_pred)
            Ten_actual = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_actual)
        elif decision2 == 'MaxPerMode':
            Ten = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_pred* (max_val))
            Ten_actual = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_actual* (max_val))
        elif decision2 == 'auto':
            Ten = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_pred* (std_val)+med_val)
            Ten_actual = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_actual* (std_val)+med_val)
        elif decision2 == 'range':
            Ten = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_pred* (range_val)+min_val)
            Ten_actual = tf.einsum('ij,kj->ki',tf.convert_to_tensor(U, dtype=tf.float32),y_actual* (range_val)+min_val)    

        Ten = tf.keras.layers.Reshape((nx*ny,nv))(Ten)
        Ten_actual = tf.keras.layers.Reshape((nx*ny,nv))(Ten_actual)
        
        output_list = []
        output_list2 = []
        
        for iter in range(Ten.shape[2]-1):
            variable = Ten[:,:,iter+1]
            variable_ac = Ten_actual[:,:,iter+1]
            
            if decision1 == 'no-scaling':
                output_list.append(variable)
                output_list2.append(variable_ac)
            else:
                Med = tf.cast(tf.reshape(Media_tiempo[iter+1,:,:,0],[nx*ny]),tf.float32)
                output_list.append(variable*(Factor[iter+1])+Media[iter+1]+Med)
                output_list2.append(variable_ac*(Factor[iter+1])+Media[iter+1]+Med)
            
        Ten2 = tf.stack(output_list)
        Ten2_ac = tf.stack(output_list2)
        
        pred_sum_spec = tf.math.reduce_sum(Ten2,axis=0)
        pred_sum_spec = Reshape((nx,ny))(pred_sum_spec)
        
        ac_sum_spec = tf.math.reduce_sum(Ten2_ac,axis=0)
        ac_sum_spec = Reshape((nx,ny))(ac_sum_spec)    
        
        custom_loss = tf.reduce_mean(tf.square(y_actual-y_pred)) + tf.reduce_mean(tf.square(ac_sum_spec - pred_sum_spec))
        
        return custom_loss

    path0 = os.getcwd()

    if not os.path.exists(f'{path0}/RNN_solution'):
        os.mkdir(f'{path0}/RNN_solution')

    n_modes = 18

    selection = 'Tensor_cylinder_Re100.mat'

    decision1 = st.selectbox("Select the scaling of the variables implemented", ("auto", "pareto", "range", "no-scaling"))
    decision2 = st.selectbox("Select the scaling of the temporal coefficients implemented", ("MaxPerMode", "auto", "range", "no-scaling"))

    n_modes = st.number_input('Select the number of modes to retain during SVD', min_value = 0, max_value = None, value = 15, step = 1)

    # Inputs
    test_prop = 0.20 # test set proportion
    val_prop = 0.25 # validation set proportion val_length = (1-test_prop)*val_prop // train_length = (1-test_prop)*(1-val_prop)
    batch_size = st.slider('Select Batch Size', 4, 64, value = 8, step = 2)
    k = 10  # number of snapshots used as predictors
    p = 6   # number of snapshots used as time-ahead predictions
    epoch = st.slider('Select training epochs', 0, 500, value = 200, step=10)

    act_func1 = st.selectbox('Select hidden layer activation function', ('relu', 'elu', 'sigmoid', 'softmax', 'tanh'))
    act_func2 = st.selectbox('Select output layer activation function', ('tanh', 'relu', 'sigmoid'))
    neurons = st.slider('Select number of neurons per layer', 1, 150, value = 30, step = 1)
    shared_dim = st.slider('Select number of shared dims', 1, 100, value = 20, step = 1)
    lr = st.number_input('Select learning rate', min_value = 0.0001, max_value = 0.01, value = 0.002, step = 0.0001, format = '%.4f') #learning_rate
    lf = 'mse'

    go = st.button("Calculate")

    if go:
        with st.spinner("Please wait while the model is being trained"):
            if not os.path.exists(f'{path0}/hyb_RNN_solution/'):
                os.mkdir(f'{path0}/hyb_RNN_solution/') 
            output1 = 'yes' # Error made by SVD in each variable
            output2 = 'yes' # Plot Comparison Coefficients in all indexes
            output3 = 'yes' # Figure loss function
            output4 = 'yes' # RRMSE of each variable (Original Tensor)
            output5 = 'yes' # RRMSE of each variable (Truncated Tensor)
            output6 = 'yes' # Plot RRMSE temporal matrix
            output7 = 'yes' # Comparison of the modes
            output8 = 'yes' # Comparison of the snapshots

            if output7 == 'yes': # Indicate the number of the mode you want to compare
                nModes = [1,2,3,4,5]
            if output8 == 'yes': # Indicate the number of the variable and time step (nv, nt)
                index = np.array([
                    [0,1],
                    [1,1]])
                   
            Tensor_ = hdf5storage.loadmat(f'{path0}/{selection}')
            data = list(Tensor_.values())[-1]
            data = tf.transpose(data, (3, 1, 2, 0))

            nt, ny, nx, nv = np.shape(data)

            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            xv, yv = np.meshgrid(y, x)

            # Original Tensor Reconstruction
            Mat_orig = np.reshape(data,[nt,nv*nx*ny])
            Mat_orig = np.transpose(Mat_orig)
            Tensor_orig = np.reshape(Mat_orig,[nv,nx,ny,nt], order='F')
            sum_spec = np.sum(np.sum(Tensor_orig[1:,:,:,:],axis=3),axis=0)/(nt)

            # Centering and Scaling 
            if decision1 == 'no-scaling':
                pass
            else:
                #Dummy variables
                Tensor_orig1 = np.zeros(Tensor_orig.shape)
                Factor = np.zeros(data.shape[3])
                Media = np.zeros(data.shape[3])
                Media_tiempo = np.zeros(np.shape(Tensor_orig))

                for iter in range(nt):
                    Media_tiempo[:,:,:,iter] = np.mean(Tensor_orig,axis=3)

                Tensor_orig2 = Tensor_orig-Media_tiempo

                for iter in range(data.shape[3]):
                    variable = Tensor_orig2[iter,:,:,:]
                    if decision1 == 'range':
                        Factor[iter] = np.amax(variable)-np.amin(variable) #Range scaling
                    if decision1 == 'auto':
                        Factor[iter] = np.std(variable) #Auto scaling
                    if decision1 == 'pareto':
                        Factor[iter] = np.sqrt(np.std(variable)) #Pareto scaling
                    Media[iter] = np.mean(variable)
                    Tensor_orig1[iter,:,:,:] = (variable-Media[iter])/(Factor[iter])
            
                Mat_orig = np.reshape(Tensor_orig1,[nv*nx*ny,nt],order='F')

            # Perform SVD
            U, S, V = np.linalg.svd(Mat_orig, full_matrices=False)

            Modes = n_modes
            S = np.diag(S)

            U = U[:,0:Modes]
            S = S[0:Modes,0:Modes]
            V = V[0:Modes,:]

            AML = np.dot(S,V) 
            tensor = AML

            if output1 == 'yes':
                Mat_trunc = np.dot(U,AML)
                Ten_trunc = np.reshape(Mat_trunc,[nv,nx,ny,nt], order='F')
                
                if decision1 == 'no-scaling':
                    pass
                else:
                    for iter in range(data.shape[3]):
                        variable = Ten_trunc[iter,:,:,:]
                        Ten_trunc[iter,:,:,:] = variable*(Factor[iter])+Media[iter]+Media_tiempo[iter,:,:,:]
                        
                if output1 == 'yes': 
                    st.write('\n') 
                    RRMSE_SVD = np.zeros(data.shape[3])
                    for iter in range(data.shape[3]):
                        diff = Ten_trunc[iter,:,:,:] - Tensor_orig[iter,:,:,:]
                        RRMSE_SVD[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Tensor_orig[iter,:,:,:].flatten(),(-1,1)),ord=2)
                        st.write(f'###### SVD RRMSE error for variable {iter+1}: {np.round(RRMSE_SVD[iter]*100, 3)}%\n')
                    st.write('\n')
                    np.save(f'{path0}/hyb_RNN_solution/RRMSE_SVD.npy',RRMSE_SVD)

            if decision2 == 'no-scaling':
                tensor_norm = tensor
            elif decision2 == 'range':
                min_val = np.amin(tensor)
                range_val = np.ptp(tensor)
                tensor_norm = (tensor-min_val)/range_val
            elif decision2 == 'auto':
                med_val = np.mean(tensor)
                std_val =np.std(tensor)
                tensor_norm = (tensor-med_val)/std_val
            elif decision2 == 'MaxPerMode':
                max_val = sum(np.amax(np.abs(tensor),axis=1))
                tensor_norm = tensor / max_val

            # Dataset configuration
            total_length = tensor_norm.shape[1] # number of snapshots
            channels_n = 0                      # number of channels, assuming each snapshot is an image with n channels
            dim_x = tensor_norm.shape[0]        # following the simil that each snapshot is an image, the dimension x of that image
            dim_y = 0                           # following the simil that each snapshot is an image, the dimension y of that image

            print('total_length: ', total_length)
            print('channels_n: ', channels_n)
            print('dim_x: ', dim_x)
            print('dim_y: ', dim_y)

            # Data generator
            import math
            class DataGenerator(tf.keras.utils.Sequence): 
                'Generates data for Keras'
                def __init__(self, data, list_IDs, batch_size=5, dim=(20), 
                            k = 624, p = 1, 
                            shuffle=True, till_end = False, only_test = False):
                    'Initialization'
                    self.data = data
                    self.dim = dim
                    self.batch_size = batch_size
                    self.list_IDs = list_IDs
                    self.shuffle = shuffle
                    self.p = p
                    self.k = k
                    self.till_end = till_end
                    self.only_test = only_test
                    self.on_epoch_end()

                def __len__(self):
                    'Denotes the number of batches per epoch'
                    if self.till_end:
                        lenx = math.ceil((len(self.list_IDs) / self.batch_size))
                    else:
                        lenx = int(np.floor(len(self.list_IDs) / self.batch_size))
                    return lenx

                def __getitem__(self, index):
                    'Generate one batch of data'
                    # Generate indexes of the batch
                    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

                    # Find list of IDs
                    list_IDs_temp = [self.list_IDs[k] for k in indexes]

                    # Generate data
                    X, y = self.__data_generation(list_IDs_temp)
                    if self.only_test:
                        return X
                    else:
                        return X, y

                def on_epoch_end(self):
                    'Updates indexes after each epoch'
                    self.indexes = np.arange(len(self.list_IDs))
                    if self.shuffle == True:
                        np.random.shuffle(self.indexes)

                def __data_generation(self, list_IDs_temp):
                    'Generates data containing batch_size samples' # X : (n_samples, *dim, depth)
                    # Initialization
                    X = np.empty((self.batch_size, self.dim, self.k))
                    y = [np.empty((self.batch_size, self.dim))]*self.p

                    y_inter = np.empty((self.batch_size, self.dim, p))
                    
                    # Generate data
                    lenn = len(list_IDs_temp)
                    for i, ID in enumerate(list_IDs_temp):
                        # Store Xtrain
                        X[i,:,:] = self.data[:,ID:ID+k]
                        # Store Ytrain
                        y_inter[i,:,:] = self.data[:,ID+k:ID+k+p] 

                    for j in range(self.p):
                        y[j] = y_inter[:,:,j]
                        y[j] = np.reshape(y[j], (lenn, -1)) 
                    
                    X = X.transpose((0,2,1))

                    return X, y
    
            
            # Prepare the dataset indexes
            period_transitorio = 0
            stride_train = 1
            stride_val = 1
            stride_test = 1

            dim=(dim_x)

            test_length = int(test_prop * total_length)
            val_length  = int((total_length - test_length) * val_prop)
            train_length = total_length - val_length - test_length
                
            if int(train_length-period_transitorio-(k+p)) < 0:
                train_n = 0
            elif int((train_length-period_transitorio-(k+p))//stride_train) == 0:
                train_n = 1
            else: 
                train_n = int(((train_length-period_transitorio)-(k+p))//stride_train)
                
            if int(test_length-(k+p)) < 0:
                test_n = 0
            elif int((test_length-(k+p))//stride_test) == 0:
                test_n = 1
            else: 
                test_n = int((test_length-(k+p))//stride_test)

            if int(val_length-(k+p)) < 0:
                val_n = 0
            elif int((val_length-(k+p))//stride_val) == 0:
                val_n = 1
            else: 
                val_n = int((val_length-(k+p))//stride_val)

            # Indices for the beginnings of each batch
            train_idxs = np.empty([train_n], dtype='int')
            val_idxs = np.empty([val_n], dtype='int')
            test_idxs = np.empty([test_n], dtype='int')

            j = period_transitorio
            for i in range(train_n):
                train_idxs[i] = j
                j = j+stride_train

            j = train_length
            for i in range(val_n):
                val_idxs[i] = j
                j = j+stride_val

            j = train_length + val_length
            for i in range(test_n):
                test_idxs[i] = j
                j = j+stride_test



            # Generators
            training_generator = DataGenerator(tensor_norm, train_idxs,  
                                                dim = dim, 
                                                batch_size = batch_size,
                                                k = k, p = p, till_end = False,
                                                only_test = False,
                                                shuffle = True)
            validation_generator = DataGenerator(tensor_norm, val_idxs, 
                                                dim = dim, 
                                                batch_size = batch_size,
                                                k = k, p = p, till_end = False,
                                                only_test = False,
                                                shuffle = False)
            test_generator = DataGenerator(tensor_norm, test_idxs, 
                                                dim = dim, 
                                                batch_size = batch_size,
                                                k = k, p = p, till_end = False,
                                                only_test = True,
                                                shuffle = False)
            if output2 == 'yes':
                all_idxs = np.int_(np.linspace(0,nt-k-p,nt-k-p+1))
                all_generator = DataGenerator(tensor_norm, all_idxs, 
                                                dim = dim, 
                                                batch_size = batch_size,
                                                k = k, p = p, till_end = False,
                                                only_test = True,
                                                shuffle = False)

            print ('test_length: ', test_length)
            print ('val_length: ', val_length)
            print ('train_length: ', train_length)
            print()
            print ('test_n: ', test_n)
            print ('val_n: ', val_n)
            print ('train_n: ', train_n)
            print()
            print('test_generator_len: ', len(test_generator))
            print('validation_generator_len: ', len(validation_generator))
            print('training_generator_len: ', len(training_generator))

        # Prepare Ytest
            test_n_adjusted = int(test_n/batch_size)*batch_size  # multiplo de batch_size
            Ytest = [np.empty([test_n_adjusted, dim_x], dtype='float64')] * p
            Ytest_fl = [np.empty([test_n_adjusted, dim_x ], dtype='float64')] * p

            Ytest_inter = np.empty([test_n_adjusted, dim_x, p], dtype='float64')

            for i in range(test_n_adjusted):
                j = test_idxs[i]
                Ytest_inter[i,:,:] = tensor_norm[:,j+k:j+k+p]

            for r in range(p):    
                Ytest[r] = Ytest_inter[:,:,r]
                Ytest_fl[r] = np.copy(np.reshape(Ytest[r], (test_n_adjusted, -1)) )


            # LSTM MODEL 100 NODES
            # Build Model
            def create_model(in_shape, out_dim, p = 3, shared_dim = 1000, act_fun= act_func1):
                x = Input(shape=in_shape)
                
                
                v = LSTM(neurons)(x)
            
                v = Dense(p*neurons, activation= act_fun)(v)
                
                v = Reshape((p,neurons))(v) 
                tt = [1]*p
                
                r = TimeDistributed( Dense(shared_dim, activation=act_fun))(v)
                s = tf.split(r, tt, 1)
                for i in range(p):
                    s[i] = Flatten()(s[i])

                o = []
                for i in range(p):
                    o.append( Dense(out_dim, activation=act_func2)(s[i]) )

                m = Model(inputs=x, outputs=o)
                opt = keras.optimizers.Adam(learning_rate=lr)
                m.compile(loss=lf, optimizer=opt, metrics=[lf])
                return(m)

            #create the model

            in_shape = [k, dim_x]
            out_dim = dim_x 

            print(in_shape)
            print(out_dim)

            model= create_model(in_shape,out_dim,p,shared_dim) 


            # save the best weights 
            save_string = 'colab_Cilin3D_AML_20_LSTM_100 v1'

            # save the best weights 
            save_best_weights = f'{path0}/hyb_RNN_solution/' + save_string + '.h5'
            save_summary_stats = f'{path0}/hyb_RNN_solution/' + save_string + '.csv'
            save_last_weights = f'{path0}/hyb_RNN_solution/' + save_string + '_last_w.h5'
            save_results_metrics = f'{path0}/hyb_RNN_solution/' + save_string + '_results_metrics.csv'
                

            # Training
            np.random.seed(247531338)
            t0 = time.time()
            # Model training
            callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_delta = 0.001)]
                        #,ReduceLROnPlateau(monitor='val_loss', factor=0.8,patience=5,min_lr=1e-6)]

            model.summary(print_fn=lambda x: st.text(x))

            history = model.fit(training_generator,
                    validation_data=validation_generator,
                    epochs=epoch,
                    verbose=1,
                    callbacks=callbacks)
            
            t1 = time.time()
            print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

            model.save_weights(save_last_weights)
            st.success('The model has been trained!')
        
        st.write("### Model Training Results - RNN")

        # Aggregate the summary statistics
        summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                            #'train_acc': history.history['mean_squared_error'],
                            #'valid_acc': history.history['val_mean_squared_error'],
                            'train_loss': history.history['loss'],
                            'valid_loss': history.history['val_loss']})
        
        summary_stats.to_csv(save_summary_stats)  

        if not os.path.exists(f'{path0}/hyb_RNN_solution/plots'):
            os.mkdir(f'{path0}/hyb_RNN_solution/plots')  


        if output3 == 'yes':
            fig, ax = plt.subplots()

            ax.plot(summary_stats.train_loss, 'b',linewidth = 3) # blue
            ax.plot(summary_stats.valid_loss, 'g--',linewidth  = 3) # green
            ax.set_xlabel('Training epochs',fontsize = 12)
            ax.set_ylabel('Loss function',fontsize = 12)

            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            plt.savefig(f'{path0}/hyb_RNN_solution/loss.png')
            st.pyplot(fig)
            plt.close()

        # Find the min validation loss during the training
        min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
        print('Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.6f}'.format(min_loss))
        min_loss = round(min_loss, 4)

        # Inference
        t0 = time.time()

        model.load_weights(save_best_weights)
        Ytest_hat_fl = model.predict(test_generator, verbose=1) 
            
        t1 = time.time()
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        print('Performance measures on Test data, per sec')
        lag = 0
        num_sec = Ytest_hat_fl[0].shape[0]
        print(num_sec)
        results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
        for i in range(num_sec):
            results_table.iloc[0,i] = mean_squared_error( Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
            results_table.iloc[1,i] = mean_absolute_error(   Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
            results_table.iloc[2,i] = median_absolute_error(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
            results_table.iloc[3,i] = r2_score(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
            results_table.iloc[4,i] = smape(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
            results_table.iloc[5,i] = RRMSE( np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))
        results_table

        print('Performance measures on Test data, for all time, per time-ahead lag')

        results_table_global = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(p))
        for i in range(p):
            results_table_global.iloc[0,i] = mean_squared_error(Ytest_fl[i].flatten(), Ytest_hat_fl[i].flatten())
            results_table_global.iloc[1,i] = mean_absolute_error(Ytest_fl[i].flatten(), Ytest_hat_fl[i].flatten())
            results_table_global.iloc[2,i] = median_absolute_error(Ytest_fl[i].flatten(), Ytest_hat_fl[i].flatten())
            results_table_global.iloc[3,i] = r2_score(Ytest_fl[i].flatten(), Ytest_hat_fl[i].flatten())
            results_table_global.iloc[4,i] = smape(Ytest_fl[i].flatten(), Ytest_hat_fl[i].flatten())
            results_table_global.iloc[5,i] = RRMSE( np.reshape(Ytest_fl[i].flatten(),(-1,1)), np.reshape(Ytest_hat_fl[i].flatten(),(-1,1)))

        results_table_global['mean'] = results_table_global.mean(axis=1)
        results_table_global

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[0,:], 'b') # green
        ax.set_title("MSE score vs. forecast time index",fontsize = 14)
        ax.set_xlabel("time", fontsize = 12)
        ax.set_ylabel("MSE",fontsize = 12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/MSEscore.png')
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[1,:], 'b') # green
        ax.set_title("MAE score vs. forecast time index",fontsize = 14)
        ax.set_xlabel("time", fontsize = 12)
        ax.set_ylabel("MAE",fontsize = 12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/MAEscore.png')
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[2,:], 'b') # green
        ax.set_title("MAD score vs. forecast time index",fontsize = 14)
        ax.set_xlabel("time", fontsize = 12)
        ax.set_ylabel("MAD",fontsize = 12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/MADscore.png')
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[3,:], 'b') # green
        ax.set_title("R2 score vs. forecast time index",fontsize=14)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("R2",fontsize=12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/R2score.png')
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[4,:], 'b') # green
        ax.set_title("SMAPE score vs. forecast time index",fontsize=14)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("SMAPE",fontsize=12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/SMAPEscore.png')
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(range(num_sec),results_table.iloc[5,:], 'b') # green
        ax.set_title("RRMSE score vs. forecast time index",fontsize=14)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("RRMSE",fontsize=12)
        plt.savefig(f'{path0}/hyb_RNN_solution/plots/RRMSEscore.png')
        st.pyplot(fig)
        plt.close()

        Ytest_hat_fl = model.predict(test_generator, verbose=1) 

        # Create the multidimensional arrays for the results and ground-truth values
        mat_pred = np.zeros((Ytest_fl[0].shape[0],p,Ytest_fl[0].shape[1]))
        print(mat_pred.shape)


        # Fill the output arrays
        for i in range(p):
            for j in range(Ytest_fl[0].shape[0]):
                mat_pred[j,i,:]=Ytest_hat_fl[i][j,:] 

        if decision2 == 'no-scaling':
            pass
        elif decision2 == 'auto':
            mat_pred = mat_pred * std_val + med_val
        elif decision2 == 'range':
            mat_pred = mat_pred * range_val + min_val
        elif decision2 == 'MaxPerMode':
            mat_pred = mat_pred * max_val
                
        if output2 == 'yes':
            Ytest_all = model.predict(all_generator, verbose=1) 
            mat_all = np.zeros((Ytest_all[0].shape[0],p,Ytest_all[0].shape[1]))
            
            for i in range(p):
                for j in range(Ytest_all[0].shape[0]):
                    mat_all[j,i,:]=Ytest_all[i][j,:]
            
            if decision2 == 'no-scaling':
                pass
            elif decision2 == 'auto':
                mat_all = mat_all * std_val + med_val
            elif decision2 == 'range':
                mat_all = mat_all * range_val + min_val
            elif decision2 == 'MaxPerMode':
                mat_all = mat_all * max_val

        # Data Extraction and Tensor Reconstruction
        new_dim = [mat_pred.shape[0],mat_pred.shape[2]]

        AML_pre_LSTM = np.transpose(np.squeeze(mat_pred[:,0,:]))
        num_snap = AML_pre_LSTM.shape[1]

        mat_time_slice_index = np.zeros((Ytest_fl[0].shape[0],p),dtype=int)
        for i in range(p):
            for j in range(Ytest_fl[0].shape[0]):
                mat_time_slice_index[j,i]=test_idxs[j]+k+i     

        time_lag = mat_time_slice_index[0,0]

        # Matrix reconstruction

        Mat_pre = np.dot(U,AML_pre_LSTM)

        # Tensor reconstruction
        Ten_pre = np.reshape(Mat_pre, [nv,nx,ny,AML_pre_LSTM.shape[1]], order='F')

        for iter in range(data.shape[3]):
            variable = Ten_pre[iter,:,:,:]
            Ten_pre[iter,:,:,:] = variable*(Factor[iter])+Media[iter]+Media_tiempo[iter,:,:,:Ten_pre.shape[3]]
                
        if output2 == 'yes':
            AML_all_LSTM = np.transpose(np.squeeze(mat_all[:,0,:]))

            
        # RRMSE Measure with Original Tensor
        if output4 == 'yes':
            RRMSE_orNN = np.zeros(Tensor_orig.shape[0])
            for iter in range(Tensor_orig1.shape[0]):

                diff = Ten_pre[iter,:,:,:] - Tensor_orig[iter,:,:,time_lag:time_lag+num_snap]
                RRMSE_orNN[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Tensor_orig[iter,:,:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)
                st.write(f'###### RRMSE measure with original tensor - variable {iter+1}: {np.round(RRMSE_orNN[iter]*100, 3)}%\n')
            st.write('\n')
            np.save(f"{path0}/hyb_RNN_solution/RRMSE_orNN.npy", RRMSE_orNN)
        st.write('\n')
        # RRMSE with Truncated
        if output5 == 'yes':
            RRMSE_trNN = np.zeros(Tensor_orig.shape[0])
            for iter in range(Tensor_orig1.shape[0]):

                diff = Ten_pre[iter,:,:,:] - Ten_trunc[iter,:,:,time_lag:time_lag+num_snap]
                RRMSE_trNN[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Ten_trunc[iter,:,:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)

                st.write(f'###### RRMSE measure with truncated tensor - variable {iter+1}: {np.round(RRMSE_trNN[iter]*100, 3)}%\n')
        
            st.write('\n')
            np.save(f"{path0}/hyb_RNN_solution/RRMSE_trNN.npy", RRMSE_trNN) 
        st.write('\n')
        # RRMSE Measure
        if output6 == 'yes':
            diff = AML_pre_LSTM - AML[:,time_lag:time_lag+num_snap]
            globalRRMSE = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(AML[:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)
            st.write(f'###### Global RRMSE measure: {np.round(globalRRMSE*100, 3)}%\n')
            st.write('\n')


            rrmse_arr = np.zeros(num_snap)
            for i in range(num_snap):
                diff = AML[:,time_lag+i]-AML_pre_LSTM[:,i]
                rrmse_arr[i] = LA.norm(diff.flatten(),2)/LA.norm(np.transpose(AML[:,i]).flatten(),2)

            fig, ax = plt.subplots()
            ax.plot(rrmse_arr,'k*-')
            ax.set_xlabel('Time')
            ax.set_ylabel('RRMSE')
            plt.savefig(f'{path0}/hyb_RNN_solution/rrmse.png')
            st.pyplot(fig)
            plt.close()

        # AML Comparison
        if output7 == 'yes':
            # AML Comparation
            if not os.path.exists(f'{path0}/hyb_RNN_solution/Mode_Comparison'):
                os.mkdir(f"{path0}/hyb_RNN_solution/Mode_Comparison")

            AML_pre_1 = np.zeros([AML.shape[0],time_lag])
            AML_pre_1 = np.concatenate((AML_pre_1,AML_pre_LSTM),axis=1)

            for i in nModes:
                fig, ax = plt.subplots()
                ax.plot(AML[i-1,time_lag:],'k*-')
                ax.plot(AML_pre_1[i-1,time_lag:],'m*-')
                ax.set_xlabel("Time", fontsize = 12)
                ax.set_title(f"Comparation between AML and AMLPRE at mode {i}", fontsize = 14)
                plt.savefig(f'{path0}/hyb_RNN_solution/Mode_Comparison/aMLComparation_m_{i}_test.png')
                st.pyplot(fig)  
                plt.close()              
                if output2 == 'yes':
                    fig, ax = plt.subplots()
                    ax.plot(AML[i-1,:nt-k-p+1],'k*-', label='Original')
                    ax.plot(k+all_idxs[:train_idxs[-1]],AML_all_LSTM[i-1,:train_idxs[-1]],'b*-', label='Training')
                    ax.plot(k+all_idxs[1+train_idxs[-1]:val_idxs[-1]],AML_all_LSTM[i-1,1+train_idxs[-1]:val_idxs[-1]],'r*-', label='Validation')    
                    ax.plot(k+all_idxs[1+val_idxs[-1]:all_idxs[-1]],AML_all_LSTM[i-1,1+val_idxs[-1]:all_idxs[-1]],'g*-', label='Test')  
                    ax.set_xlabel("Time", fontsize = 12)
                    ax.legend()
                    ax.set_title(f"Mode {i}", fontsize = 14)
                    fig.tight_layout()
                    namefig1 = f'{path0}/hyb_RNN_solution/Mode_Comparison/aMLComparation_m_{i}_all.png'
                    plt.savefig(namefig1)
                    st.pyplot(fig)
                    plt.close()


        # Tensor Comparison
        if output8 == 'yes':
            if not os.path.exists(f'{path0}/hyb_RNN_solution/Snapshots'):
                os.mkdir(f"{path0}/hyb_RNN_solution/Snapshots")

            for i in range(index.shape[0]):
                namefig_orig1 = f'{path0}/hyb_RNN_solution/Snapshots/snap_comp{index[i,0]+1}_t{index[i,1]+1}.png'
                fig, ax = plt.subplots()

                ax.contourf(xv,yv,Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])],100,cmap='jet',
                            vmin = np.amin(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]),
                            vmax = np.amax(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]))
                    
                ax.contourf(-xv,yv,Ten_pre[int(index[i,0]),:,:,int(index[i,1])],100,cmap='jet',
                            vmin = np.amin(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]),
                            vmax = np.amax(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]))

                ax.set_title(f'Prediction vs.Original Data - Comp. {int(index[i,0])+1} Snapshot {int(index[i,1])+1}', fontsize = 14)

                ax = plt.gca()
                ax.set_aspect(1)
                props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax.annotate('', xy=(0.5, -0.005), xycoords='axes fraction', xytext=(0.5, 1.005),
                            arrowprops=dict(arrowstyle='-', lw = 3, color='k'))
                ax.axis('off')
                plt.savefig(namefig_orig1)
                st.pyplot(fig)
                plt.close()

        st.info("Press 'Refresh' to run a new case")
        Refresh = st.button('Refresh')
        if Refresh:
            st.stop()
