import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import scipy.io
import sys, os
import data_fetch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, r2_score

# Remove logs from tensorflow about CUDA
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, LSTM, Flatten, Conv3D
from tensorflow.keras.layers import MaxPooling3D, Permute, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.models import Model

################################################################################
# Load Data and Preprocess
################################################################################

def load_preprocess(tensor_train: np.ndarray, tensor_test: np.ndarray, 
                    train_size: float = 0.75, val_size: float = 0.25, 
                    batch_size: int = 8, model_type = 'rnn'):
    
    # Use to raise errors
    flag = {'check': True,
     'text': None}
    outputs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    outputs[0] = flag

    if (train_size + val_size != 1.0):

      flag = {'check': False, 
              'text': "Error: Params 'train_size', 'val_size' " + 
              "must add up to 1, e.g., train_size = 0.8, " + 
              "val_size = 0.2"}
      outputs[0] = flag
      
      return outputs

    if (tensor_train.ndim != 4 or tensor_test.ndim != 4):

      flag = {'check': False, 
              'text': "Error: Params 'tensor_train' and 'tensor_test' has to have a " + 
              "number of dimensions equal to 4, and the temporal dimension has to " + 
              "be the last one"}
      outputs[0] = flag
      
      return outputs

    if (model_type == 'cnn'):

      min_val = np.amin(tensor_train)
      range_val = np.ptp(tensor_train)

    elif (model_type == 'rnn'):
      
      min_val = 0
      range_val = 1

    else:

      flag = {'check': False, 
              'text': "Error: Param 'model_type' has to be an string equal to " + 
              "'cnn' or 'rnn'"}
      outputs[0] = flag
      
      return outputs


    tensor_train_norm = (tensor_train - min_val)/range_val
    tensor_test_norm = (tensor_test - min_val)/range_val
    
    # Dataset configuration
    total_length = tensor_train_norm.shape[3]
    channels_n = tensor_train_norm.shape[0]
    dim_x = tensor_train_norm.shape[1]
    dim_y = tensor_train_norm.shape[2]

    """
        Data Generator (Rolling Window)
    """
    if (model_type == 'rnn'):
      class DataGenerator(tf.keras.utils.Sequence): 
        'Generates data for Keras'
        def __init__(self, data, list_IDs, batch_size=5, dim=(2,35,50), k = 624, 
                    p = 1, shuffle=True, till_end = False, only_test = False):
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
          X = np.empty((self.batch_size, *self.dim, self.k))
          y = [np.empty((self.batch_size, *self.dim))]*self.p

          y_inter = np.empty((self.batch_size, *self.dim, p))

          # Generate data
          lenn = len(list_IDs_temp)
          for i, ID in enumerate(list_IDs_temp):
              # Store Xtrain
              X[i,:,:,:,:] = self.data[:,:,:,ID:ID+k]
              # Store Ytrain
              y_inter[i,:,:,:,:] = self.data[:,:,:,ID+k:ID+k+p] 

          for j in range(self.p):
            y[j] = y_inter[:,:,:,:,j]
            y[j] = np.reshape(y[j], (lenn, -1)) 

          X = X.transpose((0,4,2,3,1))
          X = np.reshape(X, (X.shape[0],X.shape[1],-1))

          return X, y

    elif (model_type == 'cnn'):

      class DataGenerator(tf.keras.utils.Sequence): 
        'Generates data for Keras' 
        # IMPORTANT: In Synthetic jet: 1 piston cycle=624 snapshots---SET k=624
        def __init__(self, data, list_IDs, batch_size=5, dim=(2,35,50), 
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
            X = np.empty((self.batch_size, *self.dim, self.k))
            y = [np.empty((self.batch_size, *self.dim))]*self.p

            y_inter = np.empty((self.batch_size, *self.dim, p))

            # Generate data
            lenn = len(list_IDs_temp)
            for i, ID in enumerate(list_IDs_temp):
                # Store Xtrain
                X[i,:,:,:,:] = self.data[:,:,:,ID:ID+k]
                # Store Ytrain
                y_inter[i,:,:,:,:] = self.data[:,:,:,ID+k:ID+k+p]

            for j in range(self.p):
              y[j] = y_inter[:,:,:,:,j]
              y[j] = np.reshape(y[j], (lenn, -1)) 

            X = X.transpose((0,4,2,3,1))

            return X, y


    """
        Create training, validation and test sets
    """

    # Prepare the dataset indexes
    period_transitorio = 0
    stride_train = 1
    stride_val = 1
    stride_test = 1

    dim=(channels_n, dim_x, dim_y)
    
    k = 10  # number of snapshots used as predictors
    p = 2   # number of snapshots used as time-ahead predictions

    test_length = tensor_test.shape[-1]
    val_length  = int(val_size * total_length)
    train_length = int(train_size * total_length)

    if(batch_size < 0 or 
        not isinstance(batch_size, int) or 
        batch_size > np.min([val_length,train_length])-k-1):
        
        flag = {'check': False, 
                'text': "Error: Param 'batch_size' has to be an integer " + 
                "number greater than 0 and, in this case, lower than or equal to " + 
                f"{np.min([val_length,train_length])-k-1}"}
        outputs[0] = flag
        
        return outputs
        
    if int(train_length-period_transitorio-(k+p)) < 0:
        train_n = 0
    elif int((train_length-period_transitorio-(k+p))//stride_train) == 0:
        train_n = 1
    else: 
        train_n = int(((train_length-period_transitorio)-(k+p))//stride_train) + 1
        
    if int(test_length-period_transitorio-(k+p)) < 0:
        test_n = 0
    elif int((test_length-period_transitorio-(k+p))//stride_test) == 0:
        test_n = 1
    else: 
        test_n = int((test_length-period_transitorio-(k+p))//stride_test) + 1

    if int(val_length-(k+p)) < 0:
        val_n = 0
    elif int((val_length-(k+p))//stride_val) == 0:
        val_n = 1
    else: 
        val_n = int((val_length-(k+p))//stride_val) + 1

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

    j = 0
    for i in range(test_n):
        test_idxs[i] = j
        j = j+stride_test

    # Generators
    training_generator = DataGenerator(tensor_train_norm, train_idxs,  
                                        dim = dim, 
                                        batch_size = batch_size,
                                        k = k, p = p, till_end = False,
                                        only_test = False,
                                        shuffle = True)
    validation_generator = DataGenerator(tensor_train_norm, val_idxs, 
                                        dim = dim, 
                                        batch_size = batch_size,
                                        k = k, p = p, till_end = False,
                                        only_test = False,
                                        shuffle = False)

    test_generator = DataGenerator(tensor_test_norm, test_idxs, 
                                        dim = dim, 
                                        batch_size = batch_size,
                                        k = k, p = p, till_end = False,
                                        only_test = True,
                                        shuffle = False)
    """
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
    """
    # preparar Ytest
    test_n_adjusted = int(test_n/batch_size)*batch_size
    Ytest = [np.empty([test_n_adjusted, channels_n, dim_x, dim_y], dtype='float64')] * p
    Ytest_fl = [np.empty([test_n_adjusted, channels_n * dim_x * dim_y ], dtype='float64')] * p

    Ytest_inter = np.empty([test_n_adjusted, channels_n, dim_x, dim_y, p], dtype='float64')

    for i in range(test_n_adjusted):
        j = test_idxs[i]
        Ytest_inter[i,:,:,:,:] = tensor_test_norm[:,:,:,j+k:j+k+p]

    for r in range(p):    
      Ytest[r] = Ytest_inter[:,:,:,:,r]
      Ytest_fl[r] = np.copy(np.reshape(Ytest[r], (test_n_adjusted, -1)))

    outputs = [
        flag, training_generator, validation_generator, test_generator, tensor_test, 
        tensor_test_norm, min_val, range_val, dim_x, dim_y, channels_n, Ytest, 
        Ytest_fl, k, p
    ]
    
    return outputs

################################################################################
# RNN Model
################################################################################

def lstm_model(neurons, lr, shared_dim, act_fun, act_fun2, num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
               validation_generator, path_saved: str = './'):
  
    flag = {'check': True, 
            'text': None}

    if (num_epochs <= 0 or not isinstance(num_epochs, int)):
      
      flag = {'check': False, 
              'text': "Error: Param 'num_epochs' has to be an integer " + 
              "number greater than 0, e.g, num_epochs = 70"}
      
      return [flag, 0, 0]

    if (not isinstance(path_saved, str)):

      flag = {'check': False, 
              'text': "Error: Param 'path_saved' has to be an string " + 
              "idicating a path to save the files, e.g, path_saved = './"}
      
      return [flag, 0, 0]

    # Reset Model
    tf.keras.backend.clear_session

    # Create Model
    def create_model(neurons, lr, in_shape, out_dim, shared_dim, act_fun, act_fun2, p = 3):
      neurons = int(neurons)
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
        o.append( Dense(out_dim, activation=act_fun2)(s[i]))
    
      m = Model(inputs=x, outputs=o)

      opt = keras.optimizers.Adam(learning_rate=lr)
      m.compile(loss='mse', optimizer=opt, metrics=['mae'])
      return(m)

    in_shape = [k, dim_x * dim_y * channels_n]
    out_dim = dim_x * dim_y * channels_n 

    model= create_model(neurons, lr, in_shape, out_dim, shared_dim, act_fun, act_fun2, p)

    save_string = path_saved + 'lstm_model'

    # save the best weights 
    save_best_weights = save_string + '.h5'
    save_summary_stats = save_string + '.csv'
    save_last_weights = save_string + '_last_w.h5'
    save_results_metrics = save_string + '_results_metrics.csv'

    """
        Training
    """

    print("\n########################")
    print("TRAINING - RNN Model")
    print("########################\n")

    np.random.seed(247531338)

    t0 = time.time()
    # Model training
    callbacks = [ModelCheckpoint(
                save_best_weights, 
                monitor='val_loss', 
                save_best_only=True, 
                mode='auto'),
                EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                verbose=1, 
                mode='auto', 
                min_delta = 0.001)
                ]

    model.summary(print_fn=lambda x: st.text(x))

    history = model.fit(training_generator, 
                        validation_data=validation_generator,
                        epochs=num_epochs,
                        verbose=2,
                        callbacks=callbacks)
    t1 = time.time()
    print("Minutes elapsed for training: %f" % ((t1 - t0) / 60.))

    """
        Save Model
    """

    # save the last weights 
    model.save_weights(save_last_weights)

    # Aggregate the summary statistics
    summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                'train_loss': history.history['loss'],
                                'valid_loss': history.history['val_loss']})

    summary_stats.to_csv(save_summary_stats)
    fig, ax = plt.subplots()
    ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
    ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
    ax.grid()
    ax.legend()
    figName = path_saved + "loss_evolution_lstm_model.png"
    plt.savefig(figName, format = 'svg')
    st.pyplot(fig)

    # Find the min validation loss during the training
    # min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
    # print('Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))

    return flag, model, save_best_weights

################################################################################
# CNN Model
################################################################################

def cnn_model(shared_dim, lr, act_fun, act_fun2, num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
               validation_generator, path_saved: str = './'):

    flag = {'check': True, 
            'text': None}

    if (num_epochs <= 0 or not isinstance(num_epochs, int)):
      
      flag = {'check': False, 
              'text': "Error: Param 'num_epochs' has to be an integer " + 
              "number greater than 0, e.g, num_epochs = 70"}
      
      return [flag, 0, 0]

    if (not isinstance(path_saved, str)):

      flag = {'check': False, 
              'text': "Error: Param 'path_saved' has to be an string " + 
              "idicating a path to save the files, e.g, path_saved = './"}
      
      return [flag, 0, 0]

    # Reset Model
    tf.keras.backend.clear_session

    # Create Model
    def create_model(in_shape, out_dim, lr, p = 3, shared_dim = shared_dim, act_fun= act_fun, act_fun2 = act_fun2):
      x = Input(shape=in_shape)
      Fm = Input(shape=in_shape)
      
      v = Conv3D(5, 
                 kernel_size=(2,2,2), 
                 activation=act_fun, 
                 input_shape=in_shape, 
                 data_format='channels_last')(x)
      v = MaxPooling3D(pool_size=(1,2,2), 
                       padding='valid', 
                       data_format='channels_last', dtype='float32')(v)
      v = BatchNormalization()(v)
      v = Conv3D(10, 
                 kernel_size=(2,2,2), 
                 activation=act_fun, 
                 input_shape=in_shape, 
                 data_format='channels_last')(v)
      v = MaxPooling3D(pool_size=(1,2,2), 
                       padding='valid', 
                       data_format='channels_last', dtype='float32')(v)
      v = BatchNormalization()(v)
      v = Conv3D(20, 
                 kernel_size=(2,2,2), 
                 activation=act_fun, 
                 input_shape=in_shape, 
                 data_format='channels_last')(v)
      v = MaxPooling3D(pool_size=(1,2,2), 
                       padding='valid', 
                       data_format='channels_last', dtype='float32')(v)
      v = BatchNormalization()(v)
      v = Conv3D(p, 
                 kernel_size=(1,1,1), 
                 activation=act_fun, 
                 input_shape=in_shape, 
                 data_format='channels_last')(v)
      v = Permute((4,1,2,3))(v)
      v = Reshape((p,v.shape[2]*v.shape[3]*v.shape[4]))(v)

      tt = [1]*p
      
      r = TimeDistributed(Dense(shared_dim, activation=act_fun))(v)
      s = tf.split(r, tt, 1)
      for i in range(p):
        s[i] = Flatten()(s[i])

      o = []
      for i in range(p):
        o.append( Dense(out_dim, activation=act_fun2)(s[i]) )
    
      m = Model(inputs=x, outputs=o)
      optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
      m.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
      return(m)

    in_shape = [k, dim_x, dim_y, channels_n]
    out_dim = dim_x * dim_y * channels_n 

    # Define model
    model= create_model(in_shape, out_dim, lr, p, shared_dim, act_fun, act_fun2)

    save_string = path_saved + 'cnn_model'

    # save the best weights 
    save_best_weights = save_string + '.h5'
    save_summary_stats = save_string + '.csv'
    save_last_weights = save_string + '_last_w.h5'
    save_results_metrics = save_string + '_results_metrics.csv'

    """
        Training
    """

    print("\n########################")
    print("TRAINING - CNN Model")
    print("########################\n")

    np.random.seed(247531338)

    t0 = time.time()
    # Model training
    callbacks = [ModelCheckpoint(
                save_best_weights, 
                monitor='val_loss', 
                save_best_only=True, 
                mode='auto')
                ]
    
    model.summary(print_fn=lambda x: st.text(x))

    history = model.fit(training_generator, 
                        validation_data=validation_generator,
                        epochs=num_epochs,
                        verbose=2,
                        callbacks=callbacks)
    t1 = time.time()
    print("Minutes elapsed for training: %f" % ((t1 - t0) / 60.))

    """
        Save Model
    """

    # save the last weights 
    model.save_weights(save_last_weights)

    # Aggregate the summary statistics
    summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                'train_loss': history.history['loss'],
                                'valid_loss': history.history['val_loss']})

    summary_stats.to_csv(save_summary_stats)
    
    fig, ax = plt.subplots()
    ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
    ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
    ax.grid()
    ax.legend()
    figName = path_saved + "loss_evolution_cnn_model.png"
    plt.savefig(figName, format = 'svg')
    st.pyplot(fig)

    # Find the min validation loss during the training
    # min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
    # print('Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))

    return flag, model, save_best_weights

################################################################################
# Inference
################################################################################

def RRMSE(Tensor0, Reconst):
  RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst,newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
  return(RRMSE)

def smape(A, F):
    return ((100.0/len(A)) * 
    np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

def inference(model, save_best_weights: str, test_generator, 
              Ytest_fl: np.ndarray, Ytest: np.ndarray, min_val: float, 
              range_val: float, p: int, path_saved: str = './', 
              model_type = 'rnn'):
  
    print("\n###################")
    print("INFERENCE")
    print("###################\n")

    t0 = time.time()
    # Load model with best weights' configuration obtain from training
    model.load_weights(save_best_weights)
    Ytest_hat_fl = model.predict(
        test_generator, 
        max_queue_size=10, 
        workers=6, 
        use_multiprocessing=False, 
        verbose=0)
        
    t1 = time.time()
    print("Minutes elapsed to forecast: %f" % ((t1 - t0) / 60.))

    # print('Error measure of the first prediction for each sample on Test set')
    lag = 0
    num_sec = Ytest_hat_fl[0].shape[0]
    results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
    for i in range(num_sec):
        results_table.iloc[0,i] = mean_squared_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[1,i] = mean_absolute_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[2,i] = median_absolute_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[3,i] = r2_score(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[4,i] = smape(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[5,i] = RRMSE(np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))
    
    # print(results_table)
    savename = path_saved + "table_" + model_type + f"_first_prediction.csv"
    results_table.to_csv(savename, index=True)

    # print('Error measure of the second prediction for each sample on Test set')
    lag = 1
    num_sec = Ytest_hat_fl[0].shape[0]
    results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
    for i in range(num_sec):
        results_table.iloc[0,i] = mean_squared_error( Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[1,i] = mean_absolute_error(   Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[2,i] = median_absolute_error(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[3,i] = r2_score(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[4,i] = smape(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[5,i] = RRMSE( np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))

    # print(results_table)
    savename = path_saved + "table_" + model_type + f"_second_prediction.csv"
    results_table.to_csv(savename, index=True)

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

    print(results_table_global)
    savename = path_saved + "table_" + model_type + f"_mean.csv"
    results_table_global.to_csv(savename, index=True)

    # Reshape predictions and targets for plotting
    Ytest_hat_lag_0 = np.reshape(
        Ytest_hat_fl[0], 
        (Ytest[0].shape[0], Ytest[0].shape[1], 
        Ytest[0].shape[2], Ytest[0].shape[3]))
    
    Ytest_hat_lag_1 = np.reshape(
        Ytest_hat_fl[1], 
        (Ytest[1].shape[0], Ytest[1].shape[1], 
        Ytest[1].shape[2], Ytest[1].shape[3]))
    
    Ytest_lag_0 = np.reshape(
        Ytest_fl[0], 
        (Ytest[0].shape[0], Ytest[0].shape[1], 
        Ytest[0].shape[2], Ytest[0].shape[3]))
    
    Ytest_lag_1 = np.reshape(
        Ytest_fl[1], 
        (Ytest[1].shape[0], Ytest[1].shape[1], 
        Ytest[1].shape[2], Ytest[1].shape[3]))

    Ytest_hat_lag_0 = Ytest_hat_lag_0 * range_val + min_val
    Ytest_hat_lag_1 = Ytest_hat_lag_1 * range_val + min_val
    Ytest_lag_0 = Ytest_lag_0 * range_val + min_val
    Ytest_lag_1 = Ytest_lag_1 * range_val + min_val

    return [Ytest_hat_lag_0, Ytest_hat_lag_1, Ytest_lag_0, Ytest_lag_1]

def plot_results(Ytest_hat_lag_0: np.ndarray, Ytest_hat_lag_1: np.ndarray, 
                 Ytest_lag_0: np.ndarray, Ytest_lag_1: np.ndarray, index: int, 
                 path_saved: str = './', model_type = 'rnn'):
  
    flag = {'check': True, 
            'text': None}

    if (index < 0 or index >= Ytest_lag_0.shape[0] or not isinstance(index, int)):
      
        flag = {'check': False, 
                'text': "Error: Param 'index' has to be an non-negative integer " + 
                f"number lower than or equal to {Ytest_lag_0.shape[0] - 1} , " + 
                "e.g, num_epochs = 0"}
        
        return flag

    fig = plt.figure(figsize=(15,7))
    plt.subplot(2,3,1)
    plt.contourf(Ytest_lag_0[index,0,:,:], 10)
    plt.title(f"Ground Truth - Sample {2*index+1} of test set", fontsize = 14)
    plt.subplot(2,3,2)
    plt.contourf(Ytest_hat_lag_0[index,0,:,:], 10)
    plt.title(f"Prediction - Sample {2*index+1} of test set", fontsize = 14)
    plt.subplot(2,3,3)
    plt.contourf(np.abs(Ytest_hat_lag_0[index,0,:,:] - Ytest_lag_0[index,0,:,:]), 10)
    plt.title(f"Absolute Error - Sample {2*index+1} of test set", fontsize = 14)
    plt.colorbar()

    plt.subplot(2,3,4)
    plt.contourf(Ytest_lag_1[index,0,:,:], 10)
    plt.title(f"Ground Truth - Sample {2*index+2} of test set", fontsize = 14)
    plt.subplot(2,3,5)
    plt.contourf(Ytest_hat_lag_1[index,0,:,:], 10)
    plt.title(f"Prediction - Sample {2*index+2} of test set", fontsize = 14)
    plt.subplot(2,3,6)
    plt.contourf(np.abs(Ytest_hat_lag_1[index,0,:,:] - Ytest_lag_1[index,0,:,:]), 10)
    plt.title(f"Absolute Error - Sample {2*index+2} of test set", fontsize = 14)
    plt.colorbar()

    fig.tight_layout()

    figName = path_saved + "predictions_" + model_type + f"_model_sample_{2*index+1}.svg"
    plt.savefig(figName, format = 'svg')
    #plt.show()
    st.pyplot(fig)

    return flag

################################################################################
# Module Main
################################################################################ 

def menu(model_type):
    if model_type == 'cnn':
      st.title("CNN Model")
      st.write("""
This full deep learning model uses a Convolutional Neural Network (CNN) architecture to predict future snapshots (forecasting).
""")
      st.write(" ## CNN Model - Parameter Configuration")

    if model_type == 'rnn':
      st.title("RNN Model")
      st.write("""
This full deep learning model uses a Recurrent Neural Network (CNN) architecture to predict future snapshots (forecasting).
""")
      st.write(" ## RNN Model - Parameter Configuration")

    path0 = os.getcwd()

    wantedfile = 'Tensor_cylinder_Re100.mat'

    Ten_orig = data_fetch.fetch_data(path0, wantedfile)

    # Hyperparameters
    train_size = 0.75

    tensor_train = Ten_orig[...,:int(train_size*Ten_orig.shape[-1])]
    tensor_test = Ten_orig[...,int(train_size*Ten_orig.shape[-1]):]

    val_size = 0.25

    model_type_ = model_type.upper()

    batch_size = st.slider('Select Batch Size', 0, 8, value = 4, step = 2)

    if model_type == 'cnn':

      num_epochs = st.slider('Select training epochs', 0, 100, value = 3, step = 1)
      act_fun = st.selectbox('Select hidden layer activation function', ('relu', 'elu', 'sigmoid', 'softmax', 'tanh'))
      act_fun2 = st.selectbox('Select output layer activation function', ('linear', 'tanh', 'relu', 'sigmoid'))
      shared_dim = st.slider('Select number of shared dims', 1, 100, value = 30, step = 1)
      lr = st.number_input('Select the learning rate', min_value = 0., max_value = 0.1, value = 0.01, format = '%.4f')

    elif model_type == 'rnn':
      num_epochs = st.slider('Select training epochs', 0, 200, value = 20, step = 10)
      act_fun = st.selectbox('Select hidden layer activation function', ('linear', 'relu', 'elu', 'sigmoid', 'softmax', 'tanh'))
      act_fun2 = st.selectbox('Select output layer activation function', ('linear', 'tanh', 'relu', 'sigmoid'))
      neurons = st.slider('Select number of neurons per layer', 1, 150, value = 50, step = 1)
      shared_dim = st.slider('Select number of shared dims', 1, 150, value = 50, step = 1)
      lr = st.number_input('Select the learning rate', min_value = 0., max_value = 0.1, value = 0.01, format = '%.4f')


    path_saved = f'{path0}/{model_type_}_solution/'

    # Main
    go = st.button('Calculate')
    if go:
        with st.spinner("Please wait while the model is being trained"):
            if not os.path.exists(f'{path0}/{model_type_}_solution'):
              os.mkdir(f'{path0}/{model_type_}_solution')
            flag, training_generator, validation_generator, test_generator, tensor_test, \
            tensor_test_norm, min_val, range_val, dim_x, dim_y, channels_n, Ytest, \
            Ytest_fl, k, p = load_preprocess(tensor_train, tensor_test, 
                                train_size = train_size, val_size = val_size, 
                                batch_size = batch_size, model_type = model_type)

            if (not flag['check']):
                print(flag['text'])

            else:
                if (model_type == 'rnn'):

                    flag, model, save_best_weights = lstm_model(neurons, lr, shared_dim, act_fun, act_fun2, num_epochs, k, p, 
                                                                dim_x, dim_y, channels_n, training_generator, 
                                                                validation_generator, path_saved)
                    st.success('The model has been trained!')
                    st.write("### Model Training Results - RNN")

                else:

                    flag, model, save_best_weights = cnn_model(shared_dim, lr, act_fun, act_fun2, num_epochs, k, p, 
                                            dim_x, dim_y, channels_n, training_generator, 
                                            validation_generator, path_saved)
                    st.success('The model has been trained!')
                    st.write("### Model Training Results - 3D CNN")

                if (not flag['check']):

                    print(flag['text'])

                
                else:
                    Ytest_hat_lag_0, Ytest_hat_lag_1, \
                    Ytest_lag_0, Ytest_lag_1 = inference(model, 
                                    save_best_weights, test_generator, Ytest_fl, Ytest, 
                                    min_val, range_val, p, path_saved, model_type)

                    # Plot results
                    for checkPoint in range(3): 
                        index = int(checkPoint)
                        flag = plot_results(Ytest_hat_lag_0, Ytest_hat_lag_1, 
                                        Ytest_lag_0, Ytest_lag_1, index, 
                                        path_saved, model_type)

                        if (not flag['check']):
                            print(flag['text'])
                    
                        checkPoint = 1