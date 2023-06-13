def FullDL(model_type): 
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import time
  import math
  import os
  import data_load
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  from sklearn.metrics import mean_squared_error, mean_absolute_error
  from sklearn.metrics import median_absolute_error, r2_score

  import hdf5storage

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

  def is_float(string):
      try:
          float(string)
          return True
      except ValueError:
          return False
      
  def load_preprocess(k, p, tensor_train: np.ndarray, tensor_test: np.ndarray, 
                      train_size: float = 0.75, val_size: float = 0.25, 
                      batch_size: int = 8, model_type = 'rnn'):
      
      # Use to raise errors
      flag = {'check': True,
      'text': None}
      outputs = [0,0,0,0,0,0,0,0,0,0,0,0,0]
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
          Ytest_fl]
      
      return outputs

  ################################################################################
  # RNN Model
  ################################################################################

  def lstm_model(neurons, shared_dim, act_fun, act_fun2, lr, num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
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
                "indicating a path to save the files, e.g, path_saved = './"}
        
        return [flag, 0, 0]

      # Reset Model
      tf.keras.backend.clear_session

      # Create Model
      def create_model(neurons, in_shape, out_dim, shared_dim, act_fun, act_fun2, lr, p = 3):
        x = Input(shape=in_shape)
        
        v = LSTM(neurons)(x)
        v = Dense(p*neurons, activation= act_fun)(v)
        v = Reshape((p,neurons))(v)

        tt = [1]*p
        
        r = TimeDistributed(Dense(shared_dim, activation=act_fun))(v)
        s = tf.split(r, tt, 1)
        for i in range(p):
          s[i] = Flatten()(s[i])

        o = []
        for i in range(p):
          o.append( Dense(out_dim, activation=act_fun2)(s[i]))
      
        m = Model(inputs=x, outputs=o)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        m.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return(m)

      in_shape = [k, dim_x * dim_y * channels_n]
      out_dim = dim_x * dim_y * channels_n 

      model= create_model(neurons, in_shape, out_dim, shared_dim, act_fun, act_fun2, lr, p)
      
      print('Model Summary:\n')
      model.summary()

      save_string = path_saved + 'RNN_model'

      # save the best weights 
      save_best_weights = save_string + '.h5'
      save_summary_stats = save_string + '.csv'
      save_last_weights = save_string + '_last_w.h5'
      save_results_metrics = save_string + '_results_metrics.csv'

      """
          Training
      """

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

      print('\nTraining Model Please Wait...\n')
      history = model.fit(training_generator, 
                          validation_data=validation_generator,
                          epochs=num_epochs,
                          verbose=2,
                          callbacks=callbacks)
      t1 = time.time()
      print('\nModel Trained Successfully!')
      print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

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

      print('Please CLOSE all figures to continue the run\n')

      fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Loss function evolution')
      ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
      ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
      ax.grid()
      ax.legend()
      figName = path_saved + "loss_evolution_lstm_model.jpg"
      plt.savefig(figName, format = 'jpg')
      plt.show()
      
      return flag, model, save_best_weights
  
  def lstm_model_hp(num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
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
                "indicating a path to save the files, e.g, path_saved = './"}
        
        return [flag, 0, 0]

      # Reset Model
      tf.keras.backend.clear_session

      import keras_tuner as kt
      def create_model_hp(hp):
        hp_activation = hp.Choice('hidden_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu'])
        hp_neurons = hp.Int('hp_neurons', min_value = 10, max_value = 100, step = 10)
        hp_activation_1 = hp.Choice('output_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4])
        hp_shared_dims = hp.Int('shared dims', min_value = 10, max_value = 100, step = 10)
        
        x = Input(shape=[k, dim_x * dim_y * channels_n])
        
        v = LSTM(hp_neurons)(x)
        v = Dense(p*hp_neurons, activation = hp_activation)(v)
        v = Reshape((p,hp_neurons))(v)

        tt = [1]*p
        
        r = TimeDistributed(Dense(hp_shared_dims, activation = hp_activation))(v)
        s = tf.split(r, tt, 1)
        for i in range(p):
          s[i] = Flatten()(s[i])

        o = []
        for i in range(p):
          o.append(Dense(dim_x * dim_y * channels_n, activation = hp_activation_1)(s[i]))
      
        m = Model(inputs=x, outputs=o)
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        m.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return(m)

      if tuner_ == 'Hyperband':
        tuner = kt.Hyperband(create_model_hp, objective = 'val_loss', max_epochs = 10, factor = 3, directory = 'dir_1', project_name = 'x', overwrite = True)
    
      elif tuner_ == 'RandomSearch':
          tuner = kt.RandomSearch(create_model_hp, objective = 'val_loss', max_trials = 10, directory = 'dir_1', project_name = 'x', overwrite = True)

      elif tuner_ == 'Bayesian':
          tuner = kt.BayesianOptimization(create_model_hp, objective = 'val_loss', max_trials = 10, beta = 3, directory = 'dir_1', project_name = 'x', overwrite = True)
      
      stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)

      print('\nSearching for optimal hyperparameters...')

      tuner.search(training_generator,
                validation_data=validation_generator,
                epochs=10,
                verbose=1, 
                callbacks=[stop_early])

      best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

      print('\n-----------------------------')
      print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {best_hps.get('hidden_layer_activation_function')}
Output Layer activation function: {best_hps.get('output_layer_activation_function')}
Number of neurons: {best_hps.get('hp_neurons')}
Number of shared dimensions: {best_hps.get('shared dims')}
Learning rate: {best_hps.get('learning_rate')}
Loss function: 'mse'
      ''')

      model = tuner.hypermodel.build(best_hps)

      print('Model Summary:\n')
      model.summary()

      save_string = path_saved + 'RNN_model'

      # save the best weights 
      save_best_weights = save_string + '.h5'
      save_summary_stats = save_string + '.csv'
      save_last_weights = save_string + '_last_w.h5'
      save_results_metrics = save_string + '_results_metrics.csv'

      t0 = time.time()

      callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_delta = 0.001)]

      print('\nTraining Model Please Wait...\n')

      history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks)

      t1 = time.time()
      print('\nModel Trained Successfully!')

      print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

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

      print('Please CLOSE all figures to continue the run\n')

      fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Loss function evolution')
      ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
      ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
      ax.grid()
      ax.legend()
      figName = path_saved + "loss_evolution_lstm_model.jpg"
      plt.savefig(figName, format = 'jpg')
      plt.show()
      
      return flag, model, save_best_weights

  ################################################################################
  # CNN Model
  ################################################################################

  def cnn_model(shared_dim, act_fun, act_fun1, lr, num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
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
                "indicating a path to save the files, e.g, path_saved = './"}
        
        return [flag, 0, 0]

      # Reset Model
      tf.keras.backend.clear_session

      # Create Model
      def create_model(in_shape,  out_dim, p = 3, shared_dim = shared_dim, act_fun= act_fun, act_fun1 = act_fun1, lr = lr):
        x = Input(shape=in_shape)
        Fm = Input(shape=in_shape)
        
        v = Conv3D(5, 
                  kernel_size=(2,2,2), 
                  activation=act_fun, 
                  input_shape=in_shape, 
                  data_format='channels_last')(x)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
        v = BatchNormalization()(v)
        v = Conv3D(10, 
                  kernel_size=(2,2,2), 
                  activation=act_fun, 
                  input_shape=in_shape, 
                  data_format='channels_last')(v)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
        v = BatchNormalization()(v)
        v = Conv3D(20, 
                  kernel_size=(2,2,2), 
                  activation=act_fun, 
                  input_shape=in_shape, 
                  data_format='channels_last')(v)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
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
          o.append( Dense(out_dim, activation=act_fun1)(s[i]))
      
        m = Model(inputs=x, outputs=o)
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        m.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return(m)

      in_shape = [k, dim_x, dim_y, channels_n]
      out_dim = dim_x * dim_y * channels_n 

      # Define model
      model= create_model(in_shape, out_dim, p, shared_dim)

      print('Model Summary:\n')
      model.summary()

      save_string = path_saved + 'CNN_model'

      # save the best weights 
      save_best_weights = save_string + '.h5'
      save_summary_stats = save_string + '.csv'
      save_last_weights = save_string + '_last_w.h5'
      save_results_metrics = save_string + '_results_metrics.csv'

      """
          Training
      """

      np.random.seed(247531338)

      t0 = time.time()
      # Model training
      callbacks = [ModelCheckpoint(
                  save_best_weights, 
                  monitor='val_loss', 
                  save_best_only=True, 
                  mode='auto')
                  ]
      
      print('\nTraining Model Please Wait...\n')  
      history = model.fit(training_generator, 
                          validation_data=validation_generator,
                          epochs=num_epochs,
                          verbose=2,
                          callbacks=callbacks)
      t1 = time.time()
      print('\nModel Trained Successfully!')
      print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

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
      
      print('Please CLOSE all figures to continue the run\n')

      fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Loss function evolution')
      ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
      ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
      ax.grid()
      ax.legend()
      figName = path_saved + "loss_evolution_cnn_model.jpg"
      plt.savefig(figName, format = 'jpg')
      plt.show()

      return flag, model, save_best_weights

  def cnn_model_hp(num_epochs, k, p, dim_x, dim_y, channels_n, training_generator, 
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
      import keras_tuner as kt
      def create_model_hp(hp):
        hp_activation = hp.Choice('hidden_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu'])
        hp_activation_1 = hp.Choice('output_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4])
        hp_shared_dims = hp.Int('shared dims', min_value = 10, max_value = 100, step = 10)

        in_shape = [k, dim_x, dim_y, channels_n]

        x = Input(shape=in_shape)
        
        v = Conv3D(5, 
                  kernel_size=(2,2,2), 
                  activation=hp_activation, 
                  input_shape=in_shape, 
                  data_format='channels_last')(x)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
        v = BatchNormalization()(v)
        v = Conv3D(10, 
                  kernel_size=(2,2,2), 
                  activation=hp_activation, 
                  input_shape=in_shape, 
                  data_format='channels_last')(v)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
        v = BatchNormalization()(v)
        v = Conv3D(20, 
                  kernel_size=(2,2,2), 
                  activation=hp_activation, 
                  input_shape=in_shape, 
                  data_format='channels_last')(v)
        v = MaxPooling3D(pool_size=(1,2,2), 
                        padding='valid', 
                        data_format='channels_last')(v)
        v = BatchNormalization()(v)
        v = Conv3D(p, 
                  kernel_size=(1,1,1), 
                  activation=hp_activation, 
                  input_shape=in_shape, 
                  data_format='channels_last')(v)
        v = Permute((4,1,2,3))(v)
        v = Reshape((p,v.shape[2]*v.shape[3]*v.shape[4]))(v)

        tt = [1]*p
        
        r = TimeDistributed(Dense(hp_shared_dims, activation=hp_activation))(v)
        s = tf.split(r, tt, 1)
        for i in range(p):
          s[i] = Flatten()(s[i])

        o = []
        for i in range(p):
          o.append(Dense(dim_x * dim_y * channels_n , activation=hp_activation_1)(s[i]))
      
        m = Model(inputs=x, outputs=o)
        optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
        m.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return(m)

    
      if tuner_ == 'Hyperband':
        tuner = kt.Hyperband(create_model_hp, objective = 'val_loss', max_epochs = 10, factor = 3, directory = 'dir_1', project_name = 'x', overwrite = True)
    
      elif tuner_ == 'RandomSearch':
          tuner = kt.RandomSearch(create_model_hp, objective = 'val_loss', max_trials = 10, directory = 'dir_1', project_name = 'x', overwrite = True)

      elif tuner_ == 'Bayesian':
          tuner = kt.BayesianOptimization(create_model_hp, objective = 'val_loss', max_trials = 10, beta = 3, directory = 'dir_1', project_name = 'x', overwrite = True)
      
      stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)

      print('\nSearching for optimal hyperparameters...\n')

      tuner.search(training_generator,
                validation_data=validation_generator,
                epochs=3,
                verbose=1, 
                callbacks=[stop_early])

      best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
      print('\n-----------------------------')
      print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {best_hps.get('hidden_layer_activation_function')}
Output Layer activation function: {best_hps.get('output_layer_activation_function')}
Number of shared dimensions: {best_hps.get('shared dims')}
Learning rate: {best_hps.get('learning_rate')}
Loss function: 'mse'
      ''')

      model = tuner.hypermodel.build(best_hps)

      print('Model Summary:\n')
      model.summary()

      save_string = path_saved + 'CNN_model'

      # save the best weights 
      save_best_weights = save_string + '.h5'
      save_summary_stats = save_string + '.csv'
      save_last_weights = save_string + '_last_w.h5'
      save_results_metrics = save_string + '_results_metrics.csv'

      t0 = time.time()

      callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto', min_delta = 0.0001)]

      print('\nTraining Model Please Wait...\n')

      history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks)

      t1 = time.time()
      print('\nModel Trained Successfully!')

      print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

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

      print('Please CLOSE all figures to continue the run\n')

      fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Loss function evolution')
      ax.plot(summary_stats.train_loss, 'b', label = 'Training loss') # blue
      ax.plot(summary_stats.valid_loss, 'g--', label = 'Validation loss') # green
      ax.grid()
      ax.legend()
      figName = path_saved + "loss_evolution_lstm_model.jpg"
      plt.savefig(figName, format = 'jpg')
      plt.show()
      
      return flag, model, save_best_weights

  ################################################################################
  # Inference
  ################################################################################

  def RRMSE (real, predicted):
        RRMSE = np.linalg.norm(np.reshape(real-predicted,newshape=(np.size(real),1)),ord=2)/np.linalg.norm(np.reshape(real,newshape=(np.size(real),1)))
        return RRMSE

  def smape(A, F):
      return ((100.0/len(A)) * 
      np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

  def inference(model, save_best_weights: str, test_generator, 
                Ytest_fl: np.ndarray, Ytest: np.ndarray, min_val: float, 
                range_val: float, p: int, path_saved: str = './', 
                model_type = 'rnn'):
    
    
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
      print('\nModel predicting. Please wait')
      print(f"\nPrediction complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

      if not output0 or output0.strip().lower() in ['.npy', 'npy']:
        np.save(path_saved + '/TensorPred.npy', Ytest_hat_fl)
            
      elif output0.strip().lower() in ['.mat', 'mat']:
          mdic = {"Pred": Ytest_hat_fl}
          file_mat= str(path_saved + '/TensorPred.mat')
          hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

      # print('Error measure of the first prediction for each sample on Test set')
      lag = 0
      num_sec = Ytest_hat_fl[0].shape[0]

      print('\nPerformance measures on Test data, per sec')
      results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
      for i in range(num_sec):
          results_table.iloc[0,i] = mean_squared_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[1,i] = mean_absolute_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[2,i] = median_absolute_error(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[3,i] = r2_score(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[4,i] = smape(Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[5,i] = RRMSE(np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))
      print(results_table)
      
      # print(results_table)
      savename = path_saved + "table_" + model_type + f"_first_prediction.csv"
      results_table.to_csv(savename, index=True)

      # print('Error measure of the second prediction for each sample on Test set')
      lag = 1
      num_sec = Ytest_hat_fl[0].shape[0]
      print('\nPerformance measures on Test data, per sec, for time lag = 1')
      results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
      for i in range(num_sec):
          results_table.iloc[0,i] = mean_squared_error( Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[1,i] = mean_absolute_error(   Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[2,i] = median_absolute_error(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[3,i] = r2_score(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[4,i] = smape(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
          results_table.iloc[5,i] = RRMSE( np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))

      print(results_table)
      savename = path_saved + "table_" + model_type + f"_second_prediction.csv"
      results_table.to_csv(savename, index=True)

      print('\nPerformance measures on Test data, for all time, per time-ahead lag')
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
      
      fig = plt.figure(figsize=(15,7), num = 'CLOSE TO CONTINUE RUN - Snapshot comparison')
      plt.subplot(2,3,1)
      plt.contourf(Ytest_lag_0[index,0,:,:], 10)
      plt.title(f"Ground Truth - Sample {2*index+1} of test set")
      plt.subplot(2,3,2)
      plt.contourf(Ytest_hat_lag_0[index,0,:,:], 10)
      plt.title(f"Prediction - Sample {2*index+1} of test set")
      plt.subplot(2,3,3)
      plt.contourf(np.abs(Ytest_hat_lag_0[index,0,:,:] - Ytest_lag_0[index,0,:,:]), 10)
      plt.title(f"Absolute Error - Sample {2*index+1} of test set")
      plt.colorbar()

      plt.subplot(2,3,4)
      plt.contourf(Ytest_lag_1[index,0,:,:], 10)
      plt.title(f"Ground Truth - Sample {2*index+2} of test set")
      plt.subplot(2,3,5)
      plt.contourf(Ytest_hat_lag_1[index,0,:,:], 10)
      plt.title(f"Prediction - Sample {2*index+2} of test set")
      plt.subplot(2,3,6)
      plt.contourf(np.abs(Ytest_hat_lag_1[index,0,:,:] - Ytest_lag_1[index,0,:,:]), 10)
      plt.title(f"Absolute Error - Sample {2*index+2} of test set")
      plt.colorbar()

      fig.tight_layout()

      figName = path_saved + "predictions_" + model_type + f"_model_sample_{2*index+1}.jpg"
      plt.savefig(figName, format = 'jpg')
      #plt.show()
      plt.show()

      return flag

  ################################################################################
  # Module Main
  ################################################################################ 
  model = model_type.upper()
  print(f"\n{model} Model\n")
  print('-----------------------------')
  print("Inputs: \n")
  path0 = os.getcwd()
  timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

  while True:
        filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
        print('\n\tWarning: This model can only be trained with 2-Dimensional data (as in: (variables, nx, ny, time))\n')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

  Ten_orig, _ = data_load.main(filetype)

  while True:
    train_size = input('Select train data percentage (0-1). Continue with 0.75: ') # test set proportion
    if not train_size:
        train_size = 0.75
        break
    elif is_float(train_size):
        train_size = np.round(float(train_size), 2)
        break
    else:
        print('\tError: Please select a number\n')
    
  tensor_train = Ten_orig[...,:int(train_size*Ten_orig.shape[-1])]
  tensor_test = Ten_orig[...,int(train_size*Ten_orig.shape[-1]):]

  val_size = np.round(1.0-train_size, 2)

  while True:
    batch_size = input('Select batch size (recommended power of 2). Continue with 8: ')
    if not batch_size:
       batch_size = 8
       break
    elif batch_size.isdigit():
        batch_size = int(batch_size)
        break
    else:
        print('\tError: Select a valid number (must be integer)\n')
  if model_type == 'cnn':
    while True:
      epoch = input('Select training epochs. Continue with 5: ')
      if not epoch:
          epoch = 5
          break
      elif epoch.isdigit():
          epoch = int(epoch)
          break
      else:
          print('\tError: Select a valid number (must be integer)\n')
  
  elif model_type == 'rnn':
    while True:
      epoch = input('Select training epochs. Continue with 20: ')
      if not epoch:
          epoch = 20
          break
      elif epoch.isdigit():
          epoch = int(epoch)
          break
      else:
          print('\tError: Select a valid number (must be integer)\n')

  while True:
        k = input('Select number of snapshots used as predictors. Continue with 10: ')  # number of snapshots used as predictors
        if not k:
            k = 10
            break
        elif k.isdigit():
            k = int(k)
            break
        else: 
            print('\tError: Select a valid number (must be integer)\n')

  while True:
        p = input('Select number of snapshots used as time-ahead predictions. Continue with 2: ')  # number of snapshots used as predictors
        if not p:
            p = 2
            break
        elif p.isdigit():
            p = int(p)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')
      
  print('\n-----------------------------')
  print('Model Parameters: \n')

  while True:
        hyper = input('Use optimal hyperparameters? (y/n). Continue with No: ')
        if not hyper or hyper.strip().lower() in ['no', 'n']:
            hyper = 'No'
            break
        elif hyper.strip().lower() in ['yes', 'y']:
            hyper = 'Yes'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

  if hyper == 'yes':
    print('''
Available hyperparameter tuners:
1) RandomSearch: All the hyperparameter combinations are chosen randomly.
2) Hyperband: Randomly sample all the combinations of hyperparameter and train the model for few epochs with the combinations, selecting the best candidates based on the results.
3) BayesianOptimization: Chooses first few combinations randomly, then based on the performance on these hyperparameters it chooses the next best possible hyperparameters.
    ''')
    while True:
        tuner_ = input('Select a hyperparameter tuner (1/2/3). Continue with RandomSearch: ')
        if not tuner_ or tuner_ == '1':
            tuner_ = 'RandomSearch'
            break
        elif tuner_ == '2':
            tuner_ = 'Hyperband'
            break
        elif tuner_ == '3':
            tuner_ = 'Bayesian'
            break
        else:
            print('\tError: Select a valid tuner\n')
  
  if hyper == 'No':
    while True:
        act_fun = input('Select hidden layer activation function (relu, elu, softmax, sigmoid, tanh, linear). Continue with relu: ')
        if not act_fun or act_fun.strip().lower() == 'relu':
            act_fun = 'relu'
            break
        elif act_fun.strip().lower() == 'elu':
            act_fun = 'elu'
            break
        elif act_fun.strip().lower() == 'softmax':
            act_fun = 'softmax'
            break
        elif act_fun.strip().lower() == 'sigmoid':
            act_fun = 'sigmoid'
            break
        elif act_fun.strip().lower() == 'tanh':
            act_fun = 'tanh'
            break
        elif act_fun.strip().lower() == 'linear':
            act_fun = 'linear'
            break
        else:
            print('\tError: Please select a valid option\n')
    
    while True:
            act_fun2 = input('Select output layer activation function (tanh, relu, sigmoid, linear). Continue with tanh: ')
            if not act_fun2 or act_fun2.strip().lower() == 'tanh':
                act_fun2 = 'tanh'
                break
            elif act_fun2.strip().lower() == 'relu':
                act_fun2 = 'relu'
                break
            elif act_fun2.strip().lower() == 'sigmoid':
                act_fun2 = 'sigmoid'
                break
            elif act_fun2.strip().lower() == 'linear':
                act_fun2 = 'linear'
                break
            else:
                print('\tError: Please select a valid option\n')
    if model_type == 'rnn':            
      while True:
              neurons = input('Select number of neurons per layer. Continue with 100: ')
              if not neurons:
                  neurons = 100
                  break
              elif neurons.isdigit():
                  neurons = int(neurons)
                  break
              else:
                  print('\tError: Select a valid number (must be integer)\n')

    while True:
        shared_dim = input('Select number of shared dims. Continue with 100: ')
        if not shared_dim:
            shared_dim = 100
            break
        elif shared_dim.isdigit():
            shared_dim = int(shared_dim)
            break
        else: 
            print('\tError: Select a valid number (must be integer)\n')
    
    while True:
            lr = input('Select model learning rate. Continue with 2e-3: ') 
            if not lr: 
                lr = 0.002
                break
            elif is_float(lr):
                lr = float(lr)
                break
            else:
                print('\tError: Please select a number\n')
  
    if model_type == 'rnn':
      print('\n-----------------------------')
      print(f'''
  HYPERPARAMETERS SUMMARY:\n
  Hidden Layer activation function: {act_fun}
  Output Layer activation function: {act_fun2}
  Number of neurons: {neurons}
  Number of shared dimensions: {shared_dim}
  Learning rate: {lr}
  Loss function: mse
                ''')
      print('-----------------------------')
    else:
      print('\n-----------------------------')
      print(f'''
  HYPERPARAMETERS SUMMARY:\n
  Hidden Layer activation function: {act_fun}
  Output Layer activation function: {act_fun2}
  Number of shared dimensions: {shared_dim}
  Learning rate: {lr}
  Loss function: mse
              ''')
      print('-----------------------------')

  print('Outputs:\n ')

  filen = input('Enter folder name to save the outputs or continue with default folder name: ')
  if not filen:
      filen = f'{timestr}_{model}_solution'
  else:
      filen = f'{filen}'

  if not os.path.exists(f'{path0}/{filen}'):
      os.mkdir(f'{path0}/{filen}')

  while True:
    num = input(f'Select number snapshot to plot (must be multiple of {p}): ')
    if num.isdigit():
       if int(num) % int(p) == 0 and int(num) > 1:
        num = int(int(num)/int(p))
        break
       else:
          print('\tError: introduced value is not multiple of two\n')
    else:
      print('\tError: Select a valid number format (must be integer)\n')

  while True:
    output0 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
    if not output0 or output0.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
        break
    else:
        print('\tError: Please select a valid output format\n')

  path_saved = f'{path0}/{filen}/'

    
  flag, training_generator, validation_generator, test_generator, tensor_test, \
  tensor_test_norm, min_val, range_val, dim_x, dim_y, channels_n, Ytest, \
  Ytest_fl = load_preprocess(k, p, tensor_train, tensor_test, 
                      train_size = train_size, val_size = val_size, 
                      batch_size = batch_size, model_type = model_type)

  if (not flag['check']):
      print(flag['text'])

  else:
      if (model_type == 'rnn'):
          if hyper == 'No':
            flag, model, save_best_weights = lstm_model(neurons, shared_dim, act_fun, act_fun2, lr, epoch, k, p, 
                                                        dim_x, dim_y, channels_n, training_generator, 
                                                        validation_generator, path_saved)
          if hyper == 'Yes':
            flag, model, save_best_weights = lstm_model_hp(epoch, k, p, dim_x, dim_y, channels_n, training_generator, 
                      validation_generator, path_saved)

      elif (model_type == 'cnn'):
          if hyper == 'No':
            flag, model, save_best_weights = cnn_model(shared_dim, act_fun, act_fun2, lr, epoch, k, p, 
                                    dim_x, dim_y, channels_n, training_generator, 
                                    validation_generator, path_saved)
          if hyper == 'Yes':
            flag, model, save_best_weights = cnn_model_hp(epoch, k, p, dim_x, dim_y, channels_n, training_generator, 
                      validation_generator, path_saved)

      if (not flag['check']):

          print(flag['text'])

      
      else:

          Ytest_hat_lag_0, Ytest_hat_lag_1, \
          Ytest_lag_0, Ytest_lag_1 = inference(model, 
                          save_best_weights, test_generator, Ytest_fl, Ytest, 
                          min_val, range_val, p, path_saved, model_type)

          # Plot results
          print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
          print('Please CLOSE all figures to continue the run\n')
          for checkPoint in range(num): 
              index = int(checkPoint)
              flag = plot_results(Ytest_hat_lag_0, Ytest_hat_lag_1, 
                              Ytest_lag_0, Ytest_lag_1, index, 
                              path_saved, model_type)

              if (not flag['check']):
                  print(flag['text'])
          
              checkPoint = 1

