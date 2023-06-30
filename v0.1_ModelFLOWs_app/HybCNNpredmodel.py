def hybCNN():
    import numpy as np
    import pandas as pd
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import matplotlib.pyplot as plt
    import time
    import math

    import hdf5storage
    import data_load
    import scipy.io


    from numpy import linalg as LA


    from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

    import tensorflow as tf
    from tensorflow import keras

    from tensorflow.keras.layers import Dense, Reshape, TimeDistributed, Flatten, Convolution1D
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    print("\nHybrid SVD + CNN Model\n")
    print('-----------------------------')
    print("Inputs: \n")

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def mean_absolute_percentage_error(y_true, y_pred): 
        epsilon = 1e-10 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

    def smape(A, F):
        return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

    def RRMSE (real, predicted):
        RRMSE = np.linalg.norm(np.reshape(real-predicted,newshape=(np.size(real),1)),ord=2)/np.linalg.norm(np.reshape(real,newshape=(np.size(real),1)))
        return RRMSE

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

    # Load Data
    path0 = os.getcwd()
    timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

    while True:
        filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
        print('\n\tWarning: This model can only be trained with 2-Dimensional data (as in: (variables, nx, ny, time))\n')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    data, _ = data_load.main(filetype)


    data = tf.transpose(data, (3, 2, 1, 0))

    print('\n-----------------------------')
    print("SVD Parameters: \n")

    while True:
        n_modes = input('Select number of SVD modes. Continue with 18: ')
        if not n_modes:
            n_modes = 18
            break
        elif n_modes.isdigit():
            n_modes = int(n_modes)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')

    while True:
        decision1 = input('Select first scaling: scaling of implemented variables (auto, pareto, range, no-scaling). Continue with auto: ')
        if not decision1 or decision1.strip().lower() == 'auto':
            decision1 = 'auto'
            break
        elif decision1.strip().lower() == 'pareto':
            decision1 = 'pareto'
            break
        elif decision1.strip().lower() == 'range':
            decision1 = 'range'
            break
        elif decision1.strip().lower() == 'no-scaling':
            decision1 = 'no-scaling'
            break
        else:
            print('\tError: Select a valid option\n')

    while True:
        decision2 = input('Select the scaling of the SVD temporal coefficients implemented (MaxPerMode, auto, range, no-scaling). Continue with MaxPerMode: ')
        if not decision2 or decision2.strip().lower() == 'maxpermode':
            decision2 = 'MaxPerMode'
            break
        elif decision2.strip().lower() == 'auto':
            decision2 = 'auto'
            break
        elif decision2.strip().lower() == 'range':
            decision2 = 'range'
            break
        elif decision2.strip().lower() == 'no-scaling':
            decision2 = 'no-scaling'
            break
        else:
            print('\tError: Select a valid option\n')

    # Inputs
    # Model configuration
    print('\n-----------------------------')
    print('Neural Network Training Configuration: \n')

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

    if hyper == 'Yes':
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

    while True:
        test_prop = input('Select test data percentage (0-1). Continue with 0.20: ') # test set proportion
        if not test_prop:
            test_prop = 0.2
            break
        elif is_float(test_prop):
            test_prop = float(test_prop)
            break
        else:
            print('\tError: Please select a number\n')

    while True:   
        val_prop = input('Select validation data percentage (0-1). Continue with 0.20: ') # test set proportion
        if not val_prop:
            val_prop = 0.2
            break
        elif is_float(val_prop):
            val_prop = float(val_prop)# validation set proportion val_length = (1-test_prop)*val_prop // train_length = (1-test_prop)*(1-val_prop)
            break
        else: 
            print('\tError: Please select a number\n')

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

    while True:
        epoch = input('Select training epochs. Continue with 500: ')
        if not epoch:
            epoch = 500
            break
        elif epoch.isdigit():
            epoch = int(epoch)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')

    while True:
        k = input('Select number of snapshots used as predictors (min. is 6). Continue with 10: ')  # number of snapshots used as predictors
        if not k:
            k = 10
            break
        elif k.isdigit():
            if int(k) >= 6:
                k = int(k)
                break
            else:
                print('\tError: Invalid value')
        else: 
            print('\tError: Select a valid number (must be integer)\n')

    while True:
        p = input(f'Select number of snapshots used as time-ahead predictions (max. is {k-4}). Continue with {k-4}: ')  # number of snapshots used as predictors
        if not p:
            p = k-4
            break
        elif p.isdigit():
            if int(p) <= k-4:
                p = int(p)
                break
            else:
                print('\tError: Invalid value')
        else:
            print('\tError: Select a valid number (must be integer)\n')

    if hyper == 'Yes':
        pass

    if hyper == 'No':
        print('\n-----------------------------')
        print('Model Parameters: \n')
        while True:
            act_func1 = input('Select hidden layer activation function (relu, elu, softmax, sigmoid, tanh, linear). Continue with relu: ')
            if not act_func1 or act_func1.strip().lower() == 'relu':
                act_func1 = 'relu'
                break
            elif act_func1.strip().lower() == 'elu':
                act_func1 = 'elu'
                break
            elif act_func1.strip().lower() == 'softmax':
                act_func1 = 'softmax'
                break
            elif act_func1.strip().lower() == 'sigmoid':
                act_func1 = 'sigmoid'
                break
            elif act_func1.strip().lower() == 'tanh':
                act_func1 = 'tanh'
                break
            elif act_func1.strip().lower() == 'linear':
                act_func1 = 'linear'
                break
            else:
                print('\tError: Please select a valid option\n')

        while True:
            act_func2 = input('Select output layer activation function (tanh, relu, sigmoid, linear). Continue with tanh: ')
            if not act_func2 or act_func2.strip().lower() == 'tanh':
                act_func2 = 'tanh'
                break
            elif act_func2.strip().lower() == 'relu':
                act_func2 = 'relu'
                break
            elif act_func2.strip().lower() == 'sigmoid':
                act_func2 = 'sigmoid'
                break
            elif act_func2.strip().lower() == 'linear':
                act_func2 = 'linear'
                break
            else:
                print('\tError: Please select a valid option\n')

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
        
        while True:
            lf = input('Select loss function (mse, custom_loss, mae). Continue with mse: ')
            if not lf or lf.strip().lower() == 'mse':
                lf = 'mse'
                break
            elif lf.strip().lower() == 'custom_loss':
                lf = custom_loss
                break
            elif lf.strip().lower() == 'mae':
                lf = 'mae'
                break
            else:
                print('\tError: Please select a valid option\n')

        print('\n-----------------------------')
        print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {act_func1}
Output Layer activation function: {act_func2}
Number of neurons: {neurons}
Number of shared dimensions: {shared_dim}
Learning rate: {lr}
Loss function: {lf}
            ''')

    print('-----------------------------')
    print('Outputs: \n')
    
    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_hybCNN_Solution'
    else:
        filen = f'{filen}'

    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f'{path0}/{filen}')

    while True:
        output0 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not output0 or output0.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')
    while True:
        output1 = input('Save error made by SVD for each variable (y/n). Continue with Yes: ') 
        if not output1 or output1.strip().lower() in ['y', 'yes']:
            output1 = 'yes'
            break
        elif output1.strip().lower() in ['n', 'no']:
            output1 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output3 = input('Plot loss function evolution (y/n). Continue with Yes: ')
        if not output3 or output3.strip().lower() in ['y', 'yes']:
            output3 = 'yes'
            break
        elif output3.strip().lower() in ['n', 'no']:
            output3 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output4 = input('Save RRMSE for each variable (Original data) (y/n). Continue with Yes: ')
        if not output4 or output4.strip().lower() in ['y', 'yes']:
            output4 = 'yes'
            break
        elif output4.strip().lower() in ['n', 'no']:
            output4 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output5 = input('Save RRMSE for each variable (Truncated data) (y/n). Continue with Yes: ')
        if not output5 or output5.strip().lower() in ['y', 'yes']:
            output5 = 'yes'
            break
        elif output5.strip().lower() in ['n', 'no']:
            output5 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output6 = input('Plot RRMSE temporal matrix (y/n). Continue with Yes: ')
        if not output6 or output6.strip().lower() in ['y', 'yes']:
            output6 = 'yes'
            break
        elif output6.strip().lower() in ['n', 'no']:
            output6 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        output7 = input('Plot mode comparison (y/n). Continue with Yes: ')
        if not output7 or output7.strip().lower() in ['y', 'yes']:
            output7 = 'yes'
            break
        elif output7.strip().lower() in ['n', 'no']:
            output7 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    if output7 == 'yes': # Indicate the number of the mode you want to compare
        while True:
            output71 = input('Select the number of the first "n" modes to compare. Continue with 5: ')
            if not output71:
                maxN = 5
                nModes = list(range(1, maxN+1))
                break
            elif output71.isdigit():
                maxN = int(output71)
                nModes = list(range(1, maxN+1))
                break
            else:
                print('\tError: Select a valid number (must be integer)\n')

    while True:
        output8 = input('Plot snapshot comparison (y/n). Continue with Yes: ')
        if not output8 or output8.strip().lower() in ['y', 'yes']:
            output8 = 'yes'
            break
        elif output8.strip().lower() in ['n', 'no']:
            output8 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    if output8 == 'yes': # Indicate the number of the variable and time step (nv, nt)
        while True:
            output81 = input(f'Select number of first "n" variables. Continue with first {int(data.shape[-1])}: ')
            if not output81:
                output81 = int(data.shape[-1])
                break
            elif output81.isdigit():
                output81 = int(output81)
                break
            else:
                print('\tError: Select a valid number (must be integer)\n')

        while True:
            output82 = input(f'Select time step to plot. Continue with {int(data.shape[0]/100)}: ')
            if not output82:
                output82 = int(data.shape[0]/100)
                break
            elif output82.isdigit(): 
                output82 = int(output82)
                break
            else:
                print('\tError: Select a valid number (must be integer)\n')

        index = np.array([[i, output82] for i in range(output81)])

    nt, ny, nx, nv = np.shape(data)

    # Original tensor reconstruction
    Mat_orig = np.reshape(data,[nt,nv*nx*ny])
    Mat_orig = np.transpose(Mat_orig)
    Tensor_orig = np.reshape(Mat_orig,[nv,nx,ny,nt], order='F')
    sum_spec = np.sum(np.sum(Tensor_orig[1:,:,:,:],axis=3),axis=0)/(nt)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(y, x)

    # Centering and scaling
    if decision1 == 'no-scaling':
        pass
    else:
        #Dummy variables
        print('\nScaling Data')
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

    print('Data Scaled\n')
    # Perform SVD
    print('Performing SVD')
    U, S, V = np.linalg.svd(Mat_orig, full_matrices=False)

    Modes = n_modes
    S = np.diag(S)

    U = U[:,0:Modes]
    S = S[0:Modes,0:Modes]
    V = V[0:Modes,:]

    print('SVD Complete\n')

    # AML Matrix construction
    AML = np.dot(S,V) 
    tensor = AML

    if output1 == 'yes':
        Mat_trunc = np.dot(U,AML)
        Ten_trunc = np.reshape(Mat_trunc,[nv,nx,ny,nt], order='F')
        
        if decision1 == 'no':
            pass
        else:
            for iter in range(data.shape[3]):
                variable = Ten_trunc[iter,:,:,:]
                Ten_trunc[iter,:,:,:] = variable*(Factor[iter])+Media[iter]+Media_tiempo[iter,:,:,:]
                
        if output1 == 'yes':  
            RRMSE_SVD = np.zeros(data.shape[3])
            for iter in range(data.shape[3]):
                diff = Ten_trunc[iter,:,:,:] - Tensor_orig[iter,:,:,:]
                RRMSE_SVD[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Tensor_orig[iter,:,:,:].flatten(),(-1,1)),ord=2)
                print(f'SVD RRMSE error for variable {iter+1}: {np.round(RRMSE_SVD[iter]*100, 3)}%')
            
            if not output0 or output0.strip().lower() in ['.npy', 'npy']:
                np.save(f'{path0}/{filen}/RRMSE_SVD.npy',RRMSE_SVD)

            elif output0.strip().lower() in ['.mat', 'mat']:
                mdic = {"RRMSE_SVD": RRMSE_SVD}
                file_mat= str(f'{path0}/{filen}/RRMSE_SVD.mat')
                hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

    # Scaling of temporal coeficients
    if decision2 == 'no':
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

    print('\n-----------------------------')
    print('Dataset configuration: \n')
    print('total_length: ', total_length)
    print('channels_n: ', channels_n)
    print('dim_x: ', dim_x)
    print('dim_y: ', dim_y)

    # Preparing training and test datasets
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

    print('\n-----------------------------')
    print('Model training summary: \n')
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
        Ytest_fl[r] = np.copy(np.reshape(Ytest[r], (test_n_adjusted, -1)))

    if hyper == 'No':
        def create_model_cnn(in_shape,  out_dim, p = 3, shared_dim = 1000, act_fun= act_func1):
            x = Input(shape=in_shape)
            
            v = Convolution1D(30,3)(x)
            v = Convolution1D(60,3)(v)
            v = Dense(neurons, activation= act_fun)(v)

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

        # Create the model
        in_shape = [k, dim_x]
        out_dim = dim_x 

        model= create_model_cnn(in_shape,out_dim,p,shared_dim) 

        print('\nModel Summary:\n')
        model.summary()

                    # save the best weights 
        save_string = 'hybrid_CNN_model'

        # save the best weights 
        save_best_weights = f'{path0}/{filen}/' + save_string + '.h5'
        save_summary_stats = f'{path0}/{filen}/' + save_string + '.csv'
        save_last_weights = f'{path0}/{filen}/' + save_string + '_last_w.h5'
        save_results_metrics = f'{path0}/{filen}/' + save_string + '_results_metrics.csv'

        # Model training
        np.random.seed(247531338)

        t0 = time.time()
        # Model training
        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', min_delta = 0.0001)]

        print('\nTraining Model Please Wait...\n')
        history = model.fit(training_generator,
                                    validation_data=validation_generator,
                                    epochs=epoch,
                                    verbose=1,
                                    callbacks=callbacks)
        t1 = time.time()
        print('\nModel Trained Successfully!')
        print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

    if hyper == 'Yes':
        import keras_tuner as kt
        def create_model_hp(hp):
            hp_activation = hp.Choice('hidden_layer_activation_function', values = ['relu', 'tanh', 'linear', 'sigmoid'])
            hp_neurons = hp.Int('neurons', min_value = 10, max_value = 100, step = 10)
            hp_activation_1 = hp.Choice('output_layer_activation_function', values = ['relu', 'tanh', 'linear', 'sigmoid'])
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4])
            hp_shared_dims = hp.Int('shared dims', min_value = 10, max_value = 100, step = 10)

            x = Input(shape=[k, dim_x])
                    
            v = Convolution1D(30,3)(x)
            v = Convolution1D(60,3)(v)
            v = Dense(hp_neurons, activation= hp_activation)(v)
            tt = [1]*p
            
            r = TimeDistributed(Dense(hp_shared_dims, activation=hp_activation))(v)
            s = tf.split(r, tt, 1)
            for i in range(p):
                s[i] = Flatten()(s[i])

            o = []
            for i in range(p):
                o.append(Dense(dim_x , activation=hp_activation_1)(s[i]))

            m = Model(inputs=x, outputs=o)
            
            opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
            
            m.compile(loss='mse', optimizer=opt, metrics=['mse'])
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
                    epochs=10,
                    verbose=1, 
                    callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        print('\n-----------------------------')
        print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {best_hps.get('hidden_layer_activation_function')}
Output Layer activation function: {best_hps.get('output_layer_activation_function')}
Number of neurons: {best_hps.get('neurons')}
Number of shared dimensions: {best_hps.get('shared dims')}
Learning rate: {best_hps.get('learning_rate')}
Loss function: 'mse'
            ''')

        model = tuner.hypermodel.build(best_hps)

        print('Model Summary:\n')
        model.summary()
        
        t0 = time.time()

        save_string = 'hybrid_CNN_model'

        save_best_weights = f'{path0}/{filen}/' + save_string + '.h5'
        save_summary_stats = f'{path0}/{filen}/' + save_string + '.csv'
        save_last_weights = f'{path0}/{filen}/' + save_string + '_last_w.h5'
        save_results_metrics = f'{path0}/{filen}/' + save_string + '_results_metrics.csv'

        callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_delta = 0.0001)]

        print('\nTraining Model Please Wait...\n')

        history = model.fit(training_generator,
                            validation_data=validation_generator,
                            epochs=epoch,
                            verbose=1,
                            callbacks=callbacks)

        t1 = time.time()
        print('\nModel Trained Successfully!')

        print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")


    print('Please CLOSE all figures to continue the run\n')

    # save the last weights 
    model.save_weights(save_last_weights)


    # Aggregate the summary statistics
    summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                'train_loss': history.history['loss'],
                                'valid_loss': history.history['val_loss']})

    summary_stats.to_csv(save_summary_stats)    
        
    if output3 == 'yes':
        fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Loss function evolution')

        ax.plot(summary_stats.train_loss, 'b',linewidth = 3) # blue
        ax.plot(summary_stats.valid_loss, 'g--',linewidth  = 3) # green
        ax.set_xlabel('Training epochs',fontsize = 20)
        ax.set_ylabel('Loss function',fontsize = 20)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/Lossfun_training.png')
        plt.show()
        plt.close()


    # Find the min validation loss during the training
    min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
    print('Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.6f}'.format(min_loss))
    min_loss = round(min_loss, 4)

    # Inference
    t0 = time.time()

    model.load_weights(save_best_weights)
    print('\nModel predicting. Please wait\n')
    Ytest_hat_fl = model.predict(test_generator, verbose=1) 
        
    t1 = time.time()
    print("\nPrediction complete. Time elapsed: %f" % ((t1 - t0) / 60.))

    print('\nPerformance measures on Test data, per sec')
    lag = 0
    num_sec = Ytest_hat_fl[0].shape[0]
    results_table = pd.DataFrame(index=['MSE','MAE','MAD','R2','SMAPE','RRMSE'],columns=range(num_sec))
    for i in range(num_sec):
        results_table.iloc[0,i] = mean_squared_error( Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[1,i] = mean_absolute_error(   Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[2,i] = median_absolute_error(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[3,i] = r2_score(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[4,i] = smape(  Ytest_fl[lag][i,:], Ytest_hat_fl[lag][i,:])
        results_table.iloc[5,i] = RRMSE( np.reshape(Ytest_fl[lag][i,:],(-1,1)), np.reshape(Ytest_hat_fl[lag][i,:],(-1,1)))
    print(results_table)
    results_table.to_csv(f'{path0}/{filen}/Results.csv', index = False, sep=',')

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
    results_table_global.to_csv(f'{path0}/{filen}/Global_results.csv', index = False, sep=',')

    print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
    print('Please CLOSE all figures to continue the run\n')


    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - MSE score')
    ax.plot(range(num_sec),results_table.iloc[0,:], 'b') # green
    ax.set_title("MSE score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("MSE",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/MSE_score.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - MAE score')
    ax.plot(range(num_sec),results_table.iloc[1,:], 'b') # green
    ax.set_title("MAE score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("MAE",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/MAE_score.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - MAD score')
    ax.plot(range(num_sec),results_table.iloc[2,:], 'b') # green
    ax.set_title("MAD score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("MAD",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/MAD_score.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - R2 score')
    ax.plot(range(num_sec),results_table.iloc[3,:], 'b') # green
    ax.set_title("R2 score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("R2",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/R2_score.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - SMAPE score')
    ax.plot(range(num_sec),results_table.iloc[4,:], 'b') # green
    ax.set_title("SMAPE score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("SMAPE",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/SMAPE_score.png')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - RRMSE score')
    ax.plot(range(num_sec),results_table.iloc[5,:], 'b') # green
    ax.set_title("RRMSE score vs. forecast time index",fontsize=18)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("RRMSE",fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/RRMSE_score.png')
    plt.show()
    plt.close()

    # Fill the output arrays
    Ytest_hat_fl = model.predict(test_generator, verbose=1) 

    # Create the multidimensional arrays for the results and ground-truth values
    mat_pred = np.zeros((Ytest_fl[0].shape[0],p,Ytest_fl[0].shape[1]))


    # Fill the output arrays
    for i in range(p):
        for j in range(Ytest_fl[0].shape[0]):
            mat_pred[j,i,:]=Ytest_hat_fl[i][j,:] 

    if decision2 == 'no':
        pass
    elif decision2 == 'auto':
        mat_pred = mat_pred * std_val + med_val
    elif decision2 == 'range':
        mat_pred = mat_pred * range_val + min_val
    elif decision2 == 'MaxPerMode':
        mat_pred = mat_pred * max_val
            
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

    # data reconstruction
    Ten_pre = np.reshape(Mat_pre, [nv,nx,ny,AML_pre_LSTM.shape[1]], order='F')

    for iter in range(data.shape[3]):
        variable = Ten_pre[iter,:,:,:]
        Ten_pre[iter,:,:,:] = variable*(Factor[iter])+Media[iter]+Media_tiempo[iter,:,:,:Ten_pre.shape[3]]
            
    print('\n')   
    # RRMSE Measure with Original data
    if output4 == 'yes':
        RRMSE_orNN = np.zeros(Tensor_orig.shape[0])
        for iter in range(Tensor_orig1.shape[0]):

            diff = Ten_pre[iter,:,:,:] - Tensor_orig[iter,:,:,time_lag:time_lag+num_snap]
            RRMSE_orNN[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Tensor_orig[iter,:,:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)

            print(f'RRMSE measure with original tensor - variable {iter+1}: {np.round(RRMSE_orNN[iter]*100, 3)}%')

        if not output0 or output0.strip().lower() in ['.npy', 'npy']:
            np.save(f"{path0}/{filen}/RRMSE_orNN.npy", RRMSE_orNN)
            
        elif output0.strip().lower() in ['.mat', 'mat']:
            mdic = {"RRMSE_orNN": RRMSE_orNN}
            file_mat= str(f"{path0}/{filen}/RRMSE_orNN.mat")
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')
    print()  
    # RRMSE with Truncated
    if output5 == 'yes':
        RRMSE_trNN = np.zeros(Tensor_orig.shape[0])
        for iter in range(Tensor_orig1.shape[0]):

            diff = Ten_pre[iter,:,:,:] - Ten_trunc[iter,:,:,time_lag:time_lag+num_snap]
            RRMSE_trNN[iter] = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(Ten_trunc[iter,:,:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)

            print(f'RRMSE measure with truncated tensor - variable {iter+1}: {np.round(RRMSE_trNN[iter]*100, 3)}%')
        
        if not output0 or output0.strip().lower() in ['.npy', 'npy']:
            np.save(f"{path0}/{filen}/RRMSE_trNN.npy", RRMSE_trNN)
            
        elif output0.strip().lower() in ['.mat', 'mat']:
            mdic = {"RRMSE_trNN": RRMSE_trNN}
            file_mat= str(f"{path0}/{filen}/RRMSE_trNN.mat")
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')
 
    # RRMSE Measure
    if output6 == 'yes':
        diff = AML_pre_LSTM - AML[:,time_lag:time_lag+num_snap]
        globalRRMSE = LA.norm(np.reshape(diff.flatten(),(-1,1)),ord=2)/LA.norm(np.reshape(AML[:,time_lag:time_lag+num_snap].flatten(),(-1,1)),ord=2)
        print(f'\nGlobal RRMSE measure - {np.round(globalRRMSE*100, 3)}%')

        rrmse_arr = np.zeros(num_snap)
        for i in range(num_snap):
            diff = AML[:,time_lag+i]-AML_pre_LSTM[:,i]
            rrmse_arr[i] = LA.norm(diff.flatten(),2)/LA.norm(np.transpose(AML[:,i]).flatten(),2)

        fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - RRMSE Measure')
        ax.plot(rrmse_arr,'k*-')
        ax.set_xlabel('Time')
        ax.set_ylabel('RRMSE')
        ax.set_title('RRMSE measure')
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/rrmse_measure.png')
        plt.show()
        plt.close()

    # AML Comparison
    if output7 == 'yes':
        # AML Comparation
        if not os.path.exists(f'{path0}/{filen}/Mode_Comparison'):
            os.mkdir(f"{path0}/{filen}/Mode_Comparison")

        AML_pre_1 = np.zeros([AML.shape[0],time_lag])
        AML_pre_1 = np.concatenate((AML_pre_1,AML_pre_LSTM),axis=1)

        for i in nModes:
            fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - AML vs AMLPRE per mode')
            ax.plot(AML[i-1,time_lag:],'k*-')
            ax.plot(AML_pre_1[i-1,time_lag:],'m*-')
            ax.set_xlabel("Time")
            ax.set_title(f"Comparation between AML and AMLPRE at mode {i}")
            plt.tight_layout()
            plt.savefig(f'{path0}/{filen}/Mode_Comparison/aMLComparation_m_{i}_test.png')
            plt.show() 
            plt.close()              

    # data Comparison
    if output8 == 'yes':
        if not os.path.exists(f'{path0}/{filen}/Snapshots'):
            os.mkdir(f"{path0}/{filen}/Snapshots")

        for i in range(index.shape[0]):
            namefig_orig1 = f'{path0}/{filen}/Snapshots/snap_comp_{index[i,0]+1}_t_{index[i,1]+1}.png'
            fig, ax = plt.subplots(num = f'CLOSE TO CONTINUE RUN - Snapshot comparison - snap_comp_{index[i,0]+1}_t_{index[i,1]+1}')

            ax.contourf(xv,yv,Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])],100,cmap='jet',
                        vmin = np.amin(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]),
                        vmax = np.amax(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]))
                
            ax.contourf(-xv,yv,Ten_pre[int(index[i,0]),:,:,int(index[i,1])],100,cmap='jet',
                        vmin = np.amin(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]),
                        vmax = np.amax(Tensor_orig[int(index[i,0]),:,:,time_lag+int(index[i,1])]))

            ax.set_title('Prediction vs. Original Data')
            ax = plt.gca()
            ax.set_aspect(1)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.annotate('', xy=(0.5, -0.005), xycoords='axes fraction', xytext=(0.5, 1.005),
                        arrowprops=dict(arrowstyle='-', lw = 3, color='k'))
            plt.savefig(namefig_orig1)
            plt.tight_layout()
            plt.show()

    if not output0 or output0.strip().lower() in ['.npy', 'npy']:
        np.save(f"{path0}/{filen}/TensorPred.npy", Ten_pre)
            
    elif output0.strip().lower() in ['.mat', 'mat']:
        mdic = {"Ten_pre": Ten_pre}
        file_mat= str(f"{path0}/{filen}/TensorPred.mat")
        hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

            




