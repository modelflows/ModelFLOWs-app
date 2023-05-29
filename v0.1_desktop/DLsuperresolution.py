def DNNreconstruct():   
    import numpy as np
    import data_load
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
    import hdf5storage
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from math import floor


    timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

    tf.keras.backend.set_floatx('float64') 

    pd.set_option('display.max_columns',100) 
    pd.set_option('display.max_rows',100)

    path0 = os.getcwd()

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def downsampling():
        while True:
            filetype = input('Select the downsampled input file format (.mat, .npy, .csv, .pkl, .h5): ')
            print('\n\tWarning: This model can only be trained with 2-Dimensional or 3-Dimensional data (as in: (variables, nx, ny, time) or (variables, nx, ny, nz, time))\n')
            if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
                break
            else: 
                print('\tError: The selected input file format is not supported\n')

        Tensor, _ = data_load.main(filetype)

        dim = Tensor.ndim
        
        if dim == 3:
            var_sel=Tensor
            ny_sel=Tensor.shape[0]
            nx_sel=Tensor.shape[1]
        elif dim >= 4:
            if dim == 4:
                if Tensor.shape[2] < Tensor.shape[1]:
                    Tensor = np.transpose(Tensor,(3,1,2,0))
                else: 
                    Tensor = np.transpose(Tensor,(3,2,1,0))

            if dim == 5:
                Tensor = np.transpose(Tensor,(4,1,2,3,0))
            
            var_sel=Tensor
            ny_sel=Tensor.shape[1]
            nx_sel=Tensor.shape[2]
        
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
        print('min_val/max_val=',min_val, max_val)
        print('med_val=',med_val)
        print('range_val=',range_val)
        print('std_val=',std_val)
        print(np.quantile(tensor.flatten(), np.arange(0,1,0.1)))
        tensor_norm = XUds / max_val_XUds
        print(np.quantile(tensor_norm.flatten(), np.arange(0,1,0.1)))
        print(scipy.stats.describe(tensor_norm.flatten()))
        
        return tensor_norm

    def mean_absolute_percentage_error(y_true, y_pred): 
        epsilon = 1e-10 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

    def smape(A, F):
        return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

    def RRMSE (real, predicted):
        RRMSE = np.linalg.norm(np.reshape(real-predicted,newshape=(np.size(real),1)),ord=2)/np.linalg.norm(np.reshape(real,newshape=(np.size(real),1)))
        return RRMSE

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
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5), num=f'CLOSE TO CONTINUE RUN - Snapshot comparison')
        fig.tight_layout()

        im1 = ax[0].contourf(yg,xg,np.transpose(Xr_test[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[0].set_ylim(bottom=0,top=16)
        ax[0].set_axis_off()
        ax[0].set_title('Real Data')

        im2 = ax[1].contourf(y,x,np.transpose(Xr_sel[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[1].set_ylim(bottom=0,top=16)
        ax[1].set_axis_off()
        ax[1].set_title('Initial Data')

        im3 = ax[2].contourf(yg,xg,np.transpose(Xr_hat[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        #ax[2].set_ylim(bottom=0,top=16)
        ax[2].set_axis_off()
        ax[2].set_title('Reconstruction')
            
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.09, 0.01, 0.85])
        cbar = fig.colorbar(im1, cax = cbar_ax)
        cbar.ax.tick_params(labelsize=24)
        plt.savefig(figname)  
        plt.show()


    def figures_results(n_snap, var, yg, xg, y, x, Xr_test, Xr_sel, Xr_hat, figname):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 5))
        fig.tight_layout()

        im1 = ax[0].contourf(yg,xg,np.transpose(Xr_test[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[n_snap,...,var],(1,0))))
        ax[0].set_title('Real Data')
        ax[0].set_axis_off()

        im2 = ax[1].contourf(y,x,np.transpose(Xr_sel[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_sel[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_sel[n_snap,...,var],(1,0))))
        ax[1].set_title('Initial Data')
        ax[1].set_axis_off()

        im3 = ax[2].contourf(yg,xg,np.transpose(Xr_hat[n_snap,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_hat[n_snap,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_hat[n_snap,...,var],(1,0))))
        ax[2].set_title('Reconstruction')
        ax[2].set_axis_off()
            
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.09, 0.01, 0.85])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=24)
        plt.savefig(figname)       
        plt.close(fig)   

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
            
            im2 = ax[1].contourf(y,x,np.transpose(Xr_sel[i,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[0,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[0,...,var],(1,0))))
            ax[1].set_axis_off()
            
            im3 = ax[2].contourf(yg,xg,np.transpose(Xr_hat[i,...,var],(1,0)),
                        vmin=np.min(np.transpose(Xr_test[0,...,var],(1,0))),
                        vmax=np.max(np.transpose(Xr_test[0,...,var],(1,0))))
            ax[2].set_axis_off()
                    
            figure.subplots_adjust(right=0.8, top=0.9)
            cbar_ax = figure.add_axes([0.85, 0.09, 0.01, 0.85])
            cbar = figure.colorbar(im1, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=20)


        anim = animation.FuncAnimation(figure, animation_function, frames=test_ind.size, interval=1000, blit=False)
        writergif = animation.PillowWriter(fps=5) 
        anim.save(videoname, writer=writergif)

        # Showing the video slows down the performance

    def custom_loss(y_actual,y_pred):
        import tensorflow as tf
        a = tf.norm((y_actual-y_pred),ord = 'euclidean')
        b = tf.norm(y_actual, ord='euclidean')
        custom_loss = a/b

        return custom_loss

    # Inputs
    print('\nDeep Learning Data Enhancement Model')
    print('------------------------')
    print('Inputs: \n')

    [Xr_sel, ny_sel, nx_sel]=downsampling() 
    print('\nDownsampled Data Summary:\n')
    print('Xr_sel: ', Xr_sel.shape) 
    print('ny_sel =', ny_sel) 
    print('nx_sel =', nx_sel)
    print()

    while True:
        filetype = input('Select the ground truth input file format (.mat, .npy, .csv, .pkl, .h5): ')
        if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
            break
        else: 
            print('\tError: The selected input file format is not supported\n')

    Tensor, _ = data_load.main(filetype)

    dim = Tensor.ndim

    if dim == 4:
        dims = 'Data2D'
        if Tensor.shape[2] < Tensor.shape[1]:
            Xr = np.transpose(Tensor,(3,1,2,0))
        else: 
            Xr = np.transpose(Tensor,(3,2,1,0))
        nt, ny, nx, nvar = Xr.shape

        print('\nData Summary:\n')
        print('Data shape: ',Xr.shape)
        print('nvar =',nvar,'ny =', ny, 'nx =', nx, 'nt =', nt)
        print()
        
    if dim == 5:
        dims = 'Data3D'
        Xr = np.transpose(Tensor,(4,1,2,3,0))
        nt, ny, nx, nz, nvar = Xr.shape

        print('\nData Summary:\n')
        print('Data shape: ',Xr.shape)
        print('nvar =',nvar,'ny =', ny, 'nx =', nx, 'nz =', nz, 'nt =', nt)
        print('\n')
  
    while True:
        decision1 = input('Apply a first scaling to the input data? (y/n). Continue with No: ')
        if not decision1 or decision1.strip().lower() in ['n', 'no']:
            decision1='no-scaling'
            break
        elif decision1.strip().lower() in ['y', 'yes']:
            decision1='scaling'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    if decision1 == 'scaling':
        while True:
            decision2 = input('Output scaling graphs? (y/n). Continue with No: ')
            if not decision2 or decision2.strip().lower() in ['n', 'no']:
                decision2 = 'no'
                break
            elif decision2.strip().lower() in ['y', 'yes']:
                decision2 = 'yes'
                break
            else:
                print('\tError: Select yes or no (y/n)\n')
    else:
        decision2 = 'no'

    while True:
        decision3 = input('Apply a second scaling to the input data? (y/n). Continue with No: ')
        if not decision3 or decision3.strip().lower() in ['n', 'no']:
            decision3='no'
            break
        elif decision3.strip().lower() in ['y', 'yes']:
            decision3='yes'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        decision4 = input('Shuffle train and test data? (y/n). Continue with No: ')
        if not decision4 or decision4.strip().lower() in ['n', 'no']:
            decision4='no-shuffle'
            break
        elif decision4.strip().lower() in ['y', 'yes']:
            decision4 = 'shuffle'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    print('\n-----------------------------')
    print('Model Configuration: \n')

    while True:
        hyper = input('Use optimal hyperparameters? (y/n). Continue with No: ')
        if not hyper or hyper.strip().lower() in ['n', 'no']:
            hyper = 'no'
            break
        elif hyper.strip().lower() in ['y', 'yes']:
             hyper='yes'
             break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
            bs = input('Select batch size (recommended power of 2). Continue with 16: ')
            if not bs:
                bs = 16
                break
            if bs.isdigit():
                bs = int(bs)
                break
            else:
                print('\tError: Select a valid number format (must be integer)\n')

    if hyper=='yes':
        pass

    elif hyper == 'no':
        
        while True:
            neurons = input('Select the number of neurons per layer. Continue with 100: ')
            if not neurons:
                neurons = 100
                break
            elif neurons.isdigit():
                neurons = int(neurons)
                break
            else:
                print('\tError: Select a valid number format (must be integer)\n')           
            
        while True:
            act_func = input('Select hidden layer activation function (relu, elu, softmax, sigmoid, tanh, linear). Continue with relu: ')
            if not act_func or act_func.strip().lower() == 'relu':
                act_func = 'relu'
                break
            elif act_func.strip().lower() == 'elu':
                act_func = 'elu'
                break
            elif act_func.strip().lower() == 'softmax':
                act_func = 'softmax'
                break
            elif act_func.strip().lower() == 'sigmoid':
                act_func = 'sigmoid'
                break
            elif act_func.strip().lower() == 'tanh':
                act_func = 'tanh'
                break
            elif act_func.strip().lower() == 'linear':
                act_func = 'linear'
                break
            else:
                print('\tError: Please select a valid option\n')
    
        while True:
            act_fun1 = input('Select output layer activation function (tanh, relu, elu, linear). Continue with relu: ')
            if not act_fun1 or act_fun1.strip().lower() == 'relu':
                act_fun1 = 'relu'
                break
            elif act_fun1.strip().lower() == 'tanh':
                act_fun1 = 'tanh'
                break
            elif act_fun1.strip().lower() == 'elu':
                act_fun1 = 'elu'
                break
            elif act_fun1.strip().lower() == 'linear':
                act_fun1 = 'linear'
                break
            else:
                print('\tError: Please select a valid option\n')

        while True:
            learn_rate = input('Select the model learning rate. Continue with 1e-3: ')
            if not learn_rate:
                learn_rate=0.001
                break
            elif is_float(learn_rate):
                learn_rate = float(learn_rate)
                break
            else:
                print('\tError: Please select a number\n')

        while True:
            lossf = input('Select a loss function ("mse", "custom"). Continue with mse: ')
            if not lossf or lossf.strip().lower() == 'mse':
                lossf = 'mse'
                break
            elif lossf.lower().strip() == 'custom':
                lossf='custom'
                break
            else:
                print('\tError: Select a valid option\n') 

        if lossf=='custom':
            loss_function=custom_loss
        if lossf=='mse':
            loss_function='mse'

        print('\n-----------------------------')
        print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {act_func}
Output Layer activation function: {act_fun1}
Batch size: {bs}
Number of neurons: {neurons}
Learning rate: {learn_rate}
Loss function: {lossf}
        ''')

    print('-----------------------------')
    print('Training configuration: \n')
    while True:
        n_train = input(f'Select the number of samples for the training data. Continue with {round(int(nt)*0.8)} samples (80% aprox. RECOMMENDED): ')
        if not n_train:
            n_train = round(int(nt)*0.8)
            break
        elif n_train.isdigit():
            n_train = int(n_train)
            break
        else:
            print('\tError: Select a valid number format (must be integer)')

    while True:
        epoch = input('Select the number of training epoch. Continue with 500: ')
        if not epoch:
            epoch = 500
            break
        elif epoch.isdigit():
            epoch = int(epoch)
            break
        else:
            print('\tError: Select a valid number format (must be integer)')

    print('\n-----------------------------')
    print('Outputs: \n')

    filen = input('Enter folder name to save the outputs or continue with default folder name: ')
    if not filen:
        filen = f'{timestr}_DL_superresolution'
    else:
        filen = f'{filen}'

    while True:
        decision5 = input('Would you like to save the results? (y/n). Continue with Yes: ')
        if not decision5 or decision5.strip().lower() in ['y', 'yes']:
            decision5='yes'
            break
        elif decision5.strip().lower() in ['n', 'no']:
            decision5 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')


    while True:
        decision7 = input('Plot data comparison between the reconstruction, initial data, and ground truth? (y/n). Continue with Yes: ')
        if not decision7 or decision7.strip().lower() in ['y', 'yes']:
            decision7='yes'
            break
        elif decision7.strip().lower() in ['n', 'no']:
            decision7 = 'no'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        decisionx = input('Create comparison videos of the test data versus the predicted data? (y/n). Continue with No: ')
        if not decisionx or decisionx.strip().lower() in ['n', 'no']:
            decisionx='no'
            break
        elif decisionx.strip().lower() in ['y', 'yes']:
            decisionx = 'yes'
            break
        else:
            print('\tError: Select yes or no (y/n)\n')

    while True:
        decision_20 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
        if not decision_20 or decision_20.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
            break
        else:
            print('\tError: Please select a valid output format\n')

    if not os.path.exists(f'{path0}/{filen}'):
        os.mkdir(f'{path0}/{filen}')

    filename1 = f'{decision1}_{decision4}'

    if not os.path.exists(f'{path0}/{filen}/{filename1}'):
        os.mkdir(f'{path0}/{filen}/{filename1}')

    filename = f'{filen}/{filename1}'

    if Xr.ndim==3: 
        nt=100
        Xr= np.repeat(Xr[np.newaxis,...],nt, axis=0) 
        Xr_sel= np.repeat(Xr_sel[np.newaxis,...],nt, axis=0) 
        print('Xr: ',Xr.shape) 
        print('Xr_sel: ',Xr_sel.shape)

    if dims=='Data3D':
        Xr_sel=np.reshape(Xr_sel,(nt,ny_sel,nx_sel*nz,nvar),order='F')
        Xr=np.reshape(Xr,(nt,ny,nx*nz,nvar),order='F')
        print('Xr_sel: ',Xr_sel.shape)

    # SCALING

    if decision1=='scaling': 
        print('Scaling Data')
        var_sel_min = []
        var_sel_max = []
        for i in range(nvar):
            var_sel_min.append(np.min(Xr_sel[...,i]))
            var_sel_max.append(np.max(Xr_sel[...,i]))
            print(f'var{i+1}_sel_min: {var_sel_min[i]}, var{i+1}_sel_max: {var_sel_max[i]}')
        
        var_sel_min = np.array(var_sel_min)
        var_sel_max = np.array(var_sel_max)
        
        var_sel_sc = []
        for i in range(nvar):
            var_sel_sc.append(np.array(scale_val(Xr_sel[...,i], var_sel_min[i], var_sel_max[i])))
            print(f'var{i+1}_sel_sc shape: {np.array(var_sel_sc[i]).shape}')

        Xr_sel_sc = np.stack(var_sel_sc, axis=3)
        print('var_sel_sc shape: ', Xr_sel_sc.shape)
        
        var_min = []
        var_max = []
        for i in range(nvar):
            var_min.append(np.min(Xr[...,i]))
            var_max.append(np.max(Xr[...,i]))
            print(f'var{i+1}_min: {var_min[i]}, var{i+1}_max: {var_max[i]}')
        
        var_min = np.array(var_min)
        var_max = np.array(var_max)

        var_sc = []
        for i in range(nvar):
            var_sc.append(scale_val(Xr[...,i], var_min[i], var_max[i]))
            print(f'var{i+1}_sel_sc shape: ',var_sc[i].shape)
        
        var_sc = np.array(var_sc)
        Xr_sc = np.stack(var_sc, axis=3)
        print('var_sel_sc shape: ', Xr_sc.shape)
        print('\nData Scaled')

    else:
        None

    print('\nPlease CLOSE ALL FIGURES to continue the run\n')

    if decision2 == 'yes':
        plt.subplots(num=f'CLOSE TO CONTINUE RUN')
        plt.title('Ground truth data distribution')
        plt.hist(Xr.flatten(), bins='auto')  
        plt.ylim(0,2e5)
        plt.xlim(-5,5)
        plt.tight_layout()
        plt.show()

        plt.subplots(num=f'CLOSE TO CONTINUE RUN')
        plt.title('Scaled downsampled data distribution')
        plt.hist(Xr_sc.flatten(), bins='auto')
        plt.tight_layout()
        plt.show()

    # SVD
    print('Performing SVD')
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

        print('\nSVD Summary:')
        print('XUds: ',XUds.shape)
        print('XSds: ',XSds.shape)
        print('XVds: ',XVds.shape)
        print('\n')

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
            
        print('\nSVD Summary:')
        print('XUds: ',XUds.shape)
        print('XSds: ',XSds.shape)
        print('XVds: ',XVds.shape)
        print('\n')

    print('SVD Complete\n')

    plt.subplots(num=f'CLOSE TO CONTINUE RUN')
    plt.hist(XUds.flatten(), bins='auto')
    plt.title('Left singular vectors data distribution') 
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.subplots(num=f'CLOSE TO CONTINUE RUN')
    plt.hist(XVds.flatten(), bins='auto') 
    plt.title('Right singular vectors data distribution')
    plt.tight_layout()
    plt.show()
    plt.close()

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
    print('\nTrain-test split summary: \n')
    print('Xr_train: ',Xr_train.shape)
    print('Xr_test: ',Xr_test.shape)
    print('XUds_train: ',XUds_train.shape)
    print('XUds_test: ',XUds_test.shape)
    print('XSds_train: ',XSds_train.shape)
    print('XSds_test: ',XSds_test.shape)
    print('XVds_train: ',XVds_train.shape)
    print('XVds_test: ',XVds_test.shape)
    print('\n')

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

    if hyper == 'no':

        def create_model_1(in_U_dim, in_S_dim, in_V_dim, out_dim):
            in_U = Input(shape=(*in_U_dim,),name='in_u')  
            in_S = Input(shape=(*in_S_dim,),name='in_s') 
            in_V = Input(shape=(*in_V_dim,),name='in_v') 
            
            u = Flatten(name='u_1')(in_U) 
            u = Dense (neurons, activation=act_func, name='u_2')(u)
            XUus = Dense(out_dim[0]*in_U_dim[1]*in_U_dim[2],activation=act_fun1,name='u_3')(u)
            
            v = Flatten(name='v_1')(in_V)
            v = Dense (neurons, activation=act_func, name='v_2')(v)
            XVus = Dense(in_V_dim[0]*out_dim[1]*in_V_dim[2],activation=act_fun1,name='v_3')(v)
            
            XUus_reshape = Reshape((out_dim[0],in_U_dim[1],in_U_dim[2]),name='reshape_u')(XUus)
            XVus_reshape = Reshape((in_V_dim[0],out_dim[1],in_V_dim[2]),name='reshape_v')(XVus)
            
            X_hat = tf.einsum('ijkl,iknl,inpl->ijpl', XUus_reshape, in_S, XVus_reshape)

            m = Model(inputs=[in_U,in_S,in_V],outputs= X_hat)
            # m_upsampled_matrices = Model(inputs=[in_U,in_S,in_V],outputs= [XUus_reshape,XVus_reshape]) 

            m.compile(loss=loss_function, optimizer=Adam(learn_rate), metrics=['mse'])
            
            # return(m, m_upsampled_matrices)
            return m

        # model, model_upsampled_matrices = create_model_1(in_U_dim, in_S_dim, in_V_dim, out_dim)
        model = create_model_1(in_U_dim, in_S_dim, in_V_dim, out_dim)

        print('Model Summary:\n')
        model.summary()

        #print('\nUnsampled Matrices Model Summary:\n')
        #model_upsampled_matrices.summary()

    if hyper == 'yes':
        import keras_tuner as kt
        def create_model_hp(hp):
            hp_activation = hp.Choice('hidden_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu', 'sigmoid'])
            hp_neurons = hp.Int('hp_neurons', min_value = 10, max_value = 100, step = 10)
            hp_activation_1 = hp.Choice('output_layer_activation_function', values = ['relu', 'linear', 'tanh', 'elu', 'sigmoid'])
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4])

            in_U = Input(shape=(*in_U_dim,),name='in_u')  
            in_S = Input(shape=(*in_S_dim,),name='in_s') 
            in_V = Input(shape=(*in_V_dim,),name='in_v') 
            
            u = Flatten(name='u_1')(in_U) 
            u = Dense (hp_neurons, activation=hp_activation, name='u_2')(u)
            XUus = Dense(out_dim[0]*in_U_dim[1]*in_U_dim[2],activation=hp_activation_1,name='u_3')(u)
            
            v = Flatten(name='v_1')(in_V)
            v = Dense (hp_neurons, activation=hp_activation, name='v_2')(v)
            XVus = Dense(in_V_dim[0]*out_dim[1]*in_V_dim[2],activation=hp_activation_1,name='v_3')(v)
            
            XUus_reshape = Reshape((out_dim[0],in_U_dim[1],in_U_dim[2]),name='reshape_u')(XUus)
            XVus_reshape = Reshape((in_V_dim[0],out_dim[1],in_V_dim[2]),name='reshape_v')(XVus)
            
            X_hat = tf.einsum('ijkl,iknl,inpl->ijpl', XUus_reshape, in_S, XVus_reshape)

            m = Model(inputs=[in_U,in_S,in_V],outputs= X_hat)

            m.compile(loss='mse', optimizer=Adam(hp_learning_rate), metrics=['mse'])
            
            return m
        
        tuner = kt.Hyperband(create_model_hp, objective = 'val_loss', max_epochs = 10, factor = 3, directory = 'dir_1', project_name = 'x', overwrite = True)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

        print('\nSearching for optimal hyperparameters...\n')

        tuner.search([XUds_train,XSds_train,XVds_train],
                    Xr_train,
                    batch_size=bs,
                    epochs=50,
                    validation_split=0.15,
                    verbose=1,
                    shuffle=True,
                    callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        print('\n-----------------------------')
        print(f'''
HYPERPARAMETERS SUMMARY:\n
Hidden Layer activation function: {best_hps.get('hidden_layer_activation_function')}
Output Layer activation function: {best_hps.get('output_layer_activation_function')}
Number of neurons: {best_hps.get('hp_neurons')}
Learning rate: {best_hps.get('learning_rate')}
Loss function: 'mse'
            ''')

        model = tuner.hypermodel.build(best_hps)

        print('Model Summary:\n')
        model.summary()

    # Model training

    print('\nTraining Model Please Wait...\n')

    t0 = time.time()
    callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True, mode='auto'),
                EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')]

    history = model.fit([XUds_train,XSds_train,XVds_train],Xr_train,
                        batch_size=bs,
                        epochs=epoch,
                        validation_split=0.15,
                        verbose=1,
                        shuffle=True,
                        initial_epoch = 0,
                        callbacks=callbacks)

    t1 = time.time()

    print('\nModel Trained Successfully!')
    print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

    # Training stats

    summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                'train_acc': history.history['mse'],
                                'valid_acc': history.history['val_mse'],
                                'train_loss': history.history['loss'],
                                'valid_loss': history.history['val_loss']})

    summary_stats.to_csv(save_summary_stats)

    print(f'\nATTENTION!: All plots will be saved to {path0}/{filename}\n') 
    print('Please CLOSE all figures to continue the run\n')

    plt.subplots(num=f'CLOSE TO CONTINUE RUN - Loss function evolution')
    plt.yscale("log")
    plt.title('Training vs. Validation loss')
    plt.plot(summary_stats.train_loss, 'b', label = 'Train loss') 
    plt.plot(summary_stats.valid_loss, 'g', label = 'Valid. loss') 
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.subplots(num=f'CLOSE TO CONTINUE RUN - Accuracy evolution')
    plt.title('Training vs. Validation accuracy')
    plt.plot(summary_stats.train_acc, 'b', label = 'Train accuracy')
    plt.plot(summary_stats.valid_acc, 'g', label = 'Valid. accuracy')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    plt.close()

    min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
    print('Minimum val_loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_loss))
    min_loss = round(min_loss, 4)

    # Load the best model epoch

    model.load_weights(save_best_weights)

    # Model prediction on training data

    print('\nModel predicting. Please wait\n')
    t0 = time.time()
    Xr_hat = model.predict([XUds_test, XSds_test, XVds_test]) 
        
    t1 = time.time()
    print(f"\nPrediction complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes\n")


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
        print('Xr_test: ',Xr_test.shape)
        print('Xr_hat: ',Xr_hat.shape)
        print('Xr_sel: ',Xr_sel.shape)
        print('\n')

    xg = np.linspace(0,nx,nx)
    yg = np.linspace(0,ny,ny)

    x = np.linspace(0,nx,nx_sel)
    y = np.linspace(0,ny,ny_sel)
    xx, yy = np.meshgrid(x, y)

    # Saving results
    if decision5 == 'yes':
        os.makedirs(f"{path0}/{filename}/figures")
        os.makedirs(f"{path0}/{filename}/tables")
        os.makedirs(f"{path0}/{filename}/videos")

    while True:
        while True:
            nt = input(f'Introduce the snapshot to plot (default is first predicted snapshot {n_train + 1}). Cannot be higher than {n_train + int(test_ind.size)}: ')
            if not nt:
                nt = 0
                break
            elif nt.isdigit():
                if int(nt) > n_train and int(nt) < n_train + int(test_ind.size):
                    nt = int(nt) - n_train - 1
                    break
                else:
                    print('\tError: Selected snapshot is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')
        while True:
            nv = input(f'Introduce the component to plot (default component 1). Maximum number of components is {Xr_test.shape[-1]}: ')
            if not nv:
                nv = 0
                break
            elif nv.isdigit():
                if int(nv) <= nvar:
                    nv = int(nv)-1
                    break
                else:
                    print('\tError: Selected component is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')

        if int(nv) > 0:
            var_num = int(nv) + 1
        else:
            var_num = 1
        
        if int(nt) > 0:
            snap_num = n_train + int(nt) + 1
        else:
            snap_num = n_train + 1
        
        if 'nz' in locals():
            n5 = int(Xr.shape[3] / 2)
            fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - Snapshot Comparison XY plane')
            fig.suptitle(f'XY plane - Component {var_num} Snapshot {snap_num}')
            ax[0].contourf(yg,xg,np.transpose(Xr_test[nt,..., n5, nv], (1,0)))
            ax[0].set_title('Real Data - XY Plane')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')

            ax[1].contourf(yg,xg,np.transpose(Xr_hat[nt,..., n5, nv], (1,0)))
            ax[1].set_title('Predicted Data - XY Plane')
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            plt.tight_layout()
            plt.show()
            plt.close()

        else:
            fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - Snapshot Comparison')
            fig.suptitle(f'Component  {var_num} Snapshot {snap_num}')
            ax[0].contourf(yg,xg,np.transpose(Xr_test[nt,...,nv], (1,0)))
            ax[0].set_title('Real Data')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')

            ax[1].contourf(yg,xg,np.transpose(Xr_hat[nt,...,nv], (1,0)))
            ax[1].set_title('Predicted Data')
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')
            plt.tight_layout()
            plt.show()
            plt.close()

        Resp = input('Do you want to plot other snapshots? Yes or No (y/n). Continue with No: ')
        if not Resp or Resp.strip().lower() in ['n', 'no']:
            Resp = 0
            break
        elif Resp.strip().lower() in ['y', 'yes']:
            Resp = 1
        else:
            print('\tError: Select yes or no (y/n)\n')

    if decision5 == 'yes':
        tabname1 = "tables/NN_table1"
        tabname2 = "tables/NN_table2"
        if 'nz' in locals():
            n5 = int(Xr.shape[3] / 2)
            Xr_test_p0=Xr_test[...,n5,:]
            Xr_hat_p0=Xr_hat[...,n5,:]
            Xr_sel_p0=Xr_sel[...,n5,:]

            [results_table, results_table2] = error_tables(Xr_test_p0, Xr_hat_p0, Xr_sel_p0,tabname1,tabname2, path0, filename, nvar)
        else:
            [results_table, results_table2] = error_tables(Xr_test, Xr_hat, Xr_sel, tabname1, tabname2, path0, filename, nvar)

    if decision5 == 'yes':
        print('\nPerformance measures on Test data, for all measures:\n')
        print(results_table)
        

        print('\nPerformance measures on Test data, for one specific layer of measures:\n')  
        print(results_table2)
        print('\n')

    if decision5 == 'yes':
        print(f'Saving first 3 snapshots comparison plots for each variable to: {path0}/{filename}/figures')
        for var in range(nvar):
            print(f'\nvariable: {var+1}')
            files3 = os.listdir(f'{path0}/{filename}/figures')
            os.makedirs(f"{path0}/{filename}/figures/var{var}",exist_ok=True)

            for n_snap in range(3):
                print(f'\tSnapshot number: {n_snap}')
                if 'nz' in locals():
                    n5 = int(Xr.shape[3] / 2)
                    figname = f"{path0}/{filename}/figures/var{var}/var{var}_snap{n_train+n_snap}_p{n5}.png"
                    Xr_test_sel=Xr_test[...,n5,:]
                    Xr_hat_sel=Xr_hat[...,n5,:]
                    Xr_sel_sel=Xr_sel[...,n5,:]
                    figures_results(n_snap, var, yg, xg, y, x, Xr_test_sel, Xr_sel_sel, Xr_hat_sel, figname)
                else:
                    figname = f"{path0}/{filename}/figures/var{var}/var{var}_{n_train+n_snap}.png"
                    figures_results(n_snap, var, yg, xg, y, x, Xr_test, Xr_sel, Xr_hat, figname)

    # Video creation

    if not decision_20 or decision_20.strip().lower() in ['npy', '.npy']:
        np.save(f'{path0}/{filename}/reconstruction.npy', Xr_hat)

    if decision_20.strip().lower() in ['.mat', 'mat']:
        mdic1 = {"reconstruction": Xr_hat}
        file_mat1 = str(f'{path0}/{filename}/reconstructiom.mat')
        hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')

    if decisionx == 'yes':
        for var in range(nvar):
            print(f'Generating video for variable {nvar}. Video will be saved to {path0}/{filename}/videos')
            if 'nz' in locals ():
                n5 = int(Xr.shape[3] / 2)
                videoname=f"{path0}/{filename}/videos/var{var}_p{n5}.gif"
                Xr_test_sel=Xr_test[...,n5,:]
                Xr_hat_sel=Xr_hat[...,n5,:]
                Xr_sel_sel=Xr_sel[...,n5,:]
                videos_results(videoname, var, n_train, test_ind, xg, yg, x, y, Xr_test_sel, Xr_hat_sel, Xr_sel_sel)
            else:
                videoname=f"{path0}/{filename}/videos/var{var}.gif"
                videos_results(videoname, var, n_train, test_ind, xg, yg, x, y, Xr_test, Xr_hat, Xr_sel)

    if decision7=='yes':
        print(f'\nPlots will be saved to {path0}/{filename}/ylim/')
        for var in range(nvar):
            os.makedirs(f"{path0}/{filename}/ylim", exist_ok=True)
            os.makedirs(f"{path0}/{filename}/ylim/var{var}", exist_ok=True)
            figname = f"{path0}/{filename}/ylim/var{var}/var{var}_scatter.png"
            files3 = os.listdir(f'{path0}/{filename}/ylim')
            
            os.makedirs(f"{path0}/{filename}/ylim/var{var}",exist_ok=True)
            
            for n_snap in range(0, 1):
                print(f'\nGenerating comparison plot for variable {var}')
                if 'nz' in locals():
                    n5 = int(Xr.shape[3] / 2)
                    figname = f"{path0}/{filename}/ylim/var{var}/var{var}_snap{n_train+n_snap}_p{n5}.png"
                    
                    Xr_test_sel=Xr_test[...,n5,:]
                    Xr_hat_sel=Xr_hat[...,n5,:]
                    Xr_sel_sel=Xr_sel[...,n5,:]
                    figures_results_ylim(n_snap, var, yg, xg, y, x, Xr_test_sel, Xr_sel_sel, Xr_hat_sel, figname)
                    
                else:
                    figname = f"{path0}/{filename}/ylim/var{var}/var{var}_{n_train+n_snap}.png"
                    figures_results_ylim(n_snap, var, yg, xg, y, x, Xr_test, Xr_sel, Xr_hat, figname)