import numpy as np
np.seterr(all='raise')

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import time
import hdf5storage

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)

plt.rcParams['font.size'] = 12

from numpy import linalg as LA

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

import matplotlib.animation as animation

import data_load

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def video(Tensor, vel, Title):
    nt = Tensor.shape[-1]

    if nt in range(200, 500):
        Tensor[..., ::5]

    elif nt > 500:
        Tensor[..., ::15]

    frames = Tensor.shape[-1]

    fig, ax = plt.subplots(figsize = (8, 4), num = f'CLOSE TO CONTINUE RUN - {Title}')
    fig.tight_layout()

    def animate(i):
        ax.clear()
        ax.contourf(Tensor[vel, :, :, i]) 
        ax.set_title(Title)

    interval = 2     
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval*1e+2, blit = False)

    plt.show()

# Error functions
def mean_absolute_percentage_error(y_true, y_pred): 
    epsilon = 1e-10 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(epsilon,np.abs(y_true)))) * 100

def smape(A, F):
    return ((100.0/len(A)) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))+ np.finfo(float).eps))

# Relative root mean square error
def RRMSE (real, predicted):
    RRMSE = np.linalg.norm(np.reshape(real-predicted,newshape=(np.size(real),1)),ord=2)/np.linalg.norm(np.reshape(real,newshape=(np.size(real),1)))
    return RRMSE

################## INPUTS #################

print('\nDeep Learning Autoencoders model')
print('\n-----------------------------')
print('Inputs:\n')

while True:
    filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
    print('\n\tWarning: This model can only be trained with 2-Dimensional data (as in: (variables, nx, ny, time))\n')
    if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
        break
    else: 
        print('\tError: The selected input file format is not supported\n')

# Load data
path0 = os.getcwd()
timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

Tensor, _ = data_load.main(filetype)

tensor = Tensor # Select only the first two components
nvar, ny, nx, nt = tensor.shape

# AEs parameters 
while True:
    hyp_batch = input('Select batch size (recommended power of 2). Continue with 64: ')
    if not hyp_batch:
        hyp_batch = 64
        break
    elif  hyp_batch.isdigit():
        hyp_batch = int(hyp_batch)
        break
    else:
        print('\tError: Please introduce a number (must be integer)\n')

while True:
    hyp_epoch = input('Select training epochs. Continue with 100: ')
    if not hyp_epoch:
        hyp_epoch = 100
        break
    elif hyp_epoch.isdigit():
        hyp_epoch = int(hyp_epoch)
        break
    else:
        print('\tError: Please introduce a number (must be integer)\n')

while True:
    test_prop = input('Select test data percentage (0-1). (Recommended values >= 0.2). Continue with 0.20: ')
    if not test_prop:
        test_prop = 0.2
        break
    elif is_float(test_prop):
        test_prop = float(test_prop)
        break
    else:
        print('\tError: Please select a number\n')

while True:
    encoding_dim = input('Select autoencoder dimensions. Continue with 10: ')
    if not encoding_dim:
        encoding_dim = 10
        break
    elif encoding_dim.isdigit():
        encoding_dim = int(encoding_dim)
        break
    else:
        print('\tError: Please introduce a number (must be integer)\n')

print('\n-----------------------------')
print('Model Parameters summary:\n')
print(f'Batch Size: {hyp_batch}')
print(f'Training Epochs: {hyp_epoch}')
print(f'Test split: {test_prop}')
print(f'Autoencoder dimensions: {encoding_dim}')

print('\n-----------------------------')

## Decisions
print('Outputs: \n')

filen = input('Enter folder name to save the outputs or continue with default folder name: ')
if not filen:
    filen = f'{timestr}_AE_solution'
else:
    filen = f'{filen}'

# Create new folder:
if not os.path.exists(f'{path0}/{filen}'):
    os.mkdir(f'{path0}/{filen}')

# Save mat
folder_save = f'{path0}/{filen}'
file_savemat = "AE_output.mat"
    
while True: 
    dec1 = input('Would you like to plot the reconstruction vs the original data? (y/n). Continue with Yes: ')
    if not dec1 or dec1.strip().lower() in ['y', 'yes']:
        decision1 = True
        break
    if dec1.strip().lower() in ['n', 'no']:
        decision1 = False
        break
    else: 
        print('\tError: Select yes or no (y/n)\n')

while True: 
    dec2 = input(f'Would you like to plot the modes? (y/n). Continue with Yes: ')
    if not dec2 or dec2.strip().lower() in ['y', 'yes']:
        decision2 = True
        dec2_save = input(f'Would you like to save the mode plots? (y/n). Continue with No: ')
        if not dec2_save or dec2_save.strip().lower() in ['n', 'no']:
            decision2_save = False
            break
        elif dec2_save.strip().lower() in ['y', 'yes']:
            decision2_save = True
            break
        else:
            print('\tError: Select yes or no (y/n)\n')
    elif dec2.strip().lower() in ['n', 'no']:
        decision2 = False
        break
    else:
        print('\tError: Select yes or no (y/n)\n')
    

while True:
    dec3 = input(f'Would you like to save the output to a .mat file? (y/n). Continue with Yes: ')
    if not dec3 or dec3.strip().lower() in ['y', 'yes']:
        decision3 = True
        break
    elif dec3.strip().lower() in ['n', 'no']:
        decision3 = False
        break
    else: 
        print('\tError: Select yes or no (y/n)\n')

print('\n')

RedStep=1
tensor = tensor[:,0::RedStep,0::RedStep,0::RedStep]
ncomp, ny, nx, ntt = tensor.shape

min_val=np.array(2)

## scale between [0,1]
min_val=np.zeros(ncomp,); max_val=np.zeros(ncomp,); range_val=np.zeros(ncomp,); std_val=np.zeros(ncomp,)

tensor_norm=np.zeros(tensor.shape)

for j in range(ncomp):
    min_val   [j] = np.amin(tensor[j,:,:,:])
    max_val   [j] = np.amax(tensor[j,:,:,:])
    range_val [j] = np.ptp(tensor[j,:,:,:])
    std_val   [j] =np.std(tensor[j,:,:,:])
    tensor_norm[j,...] = (tensor[j,...]-min_val[j])/range_val[j]
    
# # ***AUTOENCODERS***
# 3. Perform the ANN
tf.random.set_seed(221) ## Remove this to experience the randomness!!!!
keras.backend.clear_session()
nxy2=ny*nx*ncomp
dim=nt

TT=tensor_norm.transpose((3,1,2,0))
ntt, ny, nx, ncomp = TT.shape
X_scale=np.reshape(TT,(dim,nxy2),order='F')
X_scale=X_scale.transpose((1,0))

input_img = Input(shape=(dim,))
encoded = Dense(encoding_dim, activation='linear')(input_img)
decoded = Dense(dim, activation='linear')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
decoder = Model(encoded, decoded)

# We compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
# Get a summary
print('Model summary\n')
autoencoder.summary()

# We do the splitting into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_scale, 
                                                    X_scale, 
                                                    test_size=test_prop) 

# CALLBACK : Early Stoping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta = 0.001)
print('\nTraining Model Please Wait...\n')
t0 = time.time()
History=autoencoder.fit(x_train, x_train,
                epochs=hyp_epoch,
                batch_size=hyp_batch,
                callbacks = [callback],
                shuffle=True,
                validation_data=(x_test, x_test))

t1 = time.time()

# get convergence history
loss_linlin=History.history['loss']
loss_v=History.history['val_loss']

print(f"\nTraining complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes\n")
# Prediction of the encoding/decoding

print('Model predicting. Please wait...\n')
t0 = time.time()

z=encoder.predict(X_scale)
x_tilde=autoencoder.predict(X_scale)

t1 = time.time()

print(f"\nPrediction complete. Time elapsed: {np.round(((t1 - t0) / 60.), 2)} minutes")

#Check error

Err=np.linalg.norm(x_tilde-X_scale)/np.linalg.norm(X_scale)
print('\nNeural Network RRMSE with all modes: '+str(np.round(Err*100, 3))+'%\n')

# ###### Plot RRMSE when reconstructing from mode 1 to nm

rrmse_ =np.zeros((z.shape[1],))
contrib=np.zeros((z.shape[1],))
for nm in range(0,encoding_dim):
    
    z1=np.zeros(z.shape)
    z1[:,0:nm] = z[:,0:nm]
    
    xx = decoder.predict(z1)
    rrmse = RRMSE(X_scale,xx)
    print('Adding mode: '+ str(nm+1) + ' - Updated Neural Network RRMSE: '+str(np.round(rrmse*100, 3))+'%\n')
    rrmse_[nm]=rrmse

print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
print('Please CLOSE all figures to continue the run\n')

fig, ax = plt.subplots(figsize=(6, 4), num = 'CLOSE TO CONTINUE RUN - Reconstruction RRMSE per added mode') # This creates the figure
plt.plot(np.arange(0,encoding_dim)+1,rrmse_*100)
plt.scatter(np.arange(0,encoding_dim)+1,rrmse_*100)
plt.title('Reconstruction RRMSE per added mode')
plt.xlabel('Mode number added')
plt.ylabel('RRMSE (%)')
fig.tight_layout()
plt.savefig(f'{path0}/{filen}/RRMSE_training.png')
plt.show()
plt.close()   


# ###### Plot the relative RRMSE when  eliminating each mode

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
    print('Eliminated mode: '+str(idel+1)+' - New Neural Network RRMSE: '+str(np.round(rrmse*100, 3))+'% - Neural Network RRMSE increase (compared to RRMSE with all modes): '+str(np.round(incr_rr*100, 3)) + '%\n')

fig, ax = plt.subplots(figsize=(6, 4), num = 'Neural Network RRMSE after eliminating mode "n"') # This creates the figure
plt.xlabel('Eliminated mode "n"')
plt.ylabel('RRMSE increase (%)')
plt.title('Neural Network RRMSE after eliminating mode "n"')
plt.plot(np.arange(1,encoding_dim+1),incr_rr_*100)
plt.scatter(np.arange(1,encoding_dim+1),incr_rr_*100)
fig.tight_layout()
plt.savefig(f'{path0}/{filen}/RRMSE_after_mode_elimination.png')
plt.show()
plt.close()

## indexes for the sorting

I = np.argsort(incr_rr_)
modes_sorted =  np.flip(I)

## modes 
z_sort = z[:,modes_sorted]

lim=int(dim/ncomp)
RR=x_tilde[:,0:dim:1]

ntt= x_tilde.shape[1]

RR = np.transpose(RR,(1,0))
ZZT=np.reshape(RR,(ntt,ny,nx,ncomp),order='F')
ZZT= np.transpose(ZZT,(3,1,2,0))


# ## Decisions
print('ATTENTION! The run will not continue until all figure windows are closed')
print('Recommendation: Close figures and check folder\n')

if not os.path.exists(f'{path0}/{filen}/plots'):
    os.mkdir(f'{path0}/{filen}/plots')

titles = []
[titles.append(f'Component {i+1}') for i in range(ncomp)]
titles_ = []
[titles_.append(f'comp_{i+1}') for i in range(ncomp)]

if decision1 == True:
    while True:
        while True:
            compp = input(f'Select component to plot (must be lower than {ncomp}). Continue with 1: ')
            if not compp :
                iv = 0
                break
            elif compp.isdigit():
                if int(compp) <= ZZT.shape[0]:
                    iv = int(compp) - 1
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')
        while True:
            timee = input(f'Select the snapshot to represent (must be lower than {ZZT.shape[-1]}). Continue with time snapshot 1: ')
            if not timee:
                it = 0
                break
            elif timee.isdigit():
                if int(timee) <= ZZT.shape[-1]:
                    it = int(timee) - 1
                    break
                else:
                    print(f'\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')

        
        # CONTOUR AUTOENCODER-- CHECK RECONSTRUCTION
        fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Real Data vs Reconstructed Data')
        fig.suptitle(f'Real vs Reconst. data: {titles[iv]} - snapshot: {it + 1}')
        ax[0].contourf(tensor[iv,:,:,it])
        ax[0].set_title(f'Real data')
        ax[1].contourf(ZZT[iv,:,:,it])
        ax[1].set_title(f'Reconstructed data')
        fig.tight_layout()
        plt.savefig(f'{path0}/{filen}/plots/RealReconst_data_snapshot_{it+1}_{titles_[iv]}.png')
        plt.show()
        plt.close()
        while True:
            Resp = input('Do you want to plot other figures? Yes or No (y/n). Continue with No: ')
            if not Resp or Resp.strip().lower() in ['n', 'no']:
                Resp = 0
                break
            elif Resp.strip().lower() in ['y', 'yes']:
                Resp = 1
                break
            else:
                print('\tError: Please select yes or no (y/n)\n')
                Resp = 1
        if Resp == 0:
            break

while True:
    opt = input(f'Select the number of snapshots to save. Continue with {int(ZZT.shape[-1])}: ')
    if not opt:
        n_time = int(ZZT.shape[-1])
        break
    elif opt.isdigit():
        if int(opt) <= int(ZZT.shape[-1]):
            n_time = int(opt)
            break
        else:
            print('\tError: Selected value is out of bounds\n')
    else:
        print('\tError: Select a valid number format (must be integer)\n')

if not opt or int(opt) != 0:
    if not os.path.exists(f'{path0}/{filen}/snapshots'):
        os.mkdir(f'{path0}/{filen}/snapshots')

    print('\n')
    print(f'Saving snapshots to {path0}/{filen}/snapshots')
    for iv in range(ZZT.shape[0]):
        print(f'\nComponent: {iv+1}')
        for it in range(n_time):
            print(f'\tSnapshot: {it+1}')
            fig, ax = plt.subplots()
            ax.contourf(ZZT[iv,:, :, it])
            ax.set_title(f'Reconstructed data - Component {iv+1} - snapshot {it+1}')
            fig.tight_layout()
            plt.savefig(f'{path0}/{filen}/snapshots/Reconst_data_snap_{it+1}_{titles_[iv]}.png')
            plt.close(fig)

    print('\nAll snapshot plots have been saved\n')

if not os.path.exists(f'{path0}/{filen}/autoencoder_dims'):
    os.mkdir(f'{path0}/{filen}/autoencoder_dims')     

if decision2 == True:
    while True:
        AEnum = input(f'Select number of modes to plot (max. is the number of encoder dimensions: {encoding_dim}). Continue with 2: ')
        if not AEnum:
            AEnum = 2
            break
        elif AEnum.isdigit():
            AEnum = int(AEnum)
            break
        else:
            print('\tError: Please introduce a valid number (must be integer)\n')
    for AEnum in range(min(AEnum,encoding_dim)):

        #SET AUTOENCODER TO PLOT
        MODE=np.transpose(z_sort[:,AEnum])

        AE=MODE[0:int(nx*ny*ncomp)]

        Rec_AE=np.reshape(AE,(ny,nx,ncomp),order='F')

        for comp in range(Rec_AE.shape[-1]):
            fig, ax = plt.subplots(figsize=(8, 4), num = f'CLOSE TO CONTINUE RUN - Mode {AEnum+1} - component {comp}')
            plt.contourf(Rec_AE[:,:,comp])
            plt.title(f'Mode {AEnum+1} - component {comp+1}')
            fig.tight_layout()
            if decision2_save ==True:
                plt.savefig(f'{path0}/{filen}/autoencoder_dims/mode_{AEnum+1}_comp_{comp}.png')
            plt.show()
            plt.close()
    
########## Save to .mat ##########
if decision3 == True:
    
    mdic = {"z": z,"z_sort": z_sort, "X_scale":X_scale, "nx":nx,"ny":ny,"ncomp":ncomp,"rrmse":rrmse_,"incr_rr":incr_rr_,"modes_sorted":modes_sorted}

    file_mat= str(f'{path0}/{filen}/' + file_savemat)

    hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

titles = []
[titles.append(f'Component {i+1}') for i in range(ZZT.shape[0])]

while True:
    dec4 = input(f'Plot video of original data and reconstructed data? (y/n). Continue with Yes: ')
    if not dec4 or dec4.strip().lower() in ['y', 'yes']:
        decision4 = True
        break
    elif dec4.strip().lower() in ['n', 'no']:
        decision4 = False
        exit()
    else:
        print('\tError: Select yes or no (y/n)\n')
        

if decision4 == True:
    while True:
        vidvel = input(f'Select a component (max. is {ncomp}). Continue with 1: ')
        if not vidvel or vidvel.strip().lower() == '1':
            vel = 0
            video(tensor, vel, Title = f'Original Data - {titles[vel]}')
            video(ZZT, vel, Title = f'Reconstructed data - {titles[vel]}')

        elif vidvel.isdigit():
            if int(vidvel) <= ncomp:
                vel = int(vidvel) - 1
                video(tensor, vel, Title = f'Original Data - {titles[vel]}')
                video(ZZT, vel, Title = f'Reconstructed data - {titles[vel]}')
            else:
                print('\tError: Selected component is out of bounds\n')
        else:
            print("\tError: Select a valid component\n")

        while True:
            resp = input('Would you like to plot another component? (y/n). Continue with Yes: ')
            if resp.strip().lower() in ['y', 'yes']:
                break
            elif resp.strip().lower() in ['n', 'no']:
                exit()
            else:
                print('\tError: Select yes or no (y/n)\n')
