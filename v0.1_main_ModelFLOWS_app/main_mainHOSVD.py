import hosvd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import data_load
import hdf5storage
import time
timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

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
        ax.contourf(Tensor[vel, ...,  i]) 
        ax.set_title(Title)

    interval = 2     
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval*1e+2, blit = False)

    plt.show()

print('\nHOSVD Algorithm')
print('\n-----------------------------')


################### Input ###################
#------------------------------------------------------------------------------------------------------------
print('Inputs:' + '\n')

# Detect current working directory:
path0 = os.getcwd()

while True:
    filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
    if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
        break
    else: 
        print('\tError: The selected input file format is not supported\n')

Tensor, _ = data_load.main(filetype)


## Number of snapshots SNAP:
while True:
    SNAP = input(f'Introduce number of snapshots. Continue with {Tensor.shape[-1]} snapshots: ')
    if not SNAP:
        SNAP = int(Tensor.shape[-1])
        break
    elif SNAP.isdigit():
        SNAP = int(SNAP)
        break
    else:
        print('\tError: Select a valid number (must be integer)\n')

## Tolerances:
while True:
    varepsilon1 = input('Introduce SVD tolerance. Continue with 1e-3: ')
    if not varepsilon1:
        varepsilon1 = 1e-3 
        break   # SVD
    elif is_float(varepsilon1):
        varepsilon1 = float(varepsilon1)
        break
    else:
        print('\tError: Please introduce a number\n')

#------------------------------------------------------------------------------------------------------
################### Output ###################

print('\n-----------------------------')
print('HOSVD summary:')
print('\n' + f'Number of snapshots set at: {SNAP}')
print(f'Tolerance set at {varepsilon1} for SVD')

print('\n-----------------------------')
print('Outputs:' + '\n')

filen = input('Enter folder name to save the outputs or continue with default folder name: ')
if not filen:
    filen = f'{timestr}_HOSVD_solution_tol_{varepsilon1}'
else:
    filen = f'{filen}'

while True:
    decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
    if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
        break
    else:
        print('\tError: Please select a valid output format\n')

print('')

TimePos = Tensor.ndim

if not os.path.exists(f'{path0}/{filen}'):
    os.mkdir(f"{path0}/{filen}")

Tensor0 = Tensor.copy()
shapeTens = list(np.shape(Tensor))
shapeTens[-1] = SNAP
Tensor = np.zeros(shapeTens)

Tensor[..., :] = Tensor0[..., 0:SNAP]

nn0 = np.array(Tensor.shape)
nn = np.array(nn0)
nn[1:np.size(nn)] = 0 

print('Performing HOSVD. Please wait...\n')
hatT, U, S, sv, nn1, n, TT = hosvd.HOSVD(Tensor, varepsilon1, nn, nn0, TimePos)
print('\nHOSVD complete!\n')

RRMSE = np.linalg.norm(np.reshape(Tensor-TT,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
print(f'Relative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')

print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
print('Please CLOSE all figures to continue the run\n')

markers = ['yo', 'gx', 'r*', 'bv', 'y+']
labels = ["Variable Singular Values",
            "X Space Singular Values",
            "Y Space Singular Values",
            "Z Space Singular Values",
            "Time Singular Values"]

fig, ax = plt.subplots(num = 'CLOSE TO CONTINUE RUN - Total Modes vs Retained Singular Values')
sub_axes = plt.axes([.6, .3, .25, .2]) 

if np.array(n).size == 4:
    labels.remove("Z Space Singular Values")

if np.array(n).size == 3:
    labels.remove("Variable Singular Values")
    labels.remove("Z Space Singular Values")

for i in range(np.array(n).size):
    ax.plot(sv[0, i] / sv[0, i][0], markers[i])
    sub_axes.plot(sv[0, i][:nn1[i]] / sv[0, i][0], markers[i])
    
ax.hlines(y=varepsilon1, xmin = 0, xmax=np.array(n).max(), linewidth=2, color='black', label = f'SVD tolerance: {varepsilon1}')   

ax.set_yscale('log')           # Logarithmic scale in y axis
sub_axes.set_yscale('log')
ax.set_xlabel('SVD modes')
    
ax.legend(labels, loc='best')
ax.set_title('Total Modes vs Retained Singular Values')
sub_axes.set_title('Retained Singular Values', fontsize = 8)
plt.savefig(f'{path0}/{filen}/SingularValues.png', bbox_inches='tight')
plt.show()
plt.close()

U = U[0].copy()

print('\nCalculating time modes. Please wait...')
if TimePos == 3:
    time_modes = np.einsum('ijl, ai, bj -> abl', S, U[0], U[1])
if TimePos == 4:
    time_modes = np.einsum('ijkl, ai, bj, ck -> abcl', S, U[0], U[1], U[2])
if TimePos == 5:
    time_modes = np.einsum('ijkml, ai, bj, ck, dm -> abcdl', S, U[0], U[1], U[2], U[3])
print('Time modes calculation complete!\n')

if not os.path.exists(f'{path0}/{filen}/time_modes'):
    os.mkdir(f'{path0}/{filen}/time_modes')

print(f'Saving first 3 time mode plots to {path0}/{filen}/time_modes')

if TimePos==3:
    for ModeNum in range(3):
        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Time mode')
        ax.contourf(time_modes[:,:,ModeNum])
        ax.set_title(f'Time modes - Mode Number {ModeNum+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.savefig(f'{path0}/{filen}/time_modes/ModeNum_{ModeNum+1}.png')
        plt.show()

if TimePos > 3:
    for ModComp in range(time_modes.shape[0]):
        for ModeNum in range(3):
            if TimePos==4:
                fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Time mode')
                ax.contourf(time_modes[ModComp,:,:,ModeNum])
                ax.set_title(f'Time modes - Component {ModComp+1} Mode Number {ModeNum+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.savefig(f'{path0}/{filen}/time_modes/TimeModeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                

            elif TimePos==5:
                nz = int(Tensor.shape[3] / 2)
                fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Time mode XY plane')
                ax.contourf(time_modes[ModComp,:,:,nz,ModeNum])
                ax.set_title(f'Time modes XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.savefig(f'{path0}/{filen}/time_modes/XY_TimeModeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                

    while True:
        while True:
            ModeNum = input(f'Introduce the mode number to plot (default mode 1). Maximum number of modes is {time_modes.shape[-1]}: ')
            if not ModeNum:
                ModeNum = 0
                break
            elif ModeNum.isdigit():
                if int(ModeNum) <= time_modes.shape[-1]:
                    ModeNum = int(ModeNum)-1  
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')
        while True:
            ModComp = input(f'Introduce the component to plot (default component 1). Maximum number of components is {time_modes.shape[0]}: ')
            if not ModComp:
                ModComp = 0
                break
            elif ModComp.isdigit():
                if int(ModComp) <= time_modes.shape[0]:
                    ModComp = int(ModComp)-1
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')

        if TimePos==4:
            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Time mode')
            ax.contourf(time_modes[ModComp,:,:,ModeNum])
            ax.set_title(f'Time modes - Component {ModComp+1} Mode Number {ModeNum+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        elif TimePos==5:
            nz = int(Tensor.shape[3] / 2)
            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Time mode XY plane')
            ax.contourf(time_modes[ModComp,:,:,nz,ModeNum])
            ax.set_title(f'Time modes XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        while True:
            Resp = input('Do you want to plot another mode? Yes or No (y/n). Continue with No: ')
            if not Resp or Resp.strip().lower() in ['n', 'no']:
                Resp = 0
                break
            elif Resp.strip().lower() in ['y', 'yes']:
                Resp = 1
                break
            else:
                print('\tError: Select yes or no (y/n)\n')
        
        if Resp == 0:
            break

        if Resp == 1:
            continue

if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
    np.save(f'{path0}/{filen}/Reconstruction.npy', TT)
    np.save(f'{path0}/{filen}/hatT.npy', hatT)
    np.save(f'{path0}/{filen}/time_modes.npy', time_modes)

if decision_2.strip().lower() in ['.mat', 'mat']:
    mdic0 = {"Reconst": TT}
    mdic1 = {"hatT": hatT}
    mdic2 = {"Time_modes": time_modes}

    file_mat0 = str(f'{path0}/{filen}/Reconstruction.mat')
    file_mat1 = str(f'{path0}/{filen}/hatT.mat')
    file_mat2 = str(f'{path0}/{filen}/time_modes.mat')

    hdf5storage.savemat(file_mat0, mdic0, appendmat=True, format='7.3')
    hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')
    hdf5storage.savemat(file_mat2, mdic2, appendmat=True, format='7.3')

if TimePos <= 3:
    exit()

elif TimePos > 3:
    while True:   
        dec4 = input(f'Plot video of original data and reconstructed data? (y/n). Continue with Yes: ')
        if not dec4 or dec4.strip().lower() in ['y', 'yes']:
            decision4 = True
            break
        if dec4.strip().lower() in ['n', 'no']:
            decision4 = False
            exit()
        else:
            print('\tError: Please select yes or no (y/n)\n')

    if TimePos == 5:
        nz = int(Tensor.shape[3] / 2)
        while True:
            plane = input('Select a plane (XY, XZ, YZ)')
            if plane.strip().lower() == 'xy':
                Tensor = Tensor[:, :, :, nz, :]
                TT = TT[:, :, :, nz, :]
                break
            elif plane.strip().lower() == 'xz':
                Tensor = Tensor[:, :, 0, :, :]
                TT = TT[:, :, 0, :, :]
                break
            elif plane.strip().lower() == 'yz':
                Tensor = Tensor[:, 0, :, :, :]
                TT = TT[:, 0, :, :, :]
                break
            else:
                print('\tError: Select a valid plane\n')

    else:
        pass

    titles = []
    [titles.append(f'Component {i+1}') for i in range(Tensor.shape[0])]

    while True:
        if decision4 == True:
            vidvel = input(f'Select a component (max is {Tensor.shape[0]}). Continue with component 1: ')
            if not vidvel:
                vel = 0
                video(Tensor, vel, Title = f'Original Data - {titles[vel]}')
                video(TT, vel, Title = f'Reconstructed data - {titles[vel]}')
                break
            elif vidvel.isdigit():
                if int(vidvel) <= Tensor.shape[0]:
                    vel = int(vidvel) - 1
                    video(Tensor, vel, Title = f'Original Data - {titles[vel]}')
                    video(TT, vel, Title = f'Reconstructed data - {titles[vel]}')
                    break
                else:
                    print("\tError: Select a valid component\n")
            else:
                print('\tError: Introduce a valid format (must be integer)\n')

    while True:
        ch1 = input('Would you like to plot another component? (y/n). Continue with No: ')
        if ch1.strip().lower() in ['y', 'yes']:
            while True:
                vidvel = input(f'Select a component (max is {Tensor.shape[0]}): ')
                if vidvel.isdigit():
                    if int(vidvel) <= Tensor.shape[0]:
                        vel = int(vidvel) - 1
                        video(Tensor, vel, Title = f'Original Data - {titles[vel]}')
                        video(TT, vel, Title = f'Reconstructed data - {titles[vel]}')
                        break
                    else:
                        print("\tError: Select a valid component\n")
                else:
                    print('\tError: Introduce a valid format (must be integer)\n')
            continue
        elif not ch1 or ch1.strip().lower() in ['n', 'no']:
            exit()

        else:
            print('\tError: Select yes or no (y/n)\n')

