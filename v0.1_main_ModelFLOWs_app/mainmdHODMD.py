import DMDd
import hosvd
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os               
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
        ax.contourf(Tensor[vel, :, :, i]) 
        ax.set_title(Title)

    interval = 2     
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval*1e+2, blit = False)

    plt.show()

print('\nMulti-dimensional HODMD' + '\n')
print('-----------------------------')

print('''
The Multi-dimensional HODMD algorithm has the following options:
1) Non-iterative Multi-dimensional HODMD
2) Iterative Multi-dimensional HODMD
''')

while True:
    typeHODMD = input('Select a type of Multi-dimensional HODMD (1/2): ')
    if typeHODMD.isdigit():
        if int(typeHODMD) == 1:
            print('\n\nNon-iterative Multi-dimensional HODMD')
            print('\n-----------------------------')
            n_iter = 1
            type = 'non-iter'
            break
        elif int(typeHODMD) == 2:
            print('\n\nIterative Multi-dimensional HODMD')
            print('\n-----------------------------')
            n_iter = 1000
            type = 'iter'
            break
        else:
            print('\tError: Select a valid option\n')
    else:
        print('\tError: Invalid value (must be integer)\n')
# Load data
path0 = os.getcwd()
print('Inputs:' + '\n')

while True:
    filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
    if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
        break
    else: 
        print('\tError: The selected input file format is not supported\n')

Tensor, _ = data_load.main(filetype)

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

## d Parameter:
print('''\nHODMD parameter (d) options:
Option 1: Select only 1 value for d. 
Option 2: Select more than one value for d.
    ''')

d_list = []
print(f'Interval of recommended number of HODMD windows (d): [{int(np.round(SNAP/10))}, {int(np.round(SNAP/2))}]. Other values are accepted')

while True:
    d_dec = input('Select an option (1/2). Continue with option 1: ')
    if not d_dec or d_dec == '1':
        d = input(f'Introduce number of HODMD windows (d): ')
        if d.isdigit():
            d = int(d)
            d_list.append(d)
            break
        else:
            print('\tError: Select a valid number (must be integer)\n')
    elif d_dec == '2':
        break
    else:
        print('\tError: Select a valid option\n')

if d_dec == '2':
    while True:
        d_list_len = input('Select the number of values for d. Continue with 2: ')
        if d_list_len.isdigit():
            d_list_len = int(d_list_len)
            for d in range(d_list_len):
                d_val = input(f'Introduce value number {d+1} for d: ')
                if d_val.isdigit():
                    d_list.append(int(d_val))
                else:
                    print('\tError: Invalid value (must be integer)\n')
            
            if all(isinstance(x, int) for x in d_list):
                print(f'Selected values: {d_list}\n')
                break
            else:
                print('\tError: One or more selected values are invalid\n')
        else:
            print('\tError: Select a valid number (must be integer)\n')
                    
# SVD Tolerance

print('''\nSVD tolerance options:
Option 1: Select only 1 value for SVD tolerance. 
Option 2: Select more than one SVD tolerance value.
    ''')

tol1_list = []

while True:
    tol_dec = input('Select an option (1/2). Continue with option 1: ')
    if not tol_dec or tol_dec == '1':
        varepsilon1 = input('Introduce SVD tolerance. Continue with 1e-3: ')
        if not varepsilon1:
            varepsilon1 = 1e-3 
            tol1_list.append(varepsilon1)
            break
        elif is_float(varepsilon1):
            varepsilon1 = float(varepsilon1)
            tol1_list.append(varepsilon1)
            break
        else:
            print('\tError:Please introduce a number\n')
    elif tol_dec == '2':
        break
    else:
        print('\tError: Select a valid option\n')

## Tolerance list
if tol_dec == '2':
    while True:
        tol1_list_len = input('Select the number of values for SVD tolerance. Continue with 2: ')
        if tol1_list_len.isdigit():
            tol1_list = []
            tol1_list_len = int(tol1_list_len)
            for tol in range(tol1_list_len):
                tol_val = input(f'Introduce SVD tolerance number {tol+1}: ')
                if is_float(tol_val):
                    tol1_list.append(float(tol_val))
                else:
                    print('\tError: Invalid value\n')
            
            if all(isinstance(x, float) for x in tol1_list):
                print(f'Selected values: {tol1_list}\n')
                break
            else:
                print('\tError: One or more selected values are invalid\n')
        else:
            print('\tError: Select a valid number (must be integer)\n')

tol2_list = []

if len(tol1_list) == 1:
    while True:
        vardec = input(f'HODMD tolerance value equal to SVD tolerance ({tol1_list[0]}) (y/n). Continue with Yes: ')
        if not vardec or vardec.strip().lower() in ['y', 'yes']:
            tol2_list = tol1_list
            break
        elif vardec.strip().lower() in ['n', 'no']:
            varepsilon2 = input('Introduce second tolerance (HODMD). Continue with 1e-3: ')
            if not varepsilon2:
                varepsilon2 = 1e-3
                tol2_list.append(varepsilon2) 
                break  
            elif is_float(varepsilon2):
                varepsilon2 = float(varepsilon2)
                tol2_list.append(varepsilon2)
                break
            else:
                print('\tError: Please introduce a number\n')
        else:
            print('\tError: Select yes or no (y/n)')

if len(tol1_list) > 1:
    tol2_list = tol1_list


##Time:
while True:
    deltaT = input('Introduce time step (deltaT). Continue with 1: ')
    if not deltaT:
        deltaT = 1
        break
    elif is_float(deltaT):
        deltaT = float(deltaT)
        break
    else:
        print('\tError: Please introduce a number\n')


Time = np.linspace(0,SNAP-1,num=SNAP)*deltaT
    
## Position of temporal variable:
TimePos = Tensor.ndim

#--------------------------------------------------------------------------------------------------------------------------------------
################### Output ###################

print('\n-----------------------------')
print('Iterative HODMD summary:')
print('\n' + f'Number of snapshots set at: {SNAP}')
print(f'd Parameter(s) set at: {d_list}')
print(f'SVD tolerance(s) {tol1_list}')
print(f'HODMD tolerance(s): {tol2_list}')
print(f'Time gradient set at deltaT: {deltaT}')

print('\n-----------------------------')
print('Outputs:\n')

filen = input('Enter folder name to save the outputs or continue with default folder name: ')
if not filen:
    filen = f'{timestr}_{type}_mdHODMD_solution'
else:
    filen = f'{filen}'

while True:
    decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
    if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
        break
    else:
        print('\tError: Please select a valid output format\n')

print('')

## Save results in folder:

Tensor0 = Tensor
shapeTens = list(np.shape(Tensor))
shapeTens[-1] = SNAP
Tensor = np.zeros(shapeTens)

Tensor[..., :] = Tensor0[..., 0:SNAP]
TensorR = Tensor.copy()

## ALGORITHM:

## ITERATIVE:
nn0 = np.shape(Tensor)
nn = np.array(nn0)
nn[1:np.size(nn)] = 0

if not os.path.exists(f'{path0}/{filen}'):
    os.mkdir(f"{path0}/{filen}")

Frequencies = []
Amplitudes = []
GrowthRates = []
d_val = []
tol_val = []
DMDmode_list = []
Tensor_rec_list = []

for d in d_list:
    for varepsilon1 in tol1_list:
        if len(tol1_list) > 1:
            varepsilon2 = varepsilon1
        elif not vardec or vardec.strip().lower() in ['y', 'yes']:
            varepsilon2 = varepsilon1
        if not os.path.exists(f'{path0}/{filen}/{d}_tol_{varepsilon1}'):
            os.mkdir(f"{path0}/{filen}/d_{d}_tol_{varepsilon1}")
        if not os.path.exists(f'{path0}/{filen}/{d}_tol_{varepsilon1}/DMDmodes'):
            os.mkdir(f"{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes")
        print(f'\nRunning HODMD for d = {d} and tol = {varepsilon1}')
        for zz in range(0,n_iter):
            if n_iter > 1:
                print(f'Iteration number: {zz+1}')
            if zz != 0:
                del S,U,Frequency,Amplitude,GrowthRate,hatT,hatMode,sv,TensorR,nnin
                TensorR = TensorReconst
            
            ## Perform HOSVD decomposition to calculate the reduced temporal matrix: hatT
            nnin = nn
            nnin = tuple(nnin)
            print('\nPerforming HOSVD. Please wait...\n')
            [hatT,U,S,sv,nn1,n,TT] = hosvd.HOSVD(TensorR,varepsilon1,nn,nn0,TimePos)
            print('\nHOSVD complete!\n')
            # hatT: Reduced temporal matrix
            
            ## Perform HODMD to the reduced temporal matrix hatT:
            print('Performing HODMD. Please wait...\n')
            [hatMode,Amplitude,Eigval,GrowthRate,Frequency] = DMDd.hodmd_IT(hatT,d,Time,varepsilon1,varepsilon2)
            print('\nHODMD complete!\n')

            ## Reconstruct the original Tensor using the DMD expansion:
            TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)

            RRMSE = np.linalg.norm(np.reshape(Tensor-TensorReconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
            print(f'Relative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')

            GRFreqAmp = np.zeros((np.size(GrowthRate),3))
            for ind in range (0,np.size(GrowthRate)):
                GRFreqAmp[ind,0] = GrowthRate[ind]
                GRFreqAmp[ind,1] = Frequency[ind]
                GRFreqAmp[ind,2] = Amplitude[ind]

            print('GrowthRate, Frequency and Amplitude:')
            print(GRFreqAmp)

            ## Break the loop when the number of singular values is the same in two consecutive iterations:

            num = 0
            for ii in range(1,np.size(nn1)):
                if nnin[ii]==nn1[ii]:
                    num = num+1
            
            if num==np.size(nn1)-1:
                break
            nn = nn1
            print('\n')

        if n_iter > 1:
            print(f'Final number of iterations for d = {d} and tol = {varepsilon1}: {zz+1}')
            print(f'\nRelative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')

        Frequencies.append(Frequency)
        Amplitudes.append(Amplitude)
        GrowthRates.append(GrowthRate)
        d_val.append(d)
        tol_val.append(varepsilon1)


        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/GRFreqAmp.npy', GRFreqAmp)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"GRFreqAmp": GRFreqAmp}
            file_mat = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/GRFreqAmp.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        # Tensor reconstruction
        TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)
        Tensor_rec_list.append(TensorReconst)

        ## Save the reconstruction of the tensor and the Growth rates, frequencies and amplitudes:

        # Reconstruction:
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/TensorReconst.npy', TensorReconst)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"TensorReconst": TensorReconst}
            file_mat = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/TensorReconst.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        ## Calculate DMD modes:
        print('Calculating DMD modes...')
        N = np.shape(hatT)[0]
        DMDmode = DMDd.modes_IT(N,hatMode,Amplitude,U,S,nn1,TimePos)
        DMDmode_list.append(DMDmode)

        # Save DMD modes:
        if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
            np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmode.npy',DMDmode)

        if decision_2.strip().lower() in ['.mat', 'mat']:
            mdic = {"DMDmode": DMDmode}
            file_mat = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmode.mat')
            hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

        print(f'\nSaving first 3 DMDmode plots to {path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes\n')

        if TimePos == 3:
            for ModeNum in range(3):
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'DMDmode - Mode Number {ModeNum+1}')
                ax[0].contourf(DMDmode[:,:,ModeNum].real)
                ax[0].set_title('Real part')
                ax[0].set_xlabel('X')
                ax[0].set_ylabel('Y')

                ax[1].contourf(DMDmode[:,:,ModeNum].imag)
                ax[1].set_title('Imaginary part')
                ax[1].set_xlabel('X')
                ax[1].set_ylabel('Y')
                plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes/ModeNum_{ModeNum+1}.png')
                plt.close(fig)

        for ModComp in range(DMDmode.shape[0]):
            for ModeNum in range(3):
                if TimePos==4:
                    fig, ax = plt.subplots(1, 2)
                    fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}')
                    ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
                    ax[0].set_title('Real part')
                    ax[0].set_xlabel('X')
                    ax[0].set_ylabel('Y')

                    ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
                    ax[1].set_title('Imaginary part')
                    ax[1].set_xlabel('X')
                    ax[1].set_ylabel('Y')
                    plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmodeComp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                    plt.close(fig)

                elif TimePos==5:
                    nz = int(Tensor.shape[3] / 2)
                    fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode XY plane')
                    fig.suptitle(f'DMDmode XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
                    ax[0].contourf(DMDmode[ModComp,:,:,nz,ModeNum].real)
                    ax[0].set_title('Real part - XY Plane')
                    ax[0].set_xlabel('X')
                    ax[0].set_ylabel('Y')

                    ax[1].contourf(DMDmode[ModComp,:,:,nz,ModeNum].imag)
                    ax[1].set_title('Imaginary part - XY Plane')
                    ax[1].set_xlabel('X')
                    ax[1].set_ylabel('Y')
                    plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmode_XY_Comp_{ModComp+1}_ModeNum_{ModeNum+1}.png')
                    plt.close(fig)
                else:
                    pass

# Frequency vs amplitudes comparison:
colored_markers = ['bo', 'kx', 'ro', 'cx', 'mo', 'yx', 'k^', 'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks', 'b^', 'g^', 'r^', 'c^', 'm^', 'y^']

total_cases = len(d_list) * len(tol1_list)

if total_cases > 1:

    plt.figure(num='CLOSE TO CONTINUE RUN - Frequency/Amplitude for all selected cases')
    for i in range(total_cases):
        plt.plot(Frequencies[i], Amplitudes[i], colored_markers[i], label = f'd = {d_val[i]} | tol = {tol_val[i]}', alpha = .5)
    plt.yscale('log')
    plt.xlabel('Frequency ($\omega_{m}$)')
    plt.ylabel('Amplitude ($a_{m}$)')
    plt.title(f'Frequency vs Amplitude for all selected cases')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/FrequencyAmplitude.png')
    plt.show()
    plt.close()

print(f'''
Run cases - Summary
-----------------------------
''')

i = 0
for d in d_list:
    for tol in tol1_list:    
        print(f'case {i+1}: d = {d} and tol = {tol}')
        i += 1


print('\n-----------------------------\n')

print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}\n') 
print('Please CLOSE all figures to continue the run\n')
## Result plots:
print('\nPlotting Frequency/GrowthRate and Frequency/Amplitude plots')
while True:
    while True:
        case = input(f'Select the case to plot (default case 1): ')
        if not case:
            case = 0
            break
        elif case.isdigit():
            if int(case) <= total_cases:
                case = int(case)-1  
                break
            else:
                print('\tError: Selected case does not exist\n')
        else:
            print('\tError: Select a valid number format (must be integer)\n')
    # Frequency vs absolute value of GrowthRate:
    plt.figure(num=f'CLOSE TO CONTINUE RUN - Case {case + 1} - Frequency/GrowthRate')
    plt.plot(Frequencies[case],np.abs(GrowthRates[case]), 'k+')
    plt.yscale('log')           # Logarithmic scale in y axis
    plt.xlabel('Frequency ($\omega_{m}$)')
    plt.ylabel('Absolute value of GrowthRate (|$\delta_{m}$|)')
    plt.title(f'd = {d_val[case]} tol = {tol_val[case]}')
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/d_{d_val[case]}_tol_{tol_val[case]}/FrequencyGrowthRate.png')
    plt.show()
    plt.close()

    # Frequency vs amplitudes:
    plt.figure(num=f'CLOSE TO CONTINUE RUN - Case {case + 1} - Frequency/Amplitude')
    plt.plot(Frequencies[case], Amplitudes[case],'r+')
    plt.yscale('log')           # Logarithmic scale in y axis
    plt.xlabel('Frequency ($\omega_{m}$)')
    plt.ylabel('Amplitude ($a_{m}$)')
    plt.title(f'd = {d_val[case]} tol = {tol_val[case]}')
    plt.tight_layout()
    plt.savefig(f'{path0}/{filen}/d_{d_val[case]}_tol_{tol_val[case]}/FrequencyAmplitude.png')
    plt.show()
    plt.close()

    while True:
        Resp = input('Do you want to plot another case? Yes or No (y/n). Continue with No: ')
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

print('\nPlotting component comparison')
dims = Tensor.ndim
while True:
    while True:
        case = input(f'Select the case to plot (default case 1): ')
        if not case:
            case = 0
            break
        elif case.isdigit():
            if int(case) <= total_cases:
                case = int(case)-1  
                break
            else:
                print('\tError: Selected case does not exist\n')
        else:
            print('\tError: Select a valid number format (must be integer)\n')
    if dims > 3:
        while True:
            c = input(f'Select a component (max is {Tensor.shape[0]}): ')
            if c.isdigit():
                if int(c) <= Tensor.shape[0]:
                    c = int(c) - 1
                    break
                else:
                    print("\tError: Selected component doesn't exist\n")
            elif not c:
                continue

    while True:
        x = input(f'Select X coordinate (must be in range [0, {Tensor.shape[2] - 1}]): ')
        if x.isdigit():
            if int(x) in range(0, Tensor.shape[2]):
                x = int(x)
                break
            else:
                print(Tensor.shape[2])
                print('\tError: Selected value is out of bounds\n')
        elif not x:
            continue

    while True:
        y = input(f'Select Y coordinate (must be in range [0, {Tensor.shape[1] - 1}]): ')
        if y.isdigit():
            if int(y) in range(0, Tensor.shape[1]):
                y = int(y)
                break
            else:
                print('\tError: Selected value is out of bounds\n')
        elif not y:
            continue
        else:
            print('\tError: Select a valid number format (must be integer)\n') 

    if dims == 3:
        fig, ax = plt.subplots(1, 2, num = f'CLOSE TO CONTINUE RUN - Case {case + 1} - component comparison')
        fig.suptitle(f'Real Data vs Reconstruction - Component comparison')
        ax[0].contourf(Tensor0[:, :, 0])
        ax[0].scatter(x, y, c='black', s=50)
        ax[1].plot(Time[:], Tensor_rec_list[case][y, x, :].real, 'k-x', label = 'Reconstructed Data')
        ax[1].plot(Time[:], Tensor[y, x, :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/d_{d_val[case]}_tol_{tol_val[case]}/OrigReconst.png')
        plt.show()
        plt.close()

    if dims == 4:
        fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - component comparison')
        fig.suptitle(f'Real Data vs Reconstruction - Case {case + 1} - Component comparison')
        ax[0].contourf(Tensor0[c, :, :, 0])
        ax[0].scatter(x, y, c='black', s=50)
        ax[1].plot(Time[:], Tensor_rec_list[case][c, y, x, :].real, 'k-x', label = 'Reconstructed Data')
        ax[1].plot(Time[:], Tensor[c, y, x, :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/d_{d_val[case]}_tol_{tol_val[case]}/OrigReconst.png')
        plt.show()
        plt.close()
        
    
    elif dims == 5:
        nz = int(Tensor.shape[3] / 2)
        fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - component comparison')
        fig.suptitle(f'Real Data vs Reconstruction - Case {case + 1} - Component comparison XY plane')
        ax[0].contourf(Tensor0[c, :, :, nz, 0])
        ax[0].scatter(x, y, c='black', s=50)
        ax[1].plot(Time[:], Tensor_rec_list[case][c, y, x, nz, :].real, 'k-x', label = 'Reconstructed Data')
        ax[1].plot(Time[:], Tensor[c, y, x, nz, :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/d_{d_val[case]}_tol_{tol_val[case]}/OrigReconst.png')
        plt.show()
        plt.close()
    
    while True:
        Resp = input('Do you want to plot another case? Yes or No (y/n). Continue with No: ')
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

print(f'Select a case, component and temporal mode to plot a DMD mode')
while True:
    while True:
        case = input(f'Select the case to plot (default case 1): ')
        if not case:
            case = 0
            break
        elif case.isdigit():
            if int(case) <= total_cases:
                case = int(case)-1  
                break
            else:
                print('\tError: Selected case does not exist\n')
        else:
            print('\tError: Select a valid number format (must be integer)\n')
    while True:
        ModeNum = input(f'Introduce the mode number to plot (default mode 1). Maximum number of modes is {DMDmode_list[case].shape[-1]}: ')
        if not ModeNum:
            ModeNum = 0
            break
        elif ModeNum.isdigit():
            if int(ModeNum) <=DMDmode_list[case].shape[-1]:
                ModeNum = int(ModeNum)-1  
                break
            else:
                print('\tError: Selected value is out of bounds\n')
        else:
            print('\tError: Select a valid number format (must be integer)\n')
    if TimePos > 3:
        while True:
            ModComp = input(f'Introduce the component to plot (default component 1). Maximum number of components is {DMDmode_list[case].shape[0]}: ')
            if not ModComp:
                ModComp = 0
                break
            elif ModComp.isdigit():
                if int(ModComp) <= DMDmode_list[case].shape[0]:
                    ModComp = int(ModComp)-1
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')

    if TimePos==3:
        fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode')
        fig.suptitle(f'Case {case + 1} - DMDmode - Mode Number {ModeNum+1}')
        ax[0].contourf(DMDmode_list[case][:,:,ModeNum].real)
        ax[0].set_title('Real part')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        ax[1].contourf(DMDmode_list[case][:,:,ModeNum].imag)
        ax[1].set_title('Imaginary part')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        plt.show()


    if TimePos==4:
        fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode')
        fig.suptitle(f'Case {case + 1} - DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}')
        ax[0].contourf(DMDmode_list[case][ModComp,:,:,ModeNum].real)
        ax[0].set_title('Real part')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        ax[1].contourf(DMDmode_list[case][ModComp,:,:,ModeNum].imag)
        ax[1].set_title('Imaginary part')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        plt.show()

    elif TimePos==5:
        nz = int(Tensor.shape[3] / 2)
        fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode XY plane')
        fig.suptitle(f'Case {case + 1} - DMDmode XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
        ax[0].contourf(DMDmode_list[case][ModComp,:,:,nz,ModeNum].real)
        ax[0].set_title('Real part - XY Plane')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        ax[1].contourf(DMDmode_list[case][ModComp,:,:,nz,ModeNum].imag)
        ax[1].set_title('Imaginary part - XY Plane')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
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

    if total_cases > 1:
        while True:
                case = input(f'Select the case to plot (default case 1): ')
                if not case:
                    case = 0
                    break
                elif case.isdigit():
                    if int(case) <= total_cases:
                        case = int(case)-1  
                        break
                    else:
                        print('\tError: Selected case does not exist\n')
                else:
                    print('\tError: Select a valid number format (must be integer)\n')
    else:
        case = 0

    if dims == 5:
        nz = int(Tensor.shape[3] / 2)
        while True:
            plane = input('Select a plane (XY, XZ, YZ)')
            if plane.strip().lower() == 'xy':
                Tensor = Tensor[:, :, :, nz, :]
                Tensor_rec_list[case] = Tensor_rec_list[case][:, :, :, nz, :]
                break
            elif plane.strip().lower() == 'xz':
                Tensor = Tensor[:, :, 0, :, :]
                Tensor_rec_list[case] = Tensor_rec_list[case][:, :, 0, :, :]
                break
            elif plane.strip().lower() == 'yz':
                Tensor = Tensor[:, 0, :, :, :]
                Tensor_rec_list[case] = Tensor_rec_list[case][:, 0, :, :, :]
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
                video(Tensor_rec_list[case], vel, Title = f'Reconstructed data - {titles[vel]}')
                break
            elif vidvel.isdigit():
                if int(vidvel) <= Tensor.shape[0]:
                    vel = int(vidvel) - 1
                    video(Tensor, vel, Title = f'Original Data - {titles[vel]}')
                    video(Tensor_rec_list[case], vel, Title = f'Reconstructed data - {titles[vel]}')
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
                        video(Tensor_rec_list[case], vel, Title = f'Reconstructed data - {titles[vel]}')
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

