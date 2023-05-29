import DMDd
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from math import floor
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import data_load
import hdf5storage
import time
timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

print('\nPredictive HODMD')

################### Input ###################
print('\n-----------------------------')
print('Inputs:' + '\n')

##Snapshot matrix Tensor:

path0 = os.getcwd()


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

while True:
    filetype = input('Select the input file format (.mat, .npy, .csv, .pkl, .h5): ')
    if filetype.strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5']:
        break
    else: 
        print('\tError: The selected input file format is not supported\n')

Tensor, _ = data_load.main(filetype)

##Number of snapshots SNAP:
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

##d Parameter:
while True:
    print(f'Interval of recommended number of HODMD windows (d): [{int(np.round(SNAP/10))}, {int(np.round(SNAP/2))}]. Other values are accepted')
    d = input(f'Introduce number of HODMD windows (d): ')
    if d.isdigit():
        d = int(d)
        break
    else:
        print('\tError: Select a valid number (must be integer)\n')

## Tolerances:
while True:
    varepsilon1 = input('Introduce first tolerance (SVD). Continue with 1e-10: ')
    if not varepsilon1:
        varepsilon1 = 1e-10 
        break   # SVD
    elif is_float(varepsilon1):
        varepsilon1 = float(varepsilon1)
        break
    else:
        print('\tError: Please introduce a number\n')

while True:
    varepsilon = input('Introduce second tolerance (HODMD). Continue with 1e-3: ')
    if not varepsilon:
        varepsilon = 1e-3  
        break    # DMD 
    elif is_float(varepsilon):
        varepsilon = float(varepsilon)
        break
    else:
        print('\tError: Please introduce a number\n')
        
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

print(f'The temporal interval of the data analysed is [{Time[0]}, {Time[-1]}]')
while True:
    TimeFin = input('Introduce final time to predict your solution. Continue with reconstruction: ')
    if not TimeFin:
        TimePred=Time
        break
    elif is_float(TimeFin):
        if float(TimeFin) >= Time[-1]:
            TimeFin = float(TimeFin) / deltaT
            SnapDif=floor(abs(TimeFin-Time[-1])*deltaT)
            TimePred = np.linspace(0,SNAP+SnapDif-1,num=SNAP+SnapDif)*deltaT
            break
        else:
            print(f'\tError: Introduce a time higher than {Time[-1]}\n')
    else:
        print('\tError: Please introduce a number\n')

# Set tolerance for growth rates
TolGR=0
while True:
    TolGRset = input('Do you want to retain the modes with growth rate (in absolute value) < tolerance? Yes or No (y/n). Continue with Yes: ')
    if TolGRset.strip().lower() in ['y', 'yes'] or not TolGRset:
        TolGR=1
        TolGRvalue = input('Introduce the tolerance for the growth rate of the modes retained. Continue with 0.5: ')
        if not TolGRvalue:
            TolGRvalue = 0.5
            break
        elif is_float(TolGRvalue):
            TolGRvalue = float(TolGRvalue)
            break
        else: 
            print('\tError: Please introduce a number\n')
    elif TolGRset.strip().lower() in ['n', 'no']:
        break
    else: 
        print('\tError: Select yes or no (y/n)\n')


while True:
    TolGRset2 = input('Do you want to set the growth rate to 0 for the temporal predictions? Yes or No (y/n). Continue with No: ')
    if TolGRset2.strip().lower() in ['n', 'no'] or not TolGRset2:
        GRset=0
        break
    elif TolGRset2.strip().lower() in ['y', 'yes']:
        GRset=1
        break
    else: 
        print('\tError: Select yes or no (y/n)\n')

print('\n-----------------------------')
print('HODMD prediction summary:')
print('\n' + f'Number of snapshots set at: {SNAP}')
print(f'd Parameter set at: {d}')
print(f'Tolerances set at {varepsilon1} for SVD and {varepsilon} for HODMD')
print(f'Time gradient set at deltaT: {deltaT}')

################### Output ###################

print('\n-----------------------------')
print('Outputs:' + '\n')

filen = input('Enter folder name to save the outputs or continue with default folder name: ')
if not filen:
    filen = f'{timestr}_HODMD_pred_solution'
else:
    filen = f'{filen}'

while True:
    decision_2 = input('Select format of saved files (.mat, .npy). Continue with ".npy": ')
    if not decision_2 or decision_2.strip().lower() in ['mat', '.mat', 'npy', '.npy']:
        break
    else:
        print('\tError: Please select a valid output format\n')

print('')
    
# Create new folder:
if not os.path.exists(f'{path0}/{filen}'):
    os.mkdir(f'{path0}/{filen}')
if not os.path.exists(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}'):
    os.mkdir(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}')
if not os.path.exists(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes'):
    os.mkdir(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes')
    
Tensor = Tensor[..., :SNAP]
Tensor0 = Tensor.copy()
dims = Tensor.ndim
shape = Tensor.shape

if dims > 2:
    dims_prod = np.prod(shape[:-1])
    Tensor = np.reshape(Tensor, (dims_prod, shape[-1]))

notone=0
for i in range(0,np.size(np.shape(Tensor))):
    if np.shape(Tensor)[i] != 1:
        notone=notone+1

if notone <= 2:
    if d==1:
        print('Performing DMD. Please wait...\n')
        [u,Amplitude,Eigval,GrowthRate,Frequency,DMDmode] = DMDd.dmd1(Tensor, Time, varepsilon1, varepsilon)
        print('\nDMD complete!')
        dt=Time[1]-Time[0]
        icomp=complex(0,1)
        if TolGR==1:
            ind = np.where(np.abs(GrowthRate)<TolGRvalue)
            GrowthRate = GrowthRate[ind]
            Frequency = Frequency[ind]
            Amplitude = Amplitude[ind]
            DMDmode = DMDmode[:,ind]
            DimIn = np.size(ind)
            if GRset==1:
                vector_ceros = np.zeros(max(GrowthRate.shape))
                GrowthRate=np.transpose(vector_ceros)
            print('\nRecalculated Mode index, GrowthRate, Frequency and Amplitude of each mode:')
            ModeMatr=np.zeros((DimIn,4))
            for ii in range(0,DimIn):
                ModeMatr[ii,0]=ii+1
                ModeMatr[ii,1]=GrowthRate[ii]
                ModeMatr[ii,2]=Frequency[ii]
                ModeMatr[ii,3]=Amplitude[ii]
            print(ModeMatr)
        mu=np.zeros(np.size(GrowthRate),dtype=np.complex128)
        for iii in range(0,np.size(GrowthRate)):
            mu[iii] = np.exp(np.dot(dt,GrowthRate[iii]+np.dot(icomp,Frequency[iii])))
        Reconst=DMDd.remake(u,TimePred,mu)
    else:
        print('Performing HODMD. Please wait...\n')
        [u,Amplitude,Eigval,GrowthRate,Frequency,DMDmode] = DMDd.hodmd(Tensor, d, Time, varepsilon1, varepsilon)
        print('\nHODMD complete!')
        dt=Time[1]-Time[0]
        icomp=complex(0,1)
        if TolGR==1:
            ind = np.where(np.abs(GrowthRate)<TolGRvalue)
            GrowthRate = GrowthRate[ind]
            Frequency = Frequency[ind]
            Amplitude = Amplitude[ind]
            DMDmode = DMDmode[:,ind]
            DimIn = np.size(ind)
            if GRset==1:
                vector_ceros = np.zeros(max(GrowthRate.shape))
                GrowthRate=np.transpose(vector_ceros)
            print('\nRecalculated Mode index, GrowthRate, Frequency and Amplitude of each mode:')
            ModeMatr=np.zeros((DimIn,4))
            for ii in range(0,DimIn):
                ModeMatr[ii,0]=ii+1
                ModeMatr[ii,1]=GrowthRate[ii]
                ModeMatr[ii,2]=Frequency[ii]
                ModeMatr[ii,3]=Amplitude[ii]
            print(ModeMatr)
        mu=np.zeros(np.size(GrowthRate),dtype=np.complex128)
        for iii in range(0,np.size(GrowthRate)):
            mu[iii] = np.exp(np.dot(dt,GrowthRate[iii]+np.dot(icomp,Frequency[iii])))
        Reconst=DMDd.remake(u,TimePred,mu)

GRFreqAmp = np.zeros((np.size(GrowthRate),3))
for ind in range (0,np.size(GrowthRate)):
    GRFreqAmp[ind,0] = GrowthRate[ind]
    GRFreqAmp[ind,1] = Frequency[ind]
    GRFreqAmp[ind,2] = Amplitude[ind]

if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
    np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/GRFreqAmp.npy', GRFreqAmp)
    np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/ModeMatr.npy',ModeMatr)


if decision_2.strip().lower() in ['.mat', 'mat']:
    mdic = {"GRFreqAmp": GRFreqAmp}
    file_mat = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/GRFreqAmp.mat')
    hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

    mdic0 = {"ModeMatr": ModeMatr}
    file_mat0 = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/ModeMatr.mat')
    hdf5storage.savemat(file_mat0, mdic0, appendmat=True, format='7.3')

if dims <= 2:
    Norm2V = np.linalg.norm(Tensor, 2)
    diff = Tensor[:,0:SNAP] - Reconst[:,0:SNAP]
    Norm2diff = np.linalg.norm(diff, ord=2)
    RelativeErrorRMS = Norm2diff/Norm2V
    print('\n' + f'The relative error (RMS) is: ' +  format(RelativeErrorRMS,'e'))

    # Max Error:
    NormInfV = np.linalg.norm(Tensor, ord=1)
    NormInfdiff = np.linalg.norm(diff, ord=1)
    RelativeErrorMax = NormInfdiff/NormInfV
    print(f'\nThe relative error (MAX) is: ' + format(RelativeErrorMax, 'e'))

elif dims > 2:
    newshape = []
    newshape.append(shape[:-1])
    newshape.append(DMDmode.shape[-1])
    newshape = list(newshape[0]) + [newshape[1]]
    DMDmode = np.reshape(DMDmode, np.array(newshape))
    newshape1 = []
    newshape1.append(shape[:-1])
    newshape1.append(Reconst.shape[-1])
    newshape1 = list(newshape1[0]) + [newshape1[1]]
    Reconst = np.reshape(Reconst, np.array(newshape1))

    RRMSE = np.linalg.norm(np.reshape(Tensor0-Reconst[..., :SNAP],newshape=(np.size(Tensor0),1)),ord=2)/np.linalg.norm(np.reshape(Tensor0,newshape=(np.size(Tensor0),1)))
    print(f'\nRelative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%')

if not decision_2 or decision_2.strip().lower() in ['npy', '.npy']:
    np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/Reconst.npy', Reconst)
    np.save(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmodes.npy', DMDmode)


if decision_2.strip().lower() in ['.mat', 'mat']:
    mdic = {"Reconst": Reconst}
    file_mat = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/Reconst.mat')
    hdf5storage.savemat(file_mat, mdic, appendmat=True, format='7.3')

    mdic1 = {"DMDmodes": DMDmode}
    file_mat1 = str(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes.mat')
    hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')

# Result plots:
print(f'\nATTENTION!: All plots will be saved to {path0}/{filen}/d_{d}_tol_{varepsilon1}\n') 
print('Please CLOSE all figures to continue the run')

plt.figure(num='CLOSE TO CONTINUE RUN - Frequency/GrowthRate')
plt.plot(Frequency,GrowthRate, 'k+')
plt.yscale('log') 
plt.xlabel('Frequency ($\omega_{n}$)')
plt.ylabel('GrowthRate ($\delta_{n}$)')
plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/GrFreq.png')
plt.show()
plt.close()

plt.figure(num='CLOSE TO CONTINUE RUN - Frequency/Amplitude')
plt.plot(Frequency, Amplitude/np.amax(Amplitude),'r+')
plt.yscale('log')           # Logarithmic scale in y axis
plt.xlabel('Frequency ($\omega_{n}$)')
plt.ylabel('Amplitude divided by max. amplitude ($a_{n}$)')
plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/FrAmp.png')
plt.show()
plt.close()

if dims <=2:
    print('\nPlotting component comparison')
    plt.figure(num='CLOSE TO END RUN - Prediction')
    plt.plot(TimePred[:],Reconst[0,:], 'k-x', label = 'Predicted Data')
    plt.plot(Time[:],Tensor0[0,:], 'r-+', label = 'Real Data')
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.title('Real Data vs. Predicted Data')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/OrigPred.png')
    plt.show()
    plt.close()

    print(f'\nSaving DMDmode plot to {path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes\n')

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f'DMDmode')
    ax[0].contourf(DMDmode.real)
    ax[0].set_title('Real part')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].contourf(DMDmode.imag)
    ax[1].set_title('Imaginary part')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes/DMDmode.png')
    plt.show()

print(f'\nSaving first 3 DMDmode plots to {path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes\n')

if dims == 3:
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
        plt.show()

    print('\nPlotting component comparison')
    while True:
        while True:
            x = input(f'Select X coordinate (must be in range [0, {Tensor0.shape[1] - 1}]): ')
            if x.isdigit():
                if int(x) in range(0, Tensor0.shape[1]):
                    x = int(x)
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            elif not x:
                continue

        while True:
            y = input(f'Select Y coordinate (must be in range [0, {Tensor0.shape[0] - 1}]): ')
            if y.isdigit():
                if int(y) in range(0, Tensor0.shape[0]):
                    y = int(y)
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            elif not y:
                continue
            else:
                print('\tError: Select a valid number format (must be integer)\n') 
        
        fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Component comparison')
        fig.suptitle(f'Real Data vs Prediction')
        ax[0].contourf(Tensor0[:, :, 0])
        ax[0].scatter(x, y, c='black', s=50)
        ax[1].plot(TimePred[:], Reconst[y, x, :], 'k-x', label = 'Predicted Data')
        ax[1].plot(Time[:], Tensor0[y, x, :], 'r-+', label = 'Real Data')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Data')
        ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/OrigPred.png')
        plt.show()
        plt.close()
    
        while True:
            Resp = input('Do you want to plot another component? Yes or No (y/n). Continue with No: ')
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

if dims > 3: 
    for ModComp in range(DMDmode.shape[0]):
        for ModeNum in range(3):
            if DMDmode.ndim == 4:
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

            elif DMDmode.ndim == 5:
                nz = int(Tensor0.shape[3] / 2)
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

    print('\nPlotting component comparison')
    while True:
        while True:
            c = input(f'Select a component (max is {Tensor0.shape[0]}): ')
            if c.isdigit():
                if int(c) <= Tensor0.shape[0]:
                    c = int(c) - 1
                    break
                else:
                    print("\tError: Selected component doesn't exist\n")
            elif not c:
                continue

        while True:
            x = input(f'Select X coordinate (must be in range [0, {Tensor0.shape[2] - 1}]): ')
            if x.isdigit():
                if int(x) in range(0, Tensor0.shape[2]):
                    x = int(x)
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            elif not x:
                continue

        while True:
            y = input(f'Select Y coordinate (must be in range [0, {Tensor0.shape[1] - 1}]): ')
            if y.isdigit():
                if int(y) in range(0, Tensor0.shape[1]):
                    y = int(y)
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            elif not y:
                continue
            else:
                print('\tError: Select a valid number format (must be integer)\n') 
        if dims == 4:
            fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Component comparison')
            fig.suptitle(f'Real Data vs Prediction - Component {c+1}')
            ax[0].contourf(Tensor0[c, :, :, 0])
            ax[0].scatter(x, y, c='black', s=50)
            ax[1].plot(TimePred[:], Reconst[c, y, x, :], 'k-x', label = 'Predicted Data')
            ax[1].plot(Time[:], Tensor0[c, y, x, :], 'r-+', label = 'Real Data')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Data')
            ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.tight_layout()
            plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/OrigPred.png')
            plt.show()
            plt.close()
        
        elif dims == 5:
            nz = int(Tensor0.shape[3] / 2)
            fig, ax = plt.subplots(1, 2, num = 'CLOSE TO CONTINUE RUN - Component comparison')
            fig.suptitle(f'Real Data vs Prediction - XY plane - Component {c+1}')
            ax[0].contourf(Tensor0[c, :, :, nz, 0])
            ax[0].scatter(x, y, c='black', s=50)
            ax[1].plot(TimePred[:], Reconst[c, y, x, nz, :], 'k-x', label = 'Predicted Data')
            ax[1].plot(Time[:], Tensor0[c, y, x, nz, :], 'r-+', label = 'Real Data')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Data')
            ax[1].legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.tight_layout()
            plt.savefig(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/OrigPred.png')
            plt.show()
            plt.close()

        while True:
            Resp = input('Do you want to plot another component? Yes or No (y/n). Continue with No: ')
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


    if not os.path.exists(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes'):
        os.mkdir(f'{path0}/{filen}/d_{d}_tol_{varepsilon1}/DMDmodes')

    print(f'\nSelect component and temporal mode to plot a DMD mode')
    while True:
        while True:
            ModeNum = input(f'Introduce the DMD mode to plot (default is 1). Maximum number of modes is {DMDmode.shape[-1]}: ')
            if not ModeNum:
                ModeNum = 0
                break
            elif ModeNum.isdigit():
                if int(ModeNum) <= DMDmode.shape[-1]:
                    ModeNum = int(ModeNum)-1  
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')
        while True:
            ModComp = input(f'Introduce the component to plot (default component 1). Maximum number of components is {DMDmode.shape[0]}: ')
            if not ModComp:
                ModComp = 0
                break
            elif ModComp.isdigit():
                if int(ModComp) <= DMDmode.shape[0]:
                    ModComp = int(ModComp)-1
                    break
                else:
                    print('\tError: Selected value is out of bounds\n')
            else:
                print('\tError: Select a valid number format (must be integer)\n')

        if dims==4:
            fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode')
            fig.suptitle(f'DMDmode - Component {ModComp+1} Mode Number {ModeNum+1}')
            ax[0].contourf(DMDmode[ModComp,:,:,ModeNum].real)
            ax[0].set_title('Real part')

            ax[1].contourf(DMDmode[ModComp,:,:,ModeNum].imag)
            ax[1].set_title('Imaginary part')
            plt.show()

        elif dims==5:
            nz = int(Tensor0.shape[3] / 2)
            fig, ax = plt.subplots(1, 2, num=f'CLOSE TO CONTINUE RUN - DMD mode XY plane')
            fig.suptitle(f'DMDmode XY plane - Component {ModComp+1} Mode Number {ModeNum+1}')
            ax[0].contourf(DMDmode[ModComp,:,:,nz,ModeNum].real)
            ax[0].set_title('Real part - XY Plane')

            ax[1].contourf(DMDmode[ModComp,:,:,nz,ModeNum].imag)
            ax[1].set_title('Imaginary part - XY Plane')
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
    
    while True:   
        dec4 = input(f'Plot video for original data and predicted data? (y/n). Continue with Yes: ')
        if not dec4 or dec4.strip().lower() in ['y', 'yes']:
            decision4 = True
            break
        if dec4.strip().lower() in ['n', 'no']:
            decision4 = False
            exit()
        else:
            print('\tError: Please select yes or no (y/n)\n')

    if dims == 5:
        nz = int(Tensor0.shape[3] / 2)
        while True:
            plane = input('Select a plane (XY, XZ, YZ)')
            if plane.strip().lower() == 'xy':
                Tensor0 = Tensor0[:, :, :, nz, :]
                Reconst = Reconst[:, :, :, nz, :]
                break
            elif plane.strip().lower() == 'xz':
                Tensor0 = Tensor0[:, :, 0, :, :]
                Reconst = Reconst[:, :, 0, :, :]
                break
            elif plane.strip().lower() == 'yz':
                Tensor0 = Tensor0[:, 0, :, :, :]
                Reconst = Reconst[:, 0, :, :, :]
                break
            else:
                print('\tError: Select a valid plane\n')

    else:
        pass

    titles = []
    [titles.append(f'Component {i+1}') for i in range(Tensor0.shape[0])]

    while True:
        if decision4 == True:
            vidvel = input(f'Select a component (max is {Tensor0.shape[0]}). Continue with component 1: ')
            if not vidvel:
                vel = 0
                video(Tensor0, vel, Title = f'Original Data - {titles[vel]}')
                video(Reconst, vel, Title = f'Predicted data - {titles[vel]}')
                break
            elif vidvel.isdigit():
                if int(vidvel) <= Tensor0.shape[0]:
                    vel = int(vidvel) - 1
                    video(Tensor0, vel, Title = f'Original Data - {titles[vel]}')
                    video(Reconst, vel, Title = f'Predicted data - {titles[vel]}')
                    break
                else:
                    print("\tError: Select a valid component\n")
            else:
                print('\tError: Introduce a valid format (must be integer)\n')

    while True:
        ch1 = input('Would you like to plot another component? (y/n). Continue with No: ')
        if ch1.strip().lower() in ['y', 'yes']:
            while True:
                vidvel = input(f'Select a component (max is {Tensor0.shape[0]}): ')
                if vidvel.isdigit():
                    if int(vidvel) <= Tensor0.shape[0]:
                        vel = int(vidvel) - 1
                        video(Tensor0, vel, Title = f'Original Data - {titles[vel]}')
                        video(Reconst, vel, Title = f'Predicted data - {titles[vel]}')
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