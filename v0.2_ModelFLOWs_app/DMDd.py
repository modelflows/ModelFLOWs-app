import numpy as np
import numba as nb

def matlen(var):
    '''Equivalent to Matlab's length()'''
    if np.size(np.shape(var))==1:
        x = np.size(var)
    else:
        x = max(np.shape(var))
    return x

def error(V,Vrec):
    '''Relative RMS and max errors''' 
    return np.linalg.norm(V-Vrec)/np.linalg.norm(V),np.linalg.norm(V-Vrec,np.inf)/np.linalg.norm(V,np.inf)

# @nb.njit(cache=True)

def truncatedSVD(A,esvd):
    '''Decomposition into singular values, truncated on esvd'''
    U,s,Wh=np.linalg.svd(A,full_matrices=False)
    n=0
    norm=np.linalg.norm(s)
    for i in range(s.size):
        if np.linalg.norm(s[i:])/norm<=esvd:
            break
        else:
            n+=1
    return U[:,:n],s[:n],Wh[:n,:]

def unfold(A,dim):
    '''Turns tensor into matrix keeping the columns on dim'''
    ax=np.arange(A.ndim)
    return np.reshape(np.moveaxis(A,ax,np.roll(ax,dim)),(A.shape[dim],A.size//A.shape[dim]))

def fold(B,shape,dim):
    '''Reverse operation to the unfold function'''
    ax=np.arange(len(shape))
    shape=np.roll(shape,-dim)
    A=np.reshape(B,shape)
    return np.moveaxis(A,ax,np.roll(ax,-dim))

def tprod(S,U):
    '''Tensor product of an ndim-array and multiple matrices'''
    T = S
    shap = list(np.shape(S))
    for i in range(0,np.size(U)):
        x = np.count_nonzero(U[0][i])
        if not x==0:
            shap[i] = np.shape(U[0][i])[0]
            H = unfold(T,i)
            T = fold(np.dot(U[0][i],H), shap, i)
    return T

def tensormatmul(S,U,dim):
    '''Internal product of tensor and matrix in dim'''
    shape=np.array(S.shape)
    shape[dim]=U.shape[0]
    return fold(U@unfold(S,dim),shape,dim)

def truncHOSVD(A,esvd):
    '''Decomposition into singular values for tensors, truncated in esvd'''
    Ulist=[]
    slist=[]
    S=A.copy()
    for i in range(A.ndim):
        [U,s,Vh]=truncatedSVD(unfold(A,i),esvd/np.sqrt(A.ndim))
        Ulist.append(U)
        S=tensormatmul(S,U.T,i)
    for i in range(A.ndim):
        s=np.zeros(S.shape[0])
        ax=np.arange(A.ndim)
        for j in range(S.shape[0]):
            s[j]=np.linalg.norm(S[j])
        slist.append(s)
        S=np.moveaxis(S,ax,np.roll(ax,1))
    return S,Ulist,slist

def dmd1(V,t,esvd,edmd):
    '''First order dynamic modal decomposition:
        Input:
            -V (IxJ): Snapshot matrix.
            -t (J): Time vector.
            -esvd: First tolerance (SVD).
            -edmd: Second tolerance (DMD modes).
        Output:
            -u (Ixn): mode matrix (columns) sorted from biggest to smallest amplitude, put to scale.
            -areal (n): amplitude vector (sorted) used to put u to scale.
            -eigval: eigenvalues
            -delta (n): growth rate vector.
            -omega (n): frequency vector.
            -DMDmode: Modes'''
    dt=t[1]-t[0]
    
    #Reduced snapshots:
    U,s,Wh=truncatedSVD(V,esvd) #I*r', r'*r', r'*J
#    Vvir=np.conj(U[:,:n].T)@V
#    Vvir=np.diag(s[:n])@Wh[:n,:] #r'*J
#    Vvir=s[:n]*Wh[:n,:]
    Vvir=np.diag(s)@Wh #r'*J
    n=s.size

    #Spatial complexity kk:
    NormS=np.linalg.norm(s,ord=2)
    kk=0
    for k in range(0,n):
        if np.linalg.norm(s[k:n],2)/NormS>esvd:
            kk=kk+1
    print(f'Spatial complexity: {kk}')
    
    #Koopman matrix reconstruction:
    Uvir,svir,Wvirh=np.linalg.svd(Vvir[:,:-1],full_matrices=False) #r'*r', r'*r', r'*(J-1)
    Rvir=Vvir[:,1:]@Wvirh.conj().T@np.diag(svir**-1)@Uvir.conj().T #r'*r'
    eigval,eigvec=np.linalg.eig(Rvir) #r'*r'
    
    #Frequencies and Growthrate:
    delta=np.log(eigval).real/dt #r'
    omega=np.log(eigval).imag/dt #r'
    
    #Amplitudes:
    A=np.zeros((eigvec.shape[0]*Vvir.shape[1],eigvec.shape[1])) #(r'*J)*r'
    b=np.zeros(eigvec.shape[0]*Vvir.shape[1])#(r'*J)*1
    for i in range(Vvir.shape[1]):
        A[i*eigvec.shape[0]:(i+1)*eigvec.shape[0],:]=eigvec@np.diag(eigval**i)
        b[i*eigvec.shape[0]:(i+1)*eigvec.shape[0]]=Vvir[:,i]
        
    Ua,sa,Wa=np.linalg.svd(A,full_matrices=False) #(r'*J)*r', r'*r', r'*r'
    a=Wa.conj().T@np.diag(sa**-1)@Ua.conj().T@b #r'
    
    #Modes:
    uvir=eigvec@np.diag(a) #r'*r'
#    u=U[:,:n]@uvir #I*r'
    u=U@uvir #I*r'
#    areal=np.zeros(a.size) #Real amplitudes
#    for i in range(u.shape[1]):
#        areal[i]=np.linalg.norm(u[:,i])/np.sqrt(V.shape[1])
    areal=np.linalg.norm(u,axis=0)/np.sqrt(V.shape[0])

    #Spectral complexity:
    kk3=0
    for m in range(0,np.size(areal)):
        if areal[m]/np.max(areal)>edmd:
            kk3=kk3+1
    print(f'Spectral complexity: {kk3}')

    idx=np.flip(np.argsort(areal))
    u=u[:,idx]
    areal=areal[idx]
    eigval=eigval[idx]
    delta=delta[idx]
    omega=omega[idx]

    #Filter important ones:
    mask=(areal/areal[0])>edmd

    #Mode Matrix:
    ModeMatr=np.zeros((kk3,4))
    for ii in range(0,kk3):
        ModeMatr[ii,0]=ii+1
        ModeMatr[ii,1]=delta[mask][ii]
        ModeMatr[ii,2]=omega[mask][ii]
        ModeMatr[ii,3]=areal[mask][ii]
    # print('Mode Number, GrowthRate, Frequency and Amplitude of each mode:')
    # print(ModeMatr)

    #Calculate modes:
    u=u[:,mask]
    U = U[:,0:kk]
    DMDmode=np.zeros((V.shape[0],kk3),dtype=np.complex128)
    Amplitude0=np.zeros(kk3)
    for m in range(0,kk3):
        NormMode=np.linalg.norm(np.dot(U,u[:,m]),ord=2)/np.sqrt(V.shape[0])
        Amplitude0[m]=NormMode
        DMDmode[:,m]=np.dot(U,u[:,m])/NormMode
    
    return u,areal[mask],eigval[mask],delta[mask],omega[mask],DMDmode

#@nb.njit(cache=True)
def hodmd(V,d,t,esvd,edmd):
    '''High order (d) modal decomposition:
        Input:
            -V (IxJ): snapshot matrix.
            -t (J): time vector.
            -d: parameter of DMD-d, higher order Koopman assumption (int>=1).
            -esvd: first tolerance (SVD).
            -edmd: second tolerance (DMD-d modes).
        Output:
            -u (Ixn): mode matrix (columns) sorted from biggest to smallest amplitude, put to scale.
            -areal (n): amplitude vector (sorted) used to put u to scale.
            -eigval: eigenvalues
            -delta (n): growth rate vector.
            -omega (n): frequency vector.
            -DMDmode: Modes'''
    dt=t[1]-t[0]
    
    #Reduced snapshots:
    U,s,Wh=truncatedSVD(V,esvd) #I*n, n*n, n*J
#    Vvir=np.conj(U[:,:n].T)@V
    Vvir=np.diag(s)@Wh #n*J
    #print("Tamaño Vvir: ",Vvir.shape)
#    Vvir=s*Wh
    n=s.size

    #Spatial complexity kk:
    NormS=np.linalg.norm(s,ord=2)
    kk=0
    for k in range(0,n):
        if np.linalg.norm(s[k:n],2)/NormS>esvd:
            kk=kk+1
    print(f'Spatial complexity: {kk}')
    
    #Reduced and grouped snapshots:
    Vdot=np.zeros((d*n,Vvir.shape[1]-d+1),dtype=Vvir.dtype) #(d*n)*(J-d+1)
    for j in range(d):
        Vdot[j*n:(j+1)*n,:]=Vvir[:,j:Vvir.shape[1]-d+j+1]
    #print("Size Vdot: ",Vdot.shape)
    
    #Reduced, grouped and again reduced snapshots:
    Udot,sdot,Whdot=truncatedSVD(Vdot,esvd) #(d*n)*N, N*N, N*(J-d+1)
    Vvd=np.diag(sdot)@Whdot #N*(J-d+1)
    #print("Size Vvd: ",Vvd.shape)
    
    #Spatial dimension reduction:
    print(f'Spatial dimension reduction: {np.size(sdot)}')

    #Koopman matrix reconstruction:
    Uvd,svd,Whvd=np.linalg.svd(Vvd[:,:-1],full_matrices=False) #N*r', r'*r', r'*(J-d)
    Rvd=Vvd[:,1:]@Whvd.conj().T@np.diag(svd**-1)@Uvd.conj().T #r'*r'
    eigval,eigvec=np.linalg.eig(Rvd) #r'*r'==N*N
    #print("Size Rvd: ",Rvd.shape)

    #Frequencies and growthrate:
    delta=np.log(eigval).real/dt #r'
    omega=np.log(eigval).imag/dt #r'
    
    #Modes 
    q=(Udot@eigvec)[(d-1)*n:d*n,:] #Taking steps back n*N
    Uvir=q/np.linalg.norm(q,axis=0) #n*N
    #print("Size Uvir: ",Uvir.shape)

    #Amplitudes:
    A=np.zeros((Uvir.shape[0]*Vvir.shape[1],Uvir.shape[1]),dtype=np.complex128) #(n*J)*N
    #print("Size A: ",A.shape)
    b=np.zeros(Uvir.shape[0]*Vvir.shape[1],dtype=Vvir.dtype)#(n*J)*1
    for i in range(Vvir.shape[1]):
        A[i*Uvir.shape[0]:(i+1)*Uvir.shape[0],:]=Uvir@np.diag(eigval**i)
        b[i*Uvir.shape[0]:(i+1)*Uvir.shape[0]]=Vvir[:,i]
#    print(A[:Uvir.shape[0],:])
    Ua,sa,Wa=np.linalg.svd(A,full_matrices=False) #(n*J)*N, N*N, N*N
    a=Wa.conj().T@np.diag(sa**-1)@Ua.conj().T@b #N
    
#    print(eigval)
    #Modes
    uvir=Uvir@np.diag(a) #n*N
    u=U@uvir #I*N
    areal=np.linalg.norm(u,axis=0)/np.sqrt(V.shape[0])
    #print("Size ufull: ",u.shape)

    #Spectral complexity:
    kk3=0
    for m in range(0,np.size(areal)):
        if areal[m]/np.max(areal)>edmd:
            kk3=kk3+1
    print(f'Spectral complexity: {kk3}')
    
    idx=np.flip(np.argsort(areal))
    u=u[:,idx]
    areal=areal[idx]
    eigval=eigval[idx]
    delta=delta[idx]
    omega=omega[idx]
    
    #Filter important ones:
    mask=(areal/areal[0])>edmd
    
    #Mode Matrix:
    ModeMatr=np.zeros((kk3,4))
    for ii in range(0,kk3):
        ModeMatr[ii,0]=ii+1
        ModeMatr[ii,1]=delta[mask][ii]
        ModeMatr[ii,2]=omega[mask][ii]
        ModeMatr[ii,3]=areal[mask][ii]
    # print('Mode Number, GrowthRate, Frequency and Amplitude of each mode:')
    # print(ModeMatr)

    #Calculate DMD modes:
    u=u[:,mask]
    U = U[:,0:kk]
    DMDmode=np.zeros((V.shape[0],kk3),dtype=np.complex128)
    Amplitude0=np.zeros(kk3)
    for m in range(0,kk3):
        NormMode=np.linalg.norm(np.dot(U.T,u[:,m]),ord=2)/np.sqrt(V.shape[0])
        Amplitude0[m]=NormMode
        DMDmode[:,m]=u[:,m]/NormMode

    return u,areal[mask],eigval[mask],delta[mask],omega[mask],DMDmode

def tensorHODMD(V,d,t,esvd,edmd):
    '''High order (d) modal decomposition for tensors:
        Input:
            -V (I1xI2x...xJ): Snapshots.
            -t (J): Time vector.
            -d: parameter of DMD-d, higher order Koopman assumption (int>=1).
            -esvd: first tolerance (SVD).
            -edmd: second tolerance (DMD-d modes).
        Output:
            -u (I1xI2x...xn): mode tensor (columns) sorted from biggest to smallest amplitude, put to scale.
            -areal (n): amplitude vector (sorted) used to put u to scale.
            -eigval: eigenvalues
            -delta (n): growth rate vector.
            -omega (n): frequency vector.
            -DMDmode: Modes'''
    
    #Reduced snapshots:
    S,Vl,sl=truncHOSVD(V,esvd) #I*n, n*n, n*J
    Svir=S.copy()
    for i in range(S.ndim-1):
        Svir=tensormatmul(Svir,Vl[i],i)
    Svir=Svir/sl[-1]

    uvir,a,eigval,delta,omega,DMDmode=hodmd((sl[-1]*Vl[-1]).T,d,t,esvd,edmd)
    u=tensormatmul(Svir,uvir.T,V.ndim-1)
    
    return u,a,eigval,delta,omega,DMDmode

def remake(u,t,mu):
    '''Reconstructs original data from DMD-d results:
        Input:
            -u (Ixn): Mode matrix (columns).
            -t (J): Time vector.
            -delta (n): vector de ratios de crecimiento.
            -omega (n): vector de frecuencias.
            -mu: np.exp(np.dot((t[1]-t[0]),delta[iii]+np.dot(complex(0,1),omega[iii])))
        Output:
            -vrec (IxJ): reconstructed snapshots'''
            
    vrec=np.zeros((u.shape[0],t.size),dtype=np.complex128)
    for i in range(t.size):
        for j in range(mu.shape[0]):
            vrec[:,i]+=u[:,j]*mu[j]**i#*np.exp((delta[j]+omega[j]*1j)*t[i])
    return vrec

#@nb.njit(cache=True)
def remakeTens(u,t,mu):
    '''Reconstructs original data from DMD-d results:
        Input:
            -u (Ixn): Mode matrix (columns).
            -t (J): Time vector.
            -delta (n): vector de ratios de crecimiento.
            -omega (n): vector de frecuencias.
            -mu: np.exp(np.dot((t[1]-t[0]),delta[iii]+np.dot(complex(0,1),omega[iii])))
        Output:
            -vrec (IxJ): reconstructed snapshots'''
            
    shape=np.array(u.shape,dtype=np.int32)
    shape[-1]=t.size
    vrec=np.zeros(tuple(shape),dtype=np.complex128)
    idx=[slice(None)]*shape.size
    for i in range(t.size):
        for j in range(u.shape[-1]):
            idx[-1]=i
            vrec[tuple(idx)]+=u.take(j,axis=-1)*mu[j]**i#*np.exp((delta[j]+omega[j]*1j)*t[i])
    return vrec

def hodmd_IT(Vvir,d,t,esvd,edmd):
    '''High order (d) modal decomposition:
        Input:
            -V (IxJ): snapshot matrix.
            -t (J): time vector.
            -d: parameter of DMD-d, higher order Koopman assumption (int>=1).
            -esvd: first tolerance (SVD).
            -edmd: second tolerance (DMD-d modes).
        Output:
            -u (Ixn): mode matrix (columns) sorted from biggest to smallest amplitude, put to scale.
            -areal (n): amplitude vector (sorted) used to put u to scale.
            -eigval: eigenvalues
            -delta (n): growth rate vector.
            -omega (n): frequency vector.
            -DMDmode: Modes'''
    dt=t[1]-t[0]
    N=Vvir.shape[0]
    K=Vvir.shape[1]

    #Reduced and grouped snapshots:
    Vdot=np.zeros((d*N,Vvir.shape[1]-d+1),dtype=Vvir.dtype) #(d*n)*(J-d+1)
    for j in range(d):
        Vdot[j*N:(j+1)*N,:]=Vvir[:,j:Vvir.shape[1]-d+j+1]
    #print("Size Vdot: ",Vdot.shape)
    
    #Reduced, grouped and again reduced snapshots:
    Udot,sdot,Whdot=truncatedSVD(Vdot,esvd) #(d*n)*N, N*N, N*(J-d+1)
    Vvd=np.diag(sdot)@Whdot #N*(J-d+1)
    #print("Size Vvd: ",Vvd.shape)
    
    #Spatial dimension reduction:
    print(f'Spatial dimension reduction: {np.size(sdot)}')

    #Koopman matrix reconstruction:
    Uvd,svd,Whvd=np.linalg.svd(Vvd[:,:-1],full_matrices=False) #N*r', r'*r', r'*(J-d)
    Rvd=Vvd[:,1:]@Whvd.conj().T@np.diag(svd**-1)@Uvd.conj().T #r'*r'
    eigval,eigvec=np.linalg.eig(Rvd) #r'*r'==N*N
    #print("Size Rvd: ",Rvd.shape)

    #Frequencies and growthrate:
    delta=np.log(eigval).real/dt #r'
    omega=np.log(eigval).imag/dt #r'

    #Modes 
    q=(Udot@eigvec)[(d-1)*N:d*N,:] #Taking steps back n*N
    Uvir=q/np.linalg.norm(q,axis=0) #n*N

    
    #Amplitudes:
    A=np.zeros((Uvir.shape[0]*Vvir.shape[1],Uvir.shape[1]),dtype=np.complex128) #(n*J)*N

    b=np.zeros(Uvir.shape[0]*Vvir.shape[1],dtype=Vvir.dtype)#(n*J)*1
    for i in range(Vvir.shape[1]):
        A[i*Uvir.shape[0]:(i+1)*Uvir.shape[0],:]=Uvir@np.diag(eigval**i)
        b[i*Uvir.shape[0]:(i+1)*Uvir.shape[0]]=Vvir[:,i]

    Ua,sa,Wa=np.linalg.svd(A,full_matrices=False) #(n*J)*N, N*N, N*N
    a=Wa.conj().T@np.diag(sa**-1)@Ua.conj().T@b #N
    
#    print(eigval)
    #Modes
    u=np.zeros((np.shape(Uvir)[0],np.size(eigval)),dtype=np.complex128)
    for m in range(0,np.size(eigval)):
        u[:,m]=np.dot(a[m],Uvir[:,m])
    #uvir=Uvir@np.diag(a) #n*N
    #u=U@uvir #I*N
    #areal=np.linalg.norm(u,axis=0)/np.sqrt(V.shape[0])
    #print("Size ufull: ",u.shape)
    areal=np.zeros(np.size(eigval),dtype=np.complex128)
    for mm in range(0,np.size(eigval)):
        GR=delta[mm]
        AmplGR=np.exp(np.dot(GR,t))
        AmplN=np.linalg.norm(AmplGR,ord=2)/np.sqrt(K)
        areal[mm]=np.dot(np.linalg.norm(u[:,mm],ord=2),AmplN)
    areal=np.real(areal)
    
    idx=np.flip(np.argsort(areal,axis=0))
    u=u[:,idx]
    areal=areal[idx]
    eigval=eigval[idx]
    delta=delta[idx]
    omega=omega[idx]

    #Spectral complexity:
    kk3=0
    for m in range(0,np.size(areal)):
        if areal[m]/np.max(areal)>edmd:
            kk3=kk3+1
    print(f'Spectral complexity: {kk3}')

    return u[:,0:kk3],areal[0:kk3],eigval[0:kk3],delta[0:kk3],omega[0:kk3]

def remakeTens_IT(t,t0,u,delta,omega):
    '''Reconstructs original data from DMD-d results:
        Input:
            -u (Ixn): Mode matrix (columns).
            -t (J): Time vector.
            -delta (n): vector de ratios de crecimiento.
            -omega (n): vector de frecuencias.
            -mu: np.exp(np.dot((t[1]-t[0]),delta[iii]+np.dot(complex(0,1),omega[iii])))
        Output:
            -vrec (IxJ): reconstructed snapshots'''

    dt=t-t0
    icomp=complex(0,1)
    mu=np.zeros(np.size(delta),dtype=np.complex128)
    for iii in range(0,np.size(delta)):
        mu[iii] = np.exp(np.dot(dt,delta[iii]+np.dot(icomp,omega[iii])))
    return np.dot(u,mu)

def reconst_IT(hatMode,Time,U,S,sv,nn,TimePos,GrowthRate,Frequency):
    '''Reconstructs original data from DMD-d results.'''
    N = np.shape(hatMode)[0]
    K = matlen(Time)
    hatTReconst = np.zeros((N,K), dtype=np.complex128)

    # Reconstruction using the DMD expansion:
    for k in range(0,K):
        hatTReconst[:,k] = remakeTens_IT(Time[k],Time[0],hatMode,GrowthRate,Frequency)
    
    # Reconstruction of the original tensor using the reduced tensor and the tensor core:
    Unondim = U
    UTnondim = np.zeros(shape=np.shape(U), dtype=object)
    UTnondim[0][TimePos-1] = np.zeros((nn[TimePos-1],np.shape(hatTReconst)[1]),dtype=np.complex128)
    for kk in range(0,nn[TimePos-1]):
        UTnondim[0][TimePos-1][kk,:] = hatTReconst[kk,:]/sv[0][TimePos-1][kk]
    
    Unondim[0][TimePos-1] = np.transpose(UTnondim[0][TimePos-1])
    TensorReconst = tprod(S,Unondim)
    return TensorReconst

def modes_IT(N,hatMode,Amplitude,U,S,nn,TimePos):
    '''Calculate DMD modes from results'''
    hatMode_m = np.zeros((N,np.size(Amplitude)), dtype=hatMode.dtype)
    for ii in range(0,np.size(Amplitude)):
        hatMode_m[:,ii] = hatMode[:,ii]/Amplitude[ii]
    
    ModesT = np.zeros((nn[TimePos-1],np.shape(hatMode_m)[1]), dtype=hatMode_m.dtype)
    for kk in range(0,nn[TimePos-1]):
        ModesT[kk,:] = hatMode_m[kk,:]
    
    # Temporal DMD modes in reduced dimension:
    Modes = U
    Modes[0,TimePos-1] = np.transpose(ModesT)
    
    # Reconstruction of the temporal DMD modes:
    DMDmode = tprod(S,Modes)
    
    return DMDmode