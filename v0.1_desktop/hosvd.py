import numpy as np
from sklearn.utils.extmath import randomized_svd

def matlen(var):
    '''Equivalent to Matlab's length()'''
    if np.size(np.shape(var))==1:
        x = np.size(var)
    else:
        x = max(np.shape(var))
    return x

def unfold(A,dim):
    '''Turns tensor into matrix keeping the columns on dim'''
    ax=np.arange(A.ndim)
    return np.reshape(np.moveaxis(A,ax,np.roll(ax,dim)),(A.shape[dim],A.size//A.shape[dim]))


def fold(B,dim,shape):
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
            T = fold(np.dot(U[0][i],H),i,shap)
    return T

def svdtrunc(A, n):
    '''Truncated svd'''
    U, S, _ = randomized_svd(A, n_components = n)
    return U, S

def HOSVD_function(T,n):
    '''Perform hosvd to tensor'''
    P = T.ndim
    U = np.zeros(shape=(1,P), dtype=object)
    UT = np.zeros(shape=(1,P), dtype=object)
    sv = np.zeros(shape=(1,P), dtype=object)
    producto = n[0]
    
    for i in range(1,P):
        producto = producto * n[i]
    
    n = list(n)       # to be able to assign values

    for i in range(0,P):
        n[i] = int(np.amin((n[i],producto/n[i])))
        A = unfold(T, i) # Modificado de unfold a unfolding
        Uaux = []
        # SVD based reduction of the current dimension (i):
        Ui, svi = svdtrunc(A, n[i])
        if n[i] < 2:
            Uaux = np.zeros((np.shape(Ui)[0],2))
            Uaux[:,0] = Ui[:,0]
            U[0][i] = Uaux
        else:
            U[0][i] = Ui[:,0:n[i]]
        # U[i] = Ui
        UT[0][i] = np.transpose(U[0][i])
        sv[0][i] = svi
    S = tprod(T, UT)
    TT = tprod(S, U) 
    return TT, S, U, sv, n

def HOSVD(Tensor,varepsilon1,nn,n,TimePos):
    '''Perform hosvd to input data and retain all the singular values'''

    if np.iscomplex(Tensor.any()) == False:
        Tensor = Tensor.astype(np.float32)
    [TT,S,U,sv,n] = HOSVD_function(Tensor,n) # El problema está aquí dentro
    ## Set the truncation of the singular values using varepsilon1
    # (automatic truncation)
    for i in range(1,np.size(nn)):
        count = 0
        for j in range(0,np.shape(sv[0][i])[0]):
            if sv[0][i][j]/sv[0][i][0]<=varepsilon1:
                pass
            else:
                count = count+1

        nn[i] = count
    print(f'Initial number of singular values: {n}')
    print(f'Number of singular values retained: {nn}')
    
    ## Perform HOSVD retaining n singular values: reconstruction of the modes
    [TT,S2,U,sv2,nn2] = HOSVD_function(Tensor,nn)
    
    ## Construct the reduced matrix containing the temporal modes:
    UT = np.zeros(shape=np.shape(U), dtype=object)
    hatT = []
    for pp in range(0,np.size(nn2)):
        UT[0][pp] = np.transpose(U[0][pp])
    for kk in range(0,nn2[TimePos-1]):
        hatT.append(np.dot(sv2[0][TimePos-1][kk],UT[0][TimePos-1][kk,:]))
    hatT = np.reshape(hatT, newshape=(len(hatT),np.size(hatT[0])))
    return hatT,U,S2,sv,nn2,n,TT