import numpy as np

def ambifunb(x, tau=None, N=None):
    """
    Narrow-band ambiguity function.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Signal if auto-AF, or [X1,X2] if cross-AF.
    tau : numpy.ndarray, optional
        Vector of lag values (default: -Nx/2:Nx/2).
    N : int, optional
        Number of frequency bins (default: length(x)).
    
    Returns:
    --------
    naf : numpy.ndarray
        Doppler-lag representation, with the doppler bins stored in the rows and the time-lags stored in the columns.
    xi : numpy.ndarray
        Vector of doppler values.
    """
    xrow = x.shape[0]
    xcol = 1
    if xcol == 0 or xcol > 2:
        raise ValueError("X must have one or two columns")
    
    if tau is None:
        if xrow % 2 == 0:
            tau = np.arange(-xrow//2+1, xrow//2)
        else:
            tau = np.arange(-(xrow-1)//2, (xrow+1)//2)
    
    tauro = 1
    taucol = tau.shape[0]
    
    if N is None:
        N = xrow
    
    naf = np.zeros((N, taucol))
    
    for icol in range(taucol):
        taui = tau[icol]
        t = np.arange(1+abs(taui), xrow-abs(taui))
        naf[t-1,icol] = x[t+taui-1] * np.conj(x[t-taui-1])
    
    naf = np.fft.fft(naf, axis=0)
    naf = np.concatenate((naf[(N+np.mod(N,2))//2:N,:], naf[0:(N+np.mod(N,2))//2,:]), axis=0)
    #print('naf',naf)
    xi = np.fft.fftfreq(N)
    return naf
