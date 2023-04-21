'''提取循环谱零点切片特征(转自MATLAB)'''
'''input: pluse or pluses(单个脉冲、脉冲集)'''
'''the fft length ,defalut L = length(pluse)'''
'''output: zeroslice_cyc/ the zero slice featrue of input pluse(循环谱零点切片特征)'''
'''import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy

def cyclespectrum(pulse):
    L = len(pulse)
    H = np.shape(pulse)[0]
    for i in range(0):
        y = pulse
        scaler = StandardScaler()
        y = scaler.fit_transform(y)
        T = len(y)
        if( L>T or L<T):
            m = np.fft.fft(y,L)
        else:
            m = np.fft.fft(y,T)
        M = round(T/64)
        y = np.zeros((2*M+1,T/2))
        for tao1 in range(0, int(T/2)):
            if (T-2*tao1 > 2*M+1):
                y[0:2*M+1, tao1+1] = np.multiply(m[int(T/2-M-tao1):int(T/2+M-tao1+1)], np.conj(m[int(T/2-M+tao1):int(T/2+M+tao1+1)]))/T
            else:
                y[M+1-int((T-2*tao1)/2):M+2+int((T-2*tao1)/2), tao1+1] = np.multiply(m[0:T-2*tao1], np.conj(m[1+2*tao1:T+1]))/T

        n=np.fliplr(y)
        Y = np.concatenate((n, y))
        w = scipy.signal.windows.hamming(2*M+1, sym=True)
        W = w * np.ones(1,T)
        Z = np.zeros_like(Y)
        Z = W * Y
        D = sum(Z)
        D = 1/(2*M+1)*D
        return abs(D[1:T/2])'''

import numpy as np
from scipy.signal import hamming

def cyclespectrum(pulse):
    L = pulse.shape[1]
    zeroslice_cyc = np.zeros((pulse.shape[0], int(L/2)))
    
    for i in range(pulse.shape[0]):
        y = pulse[i, :]
        y = (y - np.mean(y)) / np.std(y)  # 归一化去均值
        T = len(y)
        if L != T:
            m = np.fft.fft(y, L)
        else:
            m = np.fft.fft(y)
        M = round(T/64)
        y2 = np.zeros((2*M, int(T/2)), dtype=m.dtype)
        for tao1 in range(0, T//2):
            if T - 2*tao1 > 2*M+1:
                y2[:, tao1] = m[T//2-M-tao1:T//2+M-tao1] * np.conj(m[T//2-M+tao1:T//2+M+tao1]) / T
            else:
                y2[M+1-(T-2*tao1)//2:M+2+(T-2*tao1)//2, tao1] = m[1:T-2*tao1] * np.conj(m[1+2*tao1:T]) / T
        
        y2 = np.hstack((np.fliplr(y2), y2))
        
        # smooth the cyclic periodogram.
        w = hamming(2*M)
        W = np.tile(w, (T, 1)).T
        Z = W * y2
        D = np.sum(Z, axis=0)
        D = 1 / (2*M+1) * D
        zeroslice_cyc[i, :] = np.abs(D[0:T//2])
    zeroslice_cyc = zeroslice_cyc.flatten()
        
    return zeroslice_cyc


