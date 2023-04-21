import os
import numpy as np
import scipy.io as scio
import pylab
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   

locate = 'D:\\Working_Files\\work\\radar\\radar202302\\matdata\\常规信号'
filename = '常规信号.txt'

def Draw_Waveform(x,y):
    pylab.plot(x, y)
    pylab.xlabel("时间")
    pylab.ylabel("振幅")
    pylab.show()

def FFT(x,y):
    f = 2000
    N = 44100
    y = y - np.mean(y)
    start=0 #开始采样位置
    df = f/(N-1) # 分辨率
    freq = [df*n for n in range(0,N)] #N个元素
    wave_data2 = y[start:start+N]
    c=np.fft.fft(wave_data2)*2/N
    #常规显示采样频率一半的频谱
    d=int(len(c)/2)
    '''pylab.plot(freq[:d],abs(c[:d]),'r')
    pylab.show()'''
    return c[:d]

def PSD(y):
    nfft = 128
    fs = 2000
    [Pxx1,f1] = plt.psd(y,               # 离散信号
                NFFT=nfft,               # 每个窗的长度
                Fs=fs,                   # 采样频率
                detrend='mean',          # 去掉均值
                window=np.hanning(nfft), # 加汉明窗
                noverlap=int(nfft*3/4),  # 每个窗重叠75%的数据
                sides='twosided')        # 求双边谱
    '''plt.xscale('log')
    plt.show()'''
    return Pxx1,f1

#计算一个序列，它和自己延迟（lag）的序列的相关性
def autocorrelation(x,lags):
    #计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
	n = len(x)
	#x = np.array(x)
	relevance = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
	  	/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
	return relevance
    

if __name__ == "__main__":
    list = ['1', '2', '3', '4', '1_2', '1_3', '1_4', '1_2_3','1_2_4', '1_3_4', '1_2_3_4', '1_2_3_4(2)']
    num = 100
    original = []
    frequency_spectrum = []
    power_spectral_density = []
    mixed_fp = []
    mixed_fr = []
    mixed_fpr = []
    relevance_function = []
    label = []
    #frequency_spectrum = np.array(frequency_spectrum)
    #for i in range(4):
    for i in range(12):
        for j in range(126):
        #for j in range(1):
            '''导入数据'''
            dataset = str(j + 1)
            try:
                loadfile = os.path.join(locate, list[i], dataset + '.mat')
                datafile = scio.loadmat(loadfile)['tlog']
            except FileNotFoundError:
                break
            #print(datafile.shape)
            #datafile = scio.loadmat(loadfile)
            #print(datafile)
            y = datafile[:,0]
            #y = y.reshape(-1,128)
            '''数据完整性检测'''
            '''try:
                relevance = [np.correlate(y[k:]-y[k:].mean(),y[:2-k]-y[:2-k].mean())[0]/(y[k:].std()*y[:2-k].std()*(2-k)) for k in range(1,2)]
            except RuntimeWarning:
                j = j + 1
                break
            '''
            m = 0
            
            for k in range(128):
                if y[k] == 0:
                    m = m + 1
                    '''break
                else:
                    continue
            if k != 127:'''
            if m >= 1:
                '''with open(filename, 'a') as file_project:
                    file_project.write(f'The data{dataset} in {list[i]} has a problem in it, please check it out.\n')'''
                print(f'The data{dataset} in {list[i]} has a problem in it, please check it out.')
                j = j + 1
                continue

            x = np.arange(1,129)
            #print(x)
            for s in range(8):
                y = datafile[:,s]

                '''画出波形图'''
                #Draw_Waveform(x,y)
                original = np.append(original, y, axis=0)
                '''频谱分析'''
                #FFT(x,y)
                #print(FFT(x,y).shape)
                frequency_spectrum = np.append(frequency_spectrum, abs(FFT(x,y)), axis=0)
                mixed_fp = np.append(mixed_fp, abs(FFT(x,y)), axis=0)
                mixed_fr = np.append(mixed_fr, abs(FFT(x,y)), axis=0)
                '''计算功率谱密度'''
                Pxx,f = PSD(y)
                power_spectral_density = np.append(power_spectral_density, Pxx, axis=0)
                mixed_fp = np.append(mixed_fp, Pxx, axis=0)
                '''计算自相关'''
                #fig = plot_acf(y)
                relevance = autocorrelation(y,num)
                relevance = np.array(relevance)
                relevance_function = np.append(relevance_function, relevance, axis=0)
                mixed_fr = np.append(mixed_fr, relevance, axis=0)
                mixed_fpr = np.append(mixed_fp, relevance_function, axis=0)
                #plt.show()
                '''制作标签'''
                label = np.append(label, i + 1)
                '''if i == 0:
                    label = np.append(label, 1) 
                else:
                    label = np.append(label, 2)'''
    frequency_spectrum = frequency_spectrum.reshape(-1,64)
    original = original.reshape(-1,128)
    power_spectral_density = power_spectral_density.reshape(-1,128)
    mixed_fp = mixed_fp.reshape(-1,192)
    relevance_function = relevance_function.reshape(-1,num)
    mixed_fr = mixed_fr.reshape(-1,64+num)
    mixed_fpr = mixed_fpr.reshape(-1,192+num)
    #label = np.transpose(label)
    scio.savemat('original.mat', {'data': original})
    scio.savemat('frequency_spectrum.mat', {'data': frequency_spectrum})
    scio.savemat('power_spectral_density.mat', {'data': power_spectral_density})
    scio.savemat('mixed_fp.mat', {'data': mixed_fp})
    scio.savemat('relevance_function.mat', {'data': relevance_function})
    scio.savemat('mixed_fr.mat', {'data': mixed_fr})
    scio.savemat('mixed_fpr.mat', {'data': mixed_fpr})
    scio.savemat('label.mat', {'label': label})
    print(original.shape)
    print(frequency_spectrum.shape) 
    print(power_spectral_density.shape)
    print(mixed_fp.shape)
    print(relevance_function.shape)
    print(mixed_fr.shape)
    print(mixed_fpr.shape)
    print(label)
    print(label.shape)
