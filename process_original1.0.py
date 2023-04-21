import os
import numpy as np
import scipy.io as scio
import pylab
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt
from cyclespectrum import cyclespectrum
from ambifunb import ambifunb
from sklearn.decomposition import PCA
from scipy import stats
import time

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   

locate = 'D:\\Working_Files\\work\\radar\\radar202302\\matdata\\线性调频'
#filename = '常规信号.txt'

def Draw_Waveform(x,y):
    pylab.plot(x, y)
    pylab.xlabel("时间")
    pylab.ylabel("振幅")
    pylab.show()

def FFT(N,y):
    f = 1
    y = y - np.mean(y)
    start=0 #开始采样位置
    df = f/(N-1) # 分辨率
    freq = [df*n for n in range(0,N)] #N个元素
    wave_data = y[start:start+N]
    c=np.fft.fft(wave_data)*2/N
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
    original = []
    phase = []
    phase_mv = []
    #fft_phase = []
    frequency_spectrum = []
    cyclespectrum_long = []
    feature = []
    feature1 = np.zeros((1,1023))
    feature2 = np.zeros((1,1023))
    feature3 = np.zeros((1,1023))
    feature4 = np.zeros((1,1023))
    feature5 = np.zeros((1,1023))
    feature6 = np.zeros((1,1023))
    feature7 = np.zeros((1,1023))
    frequency_spectrum_long = []
    label = []
    #frequency_spectrum = np.array(frequency_spectrum)

    start_time = time.perf_counter()  # 记录程序开始时间
    #for i in range(4):
    for i in range(11):
        for j in range(126):
        #for j in range(1):
            '''导入数据'''
            dataset = str(j + 1)
            try:
                loadfile = os.path.join(locate, list[i], dataset + '.mat')
                datafile = scio.loadmat(loadfile)['tlog']
                phasefile = scio.loadmat(loadfile)['tphase']
            except FileNotFoundError:
                break
            #print(datafile.shape)
            #datafile = scio.loadmat(loadfile)
            #print(datafile)
            y = datafile[:,0]
            #y = y.reshape(-1,128)
            '''数据完整性检测'''
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
            original_one = []
            phase_one = []
            original_two = []
            x = np.arange(1,129)
            #print(x)
            for s in range(8):
                y = datafile[:,s]
                z = phasefile[:,s]
                '''画出波形图'''
                #Draw_Waveform(x,y)
                original = np.append(original, y, axis=0)
                original_one = np.append(original_one, y, axis=0)
                #phase_one = np.append(phase_one, z, axis=0)
                phase = np.append(phase, z, axis=0)
                phase_mv = np.append(phase_mv, np.mean(z))
                phase_mv = np.append(phase_mv, np.var(z))
                '''频谱分析'''
                #FFT(x,y)
                #print(FFT(x,y).shape)
                frequency_spectrum = np.append(frequency_spectrum, abs(FFT(128,y)), axis=0)
                
            original_two = original_one.reshape(8,-1)
            '''计算连接信号循环谱'''
            cyclespectrum_long = np.append(cyclespectrum_long, cyclespectrum(original_two), axis=0)
            '''对连接信号进行模糊切片分析'''
            '''data_len, m = original_two.shape              #此处注释掉内容为每通道单独分析的部分
            for i in range(data_len):
                x1 = original_two[i,:]
                x1 = stats.zscore(x1)  # `mapstd` is equivalent to `zscore` in scipy.stats module
                L = len(x1)
                slice = int(L/2)+1
                slice1 = int(L/2)+1-1
                slice2 = int(L/2)+1-2
                slice3 = int(L/2)+1-3
                slice4 = int(L/2)+1-4
                slice5 = int(L/2)+1-5
                slice6 = int(L/2)+1-6
                slice7 = int(L/2)+1-7

                naf=abs(ambifunb(x1))
                #print('naf',naf)
                #print('len_naf',len(naf),len(naf[0]))
                #print('slice1',slice1)

                feature[i,:]=naf[slice]         #取对应行
                feature1[i,:]=naf[slice1]
                feature2[i,:]=naf[slice2]
                feature3[i,:]=naf[slice3]
                feature4[i,:]=naf[slice4]
                feature5[i,:]=naf[slice5]
                feature6[i,:]=naf[slice6]
                feature7[i,:]=naf[slice7]'''
            x1 = original_one
            x1 = stats.zscore(x1)  # `mapstd` is equivalent to `zscore` in scipy.stats module
            L = len(x1)
            slice = int(L/2)+1
            '''slice1 = int(L/2)+1-1
            slice2 = int(L/2)+1-2
            slice3 = int(L/2)+1-3
            slice4 = int(L/2)+1-4
            slice5 = int(L/2)+1-5
            slice6 = int(L/2)+1-6
            slice7 = int(L/2)+1-7'''

            naf=abs(ambifunb(x1))
            #print('naf',naf)
            #print('len_naf',len(naf),len(naf[0]))
            #print('slice1',slice1)

            feature = np.append(feature, naf[slice] , axis=0)        #取对应行
            '''feature1 = np.append(feature, naf[slice1] , axis=0)
            feature2 = np.append(feature, naf[slice2] , axis=0)
            feature3 = np.append(feature, naf[slice3] , axis=0)
            feature4 = np.append(feature, naf[slice4] , axis=0)
            feature5 = np.append(feature, naf[slice5] , axis=0)
            feature6 = np.append(feature, naf[slice6] , axis=0)
            feature7 = np.append(feature, naf[slice7] , axis=0)'''
            '''连接信号频谱分析'''
            frequency_spectrum_long = np.append(frequency_spectrum_long, abs(FFT(128*8,original_one)), axis=0)
            #fft_phase = np.append(fft_phase, abs(FFT(128*8,phase_one)), axis=0)
            '''制作标签'''
            label = np.append(label, i + 1)
            '''if i == 0:
                label = np.append(label, 1) 
            else:
                label = np.append(label, 2)'''
    original = original.reshape(-1,128*8)
    phase = phase.reshape(-1,128*8)
    phase_mv = phase_mv.reshape(-1,2*8)
    frequency_spectrum = frequency_spectrum.reshape(-1,64*8)
    cyclespectrum_long = cyclespectrum_long.reshape(-1,64*8)
    #fft_phase = fft_phase.reshape(-1,64*8)
    pca = PCA(n_components=64)
    #cyclespectrum_long = pca.fit_transform(cyclespectrum_long)
    feature = feature.reshape(-1,1023)
    '''feature1 = feature1.reshape(-1,1023)
    feature2 = feature2.reshape(-1,1023)
    feature3 = feature3.reshape(-1,1023)
    feature4 = feature4.reshape(-1,1023)
    feature5 = feature5.reshape(-1,1023)
    feature6 = feature6.reshape(-1,1023)
    feature7 = feature7.reshape(-1,1023)'''
    frequency_spectrum_long = frequency_spectrum_long.reshape(-1,64*8)
    concatenated_cf = np.concatenate((cyclespectrum_long, frequency_spectrum_long), axis=1)
    concatenated_af = np.concatenate((feature, frequency_spectrum_long), axis=1)
    concatenated_cfa = np.concatenate((concatenated_cf, feature), axis=1)
    concatenated_pcfa = np.concatenate((phase, concatenated_cfa), axis=1)
    pca = PCA(n_components=64)
    phase = pca.fit_transform(phase)
    concatenated_cf = pca.fit_transform(concatenated_cf)
    concatenated_af = pca.fit_transform(concatenated_af)
    concatenated_cfa = pca.fit_transform(concatenated_cfa)
    concatenated_pcfa = pca.fit_transform(concatenated_cfa)
    #label = np.transpose(label)
    end_time = time.perf_counter()  # 记录程序结束时间
    run_time = end_time - start_time  # 计算程序运行时间

    scio.savemat('original.mat', {'data': original})
    scio.savemat('phase.mat', {'data': phase})
    scio.savemat('phase_mv.mat', {'data': phase_mv})
    scio.savemat('frequency_spectrum.mat', {'data': frequency_spectrum})
    scio.savemat('cyclespectrum_long.mat', {'data': cyclespectrum_long})
    scio.savemat('feature.mat', {'data': feature})
    '''scio.savemat('feature1.mat', {'data': feature1})
    scio.savemat('feature2.mat', {'data': feature2})
    scio.savemat('feature3.mat', {'data': feature3})
    scio.savemat('feature4.mat', {'data': feature4})
    scio.savemat('feature5.mat', {'data': feature5})
    scio.savemat('feature6.mat', {'data': feature6})
    scio.savemat('feature7.mat', {'data': feature7})'''
    scio.savemat('frequency_spectrum_long.mat', {'data': frequency_spectrum_long})
    #scio.savemat('fft_phase.mat', {'data': fft_phase})
    scio.savemat('concatenated_cf.mat', {'data': concatenated_cf})
    scio.savemat('concatenated_af.mat', {'data': concatenated_af})
    scio.savemat('concatenated_cfa.mat', {'data': concatenated_cfa})
    scio.savemat('concatenated_pcfa.mat', {'data': concatenated_pcfa})
    scio.savemat('label.mat', {'label': label})
    print('original.shape', original.shape)
    print('phase.shape', phase.shape)
    print('phase_mv.shape', phase_mv.shape)
    print('frequency_spectrum.shape', frequency_spectrum.shape)
    print('cyclespectrum_long.shape', cyclespectrum_long.shape)
    print('feature.shape',feature.shape)
    '''print('feature1.shape',feature1.shape)
    print('feature2.shape',feature2.shape)
    print('feature3.shape',feature3.shape)
    print('feature4.shape',feature4.shape)
    print('feature5.shape',feature5.shape)
    print('feature6.shape',feature6.shape)
    print('feature7.shape',feature7.shape)'''
    print('frequency_spectrum_long.shape',frequency_spectrum_long.shape)
    print('concatenated_cf.shape', concatenated_cf.shape)
    print('concatenated_af.shape', concatenated_af.shape)
    print('concatenated_cfa.shape', concatenated_cfa.shape)
    print('label', label)
    print('label.shape', label.shape)
    print("程序运行时间为：{:.2f}s".format(run_time))