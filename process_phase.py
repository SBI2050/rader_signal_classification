import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   

locate = 'D:\\Working_Files\\work\\radar\\radar202302\\matdata\\常规信号(2)'

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
    

if __name__ == "__main__":
    list = ['1', '2', '3', '4', '1_2', '1_3', '1_4', '1_2_3','1_2_4', '1_3_4', '1_2_3_4', '1_2_3_4(2)']
    phase = []
    phase_mv = []
    fft_phase = []
    phase1 = []
    phase_mv1 = []
    fft_phase1 = []
    label = []

    start_time = time.perf_counter()  # 记录程序开始时间
    for i in range(11):
        for j in range(126):
            '''导入数据'''
            dataset = str(j + 1)
            try:
                loadfile = os.path.join(locate, list[i], dataset + '.mat')
                datafile = scio.loadmat(loadfile)['tlog']
                phasefile = scio.loadmat(loadfile)['tphase']
            except FileNotFoundError:
                break
            y = datafile[:,0]
            '''数据完整性检测'''
            m = 0
            for k in range(128):
                if y[k] == 0:
                    m = m + 1
            if m >= 1:
                print(f'The data{dataset} in {list[i]} has a problem in it, please check it out.')
                j = j + 1
                continue
            phase_one = []
            phase_one1 = []
            x = np.arange(1,129)
            '''纵向'''
            for s in range(8):
                y = datafile[:,s]
                z = phasefile[:,s]
                '''相位信息特征提取'''
                phase_one = np.append(phase_one, z, axis=0)
                phase = np.append(phase, z, axis=0)
                phase_mv = np.append(phase_mv, np.mean(z))
                phase_mv = np.append(phase_mv, np.var(z))
            for t in range(128):
                y = datafile[t,:]
                z = phasefile[t,:]
                '''相位信息特征提取'''
                phase_one1 = np.append(phase_one1, z, axis=0)
                phase1 = np.append(phase1, z, axis=0)
                phase_mv1 = np.append(phase_mv1, np.mean(z))
                phase_mv1 = np.append(phase_mv1, np.var(z))
            '''频谱分析'''
            fft_phase = np.append(fft_phase, abs(FFT(128*8,phase_one)), axis=0)
            fft_phase1 = np.append(fft_phase1, abs(FFT(128*8,phase_one1)), axis=0)
            '''制作标签'''
            label = np.append(label, i + 1)
            '''横向'''

    phase = phase.reshape(-1,128*8)
    phase1 = phase1.reshape(-1,8*128)
    phase_mv = phase_mv.reshape(-1,2*8)
    phase_mv1 = phase_mv1.reshape(-1,2*128)
    fft_phase = fft_phase.reshape(-1,64*8)
    fft_phase1 = fft_phase1.reshape(-1,4*128)
    end_time = time.perf_counter()  # 记录程序结束时间
    run_time = end_time - start_time  # 计算程序运行时间

    scio.savemat('phase.mat', {'data': phase})
    scio.savemat('phase_mv.mat', {'data': phase_mv})
    scio.savemat('fft_phase.mat', {'data': fft_phase})
    scio.savemat('label.mat', {'label': label})
    scio.savemat('phase1.mat', {'data': phase1})
    scio.savemat('phase_mv1.mat', {'data': phase_mv1})
    scio.savemat('fft_phase1.mat', {'data': fft_phase1})
    print('phase.shape', phase.shape)
    print('phase_mv.shape', phase_mv.shape)
    print('fft_phase.shape', fft_phase.shape)
    print('phase1.shape', phase1.shape)
    print('phase_mv1.shape', phase_mv1.shape)
    print('fft_phase1.shape', fft_phase1.shape)
    print('label', label)
    print('label.shape', label.shape)
    print("程序运行时间为：{:.2f}s".format(run_time))