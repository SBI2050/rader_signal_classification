import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from cyclespectrum import cyclespectrum
from ambifunb import ambifunb
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

locate = 'D:\\Working_Files\\work\\radar\\radar202302\\matdata\\常规信号'

def FFT(N,y):
    y = y - np.mean(y)
    start=0 #开始采样位置
    wave_data = y[start:start+N]
    c=np.fft.fft(wave_data)*2/N
    #常规显示采样频率一半的频谱
    d=int(len(c)/2)
    return c[:d]
    

if __name__ == "__main__":
    list = ['1', '2', '3', '4', '1_2', '1_3', '1_4', '1_2_3','1_2_4', '1_3_4', '1_2_3_4', '1_2_3_4(2)']
    phase = []
    cyclespectrum_long = []
    feature = []
    frequency_spectrum_long = []
    label = []
    acc_cf = []
    acc_af = []
    acc_cfa = []
    acc_phase = []

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
            original_one = []
            original_two = []
            x = np.arange(1,129)
            for s in range(8):
                y = datafile[:,s]
                z = phasefile[:,s]
                original_one = np.append(original_one, y, axis=0)
                phase = np.append(phase, z, axis=0)
                
            original_two = original_one.reshape(8,-1)
            '''计算连接信号循环谱'''
            cyclespectrum_long = np.append(cyclespectrum_long, cyclespectrum(original_two), axis=0)
            '''对连接信号进行模糊切片分析'''
            x1 = original_one
            x1 = stats.zscore(x1)  # `mapstd` is equivalent to `zscore` in scipy.stats module
            L = len(x1)
            slice = int(L/2)+1

            naf=abs(ambifunb(x1))

            feature = np.append(feature, naf[slice] , axis=0)        #取对应行
            '''连接信号频谱分析'''
            frequency_spectrum_long = np.append(frequency_spectrum_long, abs(FFT(128*8,original_one)), axis=0)
            '''制作标签'''
            label = np.append(label, i + 1)
    phase = phase.reshape(-1,128*8)
    cyclespectrum_long = cyclespectrum_long.reshape(-1,64*8)
    feature = feature.reshape(-1,1023)
    frequency_spectrum_long = frequency_spectrum_long.reshape(-1,64*8)
    concatenated_cf = np.concatenate((cyclespectrum_long, frequency_spectrum_long), axis=1)
    concatenated_af = np.concatenate((feature, frequency_spectrum_long), axis=1)
    concatenated_cfa = np.concatenate((concatenated_cf, feature), axis=1)

    print('phase.shape', phase.shape)
    print('concatenated_cf.shape', concatenated_cf.shape)
    print('concatenated_af.shape', concatenated_af.shape)
    print('concatenated_cfa.shape', concatenated_cfa.shape)
    print('label.shape', label.shape)

    pca_x = range(1,512)
    for pca_num in range(1,512):
        pca = PCA(n_components=pca_num)
        concatenated_pca_cf = pca.fit_transform(concatenated_cf)
        concatenated_pca_af = pca.fit_transform(concatenated_af)
        concatenated_pca_cfa = pca.fit_transform(concatenated_cfa)
        phase_pca = pca.fit_transform(phase)
        print("n_components = {}".format(pca_num))
        '''
        SVM
        '''
        for counter in range(4):
            if counter == 0:
                dataFile = concatenated_pca_cf
            elif counter == 1:
                dataFile = concatenated_pca_af
            elif counter == 2:
                dataFile = concatenated_pca_cfa
            else:
                dataFile = phase_pca
        
            X_train, X_test, y_train, y_test = train_test_split(dataFile, label, test_size = 0.8, random_state=0, stratify=label)

            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            scaler.fit(dataFile)
            dataFile = scaler.transform(dataFile)
            scaler.fit(X_test)
            X_test = scaler.transform(X_test)

            svm = SVC(kernel='rbf', cache_size=3000, C = 10, gamma = 'auto', shrinking=True).fit(X_train, y_train)
            print("The calculation had been done.")
            acc = svm.score(X_test, y_test)
            print("Test set accuracy:{:.2f}".format(acc))
            out = svm.predict(dataFile).reshape((1, label.shape[0]), order='F')

            if counter == 0:
                acc_cf = np.append(acc_cf, acc)
            elif counter == 1:
                acc_af = np.append(acc_af, acc)
            elif counter == 2:
                acc_cfa = np.append(acc_cfa, acc)
            else:
                acc_phase = np.append(acc_phase, acc)

    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(pca_x, acc_cf, color='r', alpha=0.8, linewidth=1, label='concatenated_cf')
    plt.plot(pca_x, acc_af, color='b', alpha=0.8, linewidth=1, label='concatenated_af')
    plt.plot(pca_x, acc_cfa, color='g', alpha=0.8, linewidth=1, label='concatenated_cfa')
    plt.plot(pca_x, acc_cfa, color='k', alpha=0.8, linewidth=1, label='phase')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.show()
