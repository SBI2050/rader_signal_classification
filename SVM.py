import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import time

if __name__ == "__main__":
    start_time = time.perf_counter()  # 记录程序开始时间

    '''
    读取数据
    '''
    #dataFile = "original.mat"
    #dataFile = "phase.mat"
    #dataFile = "fft_phase.mat"         #寄上加寄
    #dataFile = "phase_mv.mat"
    #dataFile = "phase1.mat"
    #dataFile = "fft_phase1.mat"
    #dataFile = "phase_mv1.mat"
    #dataFile = 'frequency_spectrum.mat'
    #dataFile = 'cyclespectrum_long.mat'
    #dataFile = 'feature.mat'
    #dataFile = 'frequency_spectrum_long.mat'
    #dataFile = 'concatenated_cf.mat'
    #dataFile = 'concatenated_af.mat'
    #dataFile = 'concatenated_cfa.mat'
    dataFile = 'concatenated_pcfa.mat'
    #dataFile = 'power_spectral_density.mat'
    #dataFile = 'mixed_fp.mat'
    #dataFile = 'relevance_function.mat'
    #dataFile = 'mixed_fr.mat'
    #dataFile = 'mixed_fpr.mat'
    labelFile = 'label.mat'
    datafile = scio.loadmat(dataFile)["data"]
    labelfile = scio.loadmat(labelFile)["label"]
    print(labelfile.shape)

    #print(datafile)

    label = np.reshape(labelfile, (datafile.shape[0],), order='F')
    print(datafile[:,[0,1]].shape)
    print(label.shape)

    X_train, X_test, y_train, y_test = train_test_split(datafile, label, test_size = 0.8, random_state=0, stratify=label)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(datafile)
    datafile = scaler.transform(datafile)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', cache_size=3000, C = 10, gamma = 'auto', verbose=True, shrinking=True).fit(X_train, y_train) 
    #svm = SVC(kernel='rbf', cache_size=3000, C = 10, gamma = 'auto', verbose=True, shrinking=True, class_weight={1:3}).fit(X_train, y_train)
    #svm = SVC(kernel='rbf', cache_size=3000, C = 10, gamma = 'auto', verbose=True, shrinking=True, class_weight={1:10}).fit(X_train, y_train)  #常规信号(3)用
    #svm = SVC(kernel='rbf', cache_size=3000, C = 10, gamma = 'auto', verbose=True, shrinking=True, class_weight={1:13}).fit(X_train, y_train) 
    #joblib.dump(svm, "original_SVM_train75%_rbf_10_auto_shrinking=1.pkl")
    print("The calculation had been done.")
    print("Test set accuracy:{:.2f}".format(svm.score(X_test, y_test)))
    out = svm.predict(datafile).reshape((labelfile.shape[0], labelfile.shape[1]), order='F')
    end_time = time.perf_counter()  # 记录程序结束时间
    run_time = end_time - start_time  # 计算程序运行时间
    print("程序运行时间为：{:.2f}s".format(run_time))
