#RRMSE temporal , RRMSE spectral  and CC , correlation coefficient
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from numpy.linalg import norm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mutual_info_score
N=512
nfft=256
fs=256
def psd(x):
    Px = []
    for i in range(N // nfft):  # 窗与窗之间数据不重叠
        w = np.hanning(nfft)  # 加汉尼窗
        p = np.abs(fft(w * x[i * nfft:(i + 1) * nfft])) ** 2 / fs / norm(w) ** 2  # 计算功率谱
        Px.append(p)  # 每个窗的结果
    Px = np.mean(Px, axis=0)  # 将所有窗平均得到最终结果
    return Px
def get_rms(records):
    # rms is root mean square
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def RRMSE_temporal(x_truth,y_gen):
    return get_rms(x_truth-y_gen)/get_rms(x_truth)

def MI(x_truth,y_gen):
    return mutual_info_score(x_truth,y_gen)


def RRMSE_spectral(x_truth,y_gen):
    Px = psd(x_truth)
    Py = psd(y_gen)
    return get_rms(Px-Py)/get_rms(Px)

def RRMAE(x_truth,y_gen):
    Px = psd(x_truth)
    Py = psd(y_gen)
    return mean_absolute_error(Px,Py)
    

def cc(x_truth,y_gen):
    return np.corrcoef(x_truth,y_gen)
def log(base,x):
    return np.log(x)/np.log(base)

def get_snrdb(eeg,noise):
    if get_rms(noise)<0.001:
        eeg_noise = get_rms(eeg) / (get_rms(noise)+0.0001)
        print(get_rms(noise))
    else:
        eeg_noise = get_rms(eeg)/get_rms(noise)
    return 10*log(10,eeg_noise)

def get_snrdb_new(eeg,noise):
    if get_rms(noise)<0.001:
        eeg_noise = get_rms(eeg) / (get_rms(noise)+0.0001)
        print(get_rms(noise))
    else:
        eeg_noise = get_rms(eeg)/get_rms(noise)
    return eeg_noise
# x=np.array([[1,2],[1,2]])
#

# y=x[0].reshape(x.shape[1])
# print(get_rms(x[0]))
# print(y)
# test_input = np.load('../data/test_input.npy')
# test_output = np.load('../data/test_output.npy')
# a=np.array([[1,2],[2,3]])
# b=np.array([[1,1],[1,1]])
# print(test_input-test_output)