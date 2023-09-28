import numpy as np
import scipy.io as sio
import os
import scipy.signal as signal
from scipy.integrate import simps


def normalize(data):  # data: 62*180
    mu = np.expand_dims(np.mean(data, axis=2), axis=2)
    std = np.expand_dims(np.std(data, axis=2), axis=2)
    return (data - mu) / std


def band_power(data, fs, band):
    band = np.asarray(band)
    low, high = band
    freq, psd = signal.welch(data, fs=fs, window='hann', noverlap=0, nperseg=64, nfft=None)
    # print("freq", freq)
    # frequency resolution
    freq_res = freq[1] - freq[0]
    # find closest indices of band in frequency vector
    idx_band = np.logical_and(freq >= low, freq <= high)
    bp = simps(psd[:, idx_band], dx=freq_res)
    # print("bp", bp.shape)
    return bp


def psd_feature_extract(data, fs):
    power_features = np.empty([data.shape[0], 4])
    # print("power_features", power_features.shape)
    power_theta = band_power(data, fs, [4, 8])
    power_features[:, 0] = np.log2(power_theta)
    power_alpha = band_power(data, fs, [8, 14])
    power_features[:, 1] = np.log2(power_alpha)
    power_beta = band_power(data, fs, [14, 31])
    power_features[:, 2] = np.log2(power_beta)
    power_gamma = band_power(data, fs, [31, 45])
    power_features[:, 3] = np.log2(power_gamma)
    return power_features


def sample_deap_4D_channel(data_path, sample_path):
    for i, path in enumerate(os.listdir(data_path)):
        # 生成列表字典['s01.mat','s02.mat',...]
        print("DEAP person_%d processing" % i)
        eeg = sio.loadmat(data_path + path)['data']
        label = sio.loadmat(data_path + path)['labels']
        valence = label[:, 0]
        arousal = label[:, 1]

        signal_total = eeg[:, 0:32, -7680:]
        de_feature = np.empty([signal_total.shape[0], signal_total.shape[1], 120, 4])
        for l in range(signal_total.shape[0]):
            for s in range(120):
                de_feature[l, :, s, :] = psd_feature_extract(signal_total[l, :, 64 * s:64 * (s + 1)], fs=128)
        print("de", de_feature.shape)
        de_feature_norm = normalize(de_feature)
        de_feature_norm_reshape = de_feature_norm.transpose(0, 1, 3, 2)
        print(de_feature_norm_reshape.shape)
        # emotion label
        for j in range(40):
            if valence[j] <= 5:
                valence[j] = 0
            else:
                valence[j] = 1
            if arousal[j] <= 5:
                arousal[j] = 0
            else:
                arousal[j] = 1
        valence = valence.reshape(-1, 1)
        valence = valence.flatten()
        print(valence)
        arousal = arousal.reshape(-1, 1)
        arousal = arousal.flatten()
        print(arousal)
        # data save
        np.save(sample_path + 'person_%d data' % i, de_feature_norm_reshape)
        np.save(sample_path + 'person_%d label_V' % i, valence)
        np.save(sample_path + 'person_%d label_A' % i, arousal)


if __name__ == '__main__':
    data_path = 'F:/dataset/DEAP/'
    sample_path = 'F:/dataset/DEAP_DE/'
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    sample_deap_4D_channel(data_path, sample_path)
