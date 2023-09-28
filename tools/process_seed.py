import scipy.io as sio
import numpy as np

"""
Take the first 3 minutes of data from the clips, i.e., dim=180. 
Standardize the data of this dimension using standard deviation. 
Combine the EEG data from three sessions of each participant into one file, 
resulting in a total of 16 files (15 participant EEG files + 1 unified label file).

Standardize the dimension with dim=180, and distribute the DE features of different frequency bands alternately. 
Each channel corresponds to 5 frequency band data.
"""


def normalize(data):  # data: 62*180*5
    mu = np.expand_dims(np.mean(data, axis=1), axis=1)
    std = np.expand_dims(np.std(data, axis=1), axis=1)
    return (data - mu) / std


def sample_seed_4D_3(data_path, sample_path, feat_type):
    # label 0，1，2
    label = sio.loadmat(data_path + 'label.mat')
    label = np.array(label['label']) + 1
    label = np.tile(label, 3)  # （1,15）-> （1,45）
    label = label.transpose(1, 0)  # （1,45）->（45,1）
    label = label.flatten()
    np.save(sample_path + 'label', label)

    for i in range(15):  # subject
        print("SEED person_%d processing" % (i + 1))
        signal_total = np.empty([45, 62, 5, 180])

        for j in range(3):  # session
            file_name = str(i * 3 + j + 1) + '.mat'
            print(file_name)
            data = sio.loadmat(data_path + file_name)
            for k in range(15):  # film clips
                data_sample = data[feat_type + '%d' % (k + 1)]  # (62,235,5)
                sample = data_sample[:, 0:180, :]  # （62,180,5）
                normal_sample = normalize(sample)  # 标准差标准化：（data-均值）/ 方差 => 得到服从标准正态分布的数据（62,180,5）
                normal_sample_reshape = normal_sample.transpose(0, 2, 1)  # （62，5,180）
                signal_total[k + j * 15, :, :, :] = normal_sample_reshape  # 把一个受试者的所有会话数据都整合在一起3*15=45
        print(signal_total.shape)  # （45,62,5,180）
        np.save(sample_path + 'person_%d data' % i, signal_total)


if __name__ == '__main__':
    data_path = 'F:/dataset/SEED/ExtractedFeatures/'
    sample_path = 'F:/dataset/SEED/DE_4D/'
    feat_type = 'de_movingAve'  # de_movingAve、de_LDS等

    sample_seed_4D_3(data_path, sample_path, feat_type)
