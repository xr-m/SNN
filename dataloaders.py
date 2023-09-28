import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_seed_and_deap(cfg, index, target_sub_id):
    global source_sample, source_label
    target_sample = np.load(cfg.sample_path + cfg.sub_file_name % target_sub_id)
    if cfg.data_name == 'SEED':
        target_label = np.load(cfg.sample_path + cfg.elabels_file_name)
    elif cfg.data_name.startswith('DEAP_'):
        target_label = np.load(cfg.sample_path + cfg.elabels_file_name % target_sub_id)
    for k, j in enumerate(index):
        if k == 0:
            source_sample = np.load(cfg.sample_path + cfg.sub_file_name % j)
            if cfg.data_name == 'SEED':
                source_label = np.load(cfg.sample_path + cfg.elabels_file_name)
            elif cfg.data_name.startswith('DEAP_'):
                source_label = np.load(cfg.sample_path + cfg.elabels_file_name % j)
        else:
            data = np.load(cfg.sample_path + cfg.sub_file_name % j)
            if cfg.data_name == 'SEED':
                label = np.load(cfg.sample_path + cfg.elabels_file_name)
            elif cfg.data_name.startswith('DEAP_'):
                label = np.load(cfg.sample_path + cfg.elabels_file_name % j)

            source_sample = np.append(source_sample, data, axis=0)
            #  final format: bach_size*channel*length
            source_label = np.append(source_label, label)
    p = int(source_sample.shape[0] // target_sample.shape[0])
    target_sample = np.repeat(target_sample, axis=0, repeats=p)
    target_label = np.repeat(target_label, axis=0, repeats=p)
    source_sample = source_sample.reshape([source_sample.shape[0], cfg.in_channels, -1])
    target_sample = target_sample.reshape([target_sample.shape[0], cfg.in_channels, -1])
    source = Mydataset(source_sample, source_label)
    target = Mydataset(target_sample, target_label)
    source_loader = DataLoader(source, batch_size=cfg.batchsize, shuffle=True, drop_last=True)
    target_loader = DataLoader(target, batch_size=cfg.batchsize, shuffle=True, drop_last=True)
    return source_loader, target_loader


class Mydataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        sample = self.data[index, ...]
        sample = torch.Tensor(sample)
        label = self.label[index]
        return sample, label

    def __len__(self):
        return len(self.label)
