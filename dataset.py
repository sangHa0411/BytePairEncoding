import numpy as np
import random 
import torch
from torch.utils.data import Dataset, DataLoader , Subset, random_split

class NgramDataset :
    def __init__(self, v_size, window_size) :
        self.v_size = v_size
        self.window_size = window_size

    def get_data(self, idx_data) :
        co_occ = np.zeros((self.v_size, self.v_size))
        mid_point = int(self.window_size/2)

        for i in range(idx_data) :
            idx_list = idx_data[i]
            idx_len = len(idx_list)
            if idx_len < self.window_size :
                continue

            for j in range(len(idx_list)-self.window_size) :
                sub_list = idx_list[j:j+self.window_size]
                center = sub_list[mid_point]
                context = sub_list[:mid_point] + sub_list[mid_point+1:] 
                co_occ[center][context] += 1

        con_data = []
        tar_data = []
        occ_data = []

        for i in range(self.v_size) :
            for j in range(self.v_size) :
                if co_occ[i,j] > 0 :
                    con_data.append(i)
                    tar_data.append(j)
                    occ_data.append(co_occ[i,j])

        return con_data, tar_data, occ_data
    
class GloveDataset(Dataset) :
    def __init__(self, con_data, tar_data, occ_data, val_ratio=0.1) :
        super(GloveDataset , self).__init__()
        assert (len(con_data) == len(tar_data)) and (len(con_data) == len(occ_data))
        self.con_data = con_data
        self.tar_data = tar_data
        self.occ_data = occ_data
        self.val_ratio = val_ratio

    def __len__(self) :
        return len(self.con_data)

    def __getitem__(self , idx) :
        con_idx = self.con_data[idx]
        tar_idx = self.tar_data[idx]
        occ_idx = self.occ_data[idx]

        return {'con' : con_idx, 'tar' : tar_idx, 'occ' : occ_idx}

    def split(self) :
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        
        return train_set, val_set
