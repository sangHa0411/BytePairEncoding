import numpy as np
import random 
import torch
from torch.utils.data import Dataset, DataLoader , Subset, random_split

class NgramDataset :
    def __init__(self, window_size) :
        self.window_size = window_size

    def get_data(self, idx_data) :
        context_data = []
        for i in range(len(idx_data)) :
            idx_list = idx_data[i]
            if len(idx_list) < self.window_size :
                continue

            for j in range(len(idx_list)-self.window_size) :
                context_data.append(idx_list[j:j+self.window_size])

        random.shuffle(context_data)
        context_data = np.array(context_data)
        
        mid_point = int(self.window_size/2)
        cen_data = context_data[:,mid_point]
        neighbor_data = np.hstack([context_data[:,:mid_point],
                                   context_data[:,mid_point+1:]])

        return cen_data, neighbor_data
    
class EmbeddingDataset(Dataset) :
    def __init__(self, cen_data, neighbor_data, val_ratio=0.1) :
        super(EmbeddingDataset , self).__init__()
        self.c_data = cen_data
        self.n_data = neighbor_data
        self.val_ratio = val_ratio

    def __len__(self) :
        return len(self.c_data)

    def __getitem__(self , idx) :
        return self.c_data[idx], self.n_data[idx]
    
    def split(self) :
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        
        return train_set, val_set