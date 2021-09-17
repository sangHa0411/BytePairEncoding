import numpy as np
import random 
import torch
from torch.utils.data import Dataset, random_split

class NgramDataset :
    def __init__(self, v_size, w_size) :
        self.v_size = v_size
        self.w_size = w_size

    def get_data(self, idx_data) :
        co_occ = np.zeros((self.v_size, self.v_size))
        mid_point = int(self.window_size/2)

        ngram_data = []
        for i in range(len(idx_data)) :
            idx_list = idx_data[i]
            if len(idx_data) < self.w_size :
                continue
            for j in range(len(idx_list)-self.w_size) :
                sub_list = idx_list[j:j+self.w_size]
                ngram_data.append(sub_list)

        random.shuffle(ngram_data)

        ngram_array = np.array(ngram_data)
        cen_array = ngram_array[:, mid_point]
        con_array = np.hstack([ngram_array[:, mid_point-1], ngram_array[:,mid_point+1:]])

        return cen_array, con_array
    
class SkipGramDataset(Dataset) :
    def __init__(self, cen_array, con_array, val_ratio=0.1) :
        super(SkipGramDataset , self).__init__()
        assert len(cen_array) == len(con_array)
        self.cen_array = cen_array
        self.con_array = con_array
        self.val_ratio = val_ratio

    def __len__(self) :
        return len(self.con_array)

    def __getitem__(self , idx) :
        cen_idx = self.cen_array[idx]
        con_idx = self.con_array[idx]

        return cen_idx, con_idx

    def split(self) :
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        
        return train_set, val_set
