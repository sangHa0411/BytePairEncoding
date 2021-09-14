import torch
import torch.nn as nn
import torch.nn.functional as F

class Glove(nn.Module) :
    def __init__(self, em_size, v_size) :
        super(Glove, self).__init__()
        self.em_size = em_size
        self.v_size = v_size
        
        self.con_em = torch.nn.Embedding(v_size,em_size,0)
        self.con_b = torch.nn.Embedding(v_size, 1, 0)
        self.tar_em = torch.nn.Embedding(v_size,em_size,0)
        self.tar_b = torch.nn.Embedding(v_size, 1, 0)

        self.init_param()
        
    def init_param(self) :
        for m in self.modules() :
            if isinstance(m, nn.Embedding) :
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
       
    # con_tensor, tar_tensor : (batch_size, 1)
    def forward(self, con_tensor, tar_tensor) :
        con_tensor = con_tensor.unsqueeze(1)
        tar_tensor = tar_tensor.unsqueeze(1)
        
        con_em = self.con_em(con_tensor) # context embedding
        tar_em = self.tar_em(tar_tensor) # target embedding
        con_b = self.con_b(con_tensor) # context bias
        tar_b = self.tar_b(tar_tensor) # target bias

        tar_em_T = torch.transpose(tar_em, 1, 2) # (batch_size, embedding_dim, 1)
        w = torch.matmul(con_em, tar_em_T) # (batch_size, 1, 1)

        o_tensor = w + con_b + tar_b # (batch_size, 1, 1)
        o_tensor = torch.reshape(o_tensor, (-1,))
        return o_tensor
