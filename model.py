import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module) :
    def __init__(self, em_size, v_size, w_size) :
        super(SkipGram, self).__init__()
        self.em_size = em_size
        self.v_size = v_size
        self.w_size = w_size
        
        self.em_layer = nn.Embedding(num_embeddings = v_size,
            embedding_dim = em_size,
            padding_idx = 0
        )
        self.o_layer = nn.Linear(em_size, v_size*(w_size-1))

        self.init_param()
        
    def init_param(self) :
        nn.init.normal_(self.em_layer.weight, mean=0.0, std=0.1)
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
       
    def forward(self, in_tensor) :
        in_tensor = in_tensor.unsqueeze(1)
        em_tensor = self.em_layer(in_tensor)
        o_tensor = self.o_layer(em_tensor)
        o_tensor = torch.reshape(o_tensor, (-1,self.w_size-1,self.v_size))
        return o_tensor

    def get_weight(self) :
        em_weight = self.em_layer.weight
        o_weight = self.o_layer.weight
        o_weight = torch.reshape(o_weight, (self.v_size, self.w_size-1, self.em_size))
        o_weight = torch.mean(o_weight, dim=1)
        return (em_weight + o_weight) / 2


    def get_bias(self) :
        o_bias = self.o_layer.bias
        o_bias = torch.reshape(o_bias, (self.v_size, self.w_size-1))
        o_bias = torch.mean(o_bias, dim=1)
        return o_bias