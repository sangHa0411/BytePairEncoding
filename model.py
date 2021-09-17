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
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
       
    def forward(self, in_tensor) :
        in_tensor = in_tensor.unsqueeze(1)
        em_tensor = self.em_layer(in_tensor)
        o_tensor = self.o_layer(em_tensor)
        o_tensor = torch.reshape(o_tensor, (-1,self.w_size-1,self.v_size))

        return o_tensor
