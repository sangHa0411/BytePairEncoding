import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module) :
    def __init__(self, em_size, v_size, window_size) :
        super(SkipGram, self).__init__()
        self.em_size = em_size
        self.v_size = v_size
        self.window_size = window_size
    
        self.embedding = nn.Embedding(num_embeddings=v_size,
                                      embedding_dim=em_size,
                                      padding_idx=0)
        self.o_layer = nn.Linear(em_size, v_size*(window_size-1))
        
        self.init_param()
        
    def init_param(self) :
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        nn.init.xavier_normal_(self.o_layer.weight)
        nn.init.zeros_(self.o_layer.bias)
        
    def forward(self, in_tensor) :
        in_tensor = in_tensor.unsqueeze(1)
        em_tensor = self.embedding(in_tensor)
        
        o_tensor = self.o_layer(em_tensor)
        o_tensor = torch.reshape(o_tensor, (-1,self.window_size-1,self.v_size))
        
        return o_tensor
