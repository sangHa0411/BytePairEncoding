import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module) :
    def __init__(self, em_size, v_size) :
        super(SkipGram, self).__init__()
        self.em_size = em_size
        self.v_size = v_size
        
        self.em_layer = nn.Embedding(num_embeddings = v_size,
            embedding_dim = em_size,
            padding_idx = 0
        )
        self.o_layer = nn.Linear(em_size, v_size)

        self.init_param()
        
    def init_param(self) :
        nn.init.normal_(self.em_layer.weight, mean=0.0, std=1)
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.weight)
       
    def forward(self, in_tensor) :
        in_tensor = in_tensor.unsqueeze(1)
        em_tensor = self.em_layer(in_tensor).squeeze(1)
        o_tensor = self.o_layer(em_tensor)
        return o_tensor

    def get_weight(self) :
        return (self.em_layer.weight + self.o_layer.weight)/2

    def get_bias(self) :
        return self.o_layer.bias

class CBOW(nn.Module) :
    def __init__(self, em_size, v_size) :
        super(CBOW, self).__init__()
        self.em_size = em_size
        self.v_size = v_size

        self.em_layer = nn.Embedding(num_embeddings = v_size,
            embedding_dim = em_size,
            padding_idx = 0
        )
        self.o_layer = nn.Linear(em_size, v_size)

        self.init_param()

    def init_param(self) :
        nn.init.normal_(self.em_layer.weight, mean=0.0, std=1)
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.weight)

    def forward(self, in_tensor) :
        em_tensor = self.em_layer(in_tensor)
        h_tensor = torch.mean(em_tensor, dim=1)
        o_tensor = self.o_layer(h_tensor)
        return o_tensor

    def get_weight(self) :
        return (self.em_layer.weight + self.o_layer.weight)/2

    def get_bias(self) :
        return self.o_layer.bias

