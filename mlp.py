import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url



class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_dim = 3*32*32
        self.featurevector = None
        self.mlp = nn.Sequential(
            nn.Linear(1000*2+self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    def init_hidden(self,hidden):
        first = hidden.shape[0]
        #print('hidden',hidden.shape)
        self.featurevector = hidden.reshape(first,-1)

        return (None, None)


    def forward(self, x):
        #print(self.featurevector.shape)
        first = x.shape[0]
        x = x.reshape(first,-1)
        #print(x.shape)
        
        x = torch.cat([x,self.featurevector],1)
        assert x.shape == (first,2000+self.hidden_dim)
        return self.mlp(x)