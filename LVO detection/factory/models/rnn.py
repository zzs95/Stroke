import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class SpecialRNN(nn.Module):


    def __init__(self, 
                 rnn_module,
                 embed_size,
                 hidden_size,
                 num_classes,
                 dropout):

        super().__init__()
        
        self.embedding_dropout = SpatialDropout(dropout)
        self.embed_size = embed_size 

        self.rnn1 = rnn_module(embed_size,    hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = rnn_module(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(hidden_size*2, hidden_size*2)
        self.linear2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.linear  = nn.Linear(hidden_size*2, num_classes)


    def forward(self, x):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
        h_rnn1, _ = self.rnn1(h_embedding)
        h_rnn2, _ = self.rnn2(h_rnn1)
        
        h_conc_linear1  = F.relu(self.linear1(h_rnn1))
        h_conc_linear2  = F.relu(self.linear2(h_rnn2))
        
        hidden = h_rnn1 + h_rnn2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)
        
        return output