
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #to convert sparse one hot encodings to dense data        
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        
        #to extract n-gram models depending on the filter sizes
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #applying max pooling to extract the most important featues from different n-gram models           
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        #concatenating maxed pool data together
        out = self.dropout(torch.cat(pooled, dim = 1))
   
        return self.fc(out)

