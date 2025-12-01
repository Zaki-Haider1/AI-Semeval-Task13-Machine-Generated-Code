'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, num_classes=2, kernel_sizes=[3,4,5], num_filters=100, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        conv_outs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [(batch, num_filters, seq_len-k+1), ...]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # [(batch, num_filters), ...]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_classes=2, 
                 kernel_sizes=[1, 2, 3, 4, 5], num_filters=256, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        
        # Convolution + ReLU + MaxPool
        # We must ensure seq_len >= max(kernel_sizes) to avoid errors
        # If input is too short, we rely on padding
        conv_outs = []
        for conv in self.convs:
            out = F.relu(conv(emb)).squeeze(3) # [batch, num_filters, seq_len-k+1]
            out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch, num_filters]
            conv_outs.append(out)
            
        cat = torch.cat(conv_outs, dim=1) # [batch, num_filters * len(kernels)]
        return self.fc(self.dropout(cat))