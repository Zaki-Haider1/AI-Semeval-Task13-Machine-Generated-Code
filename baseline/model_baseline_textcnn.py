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
