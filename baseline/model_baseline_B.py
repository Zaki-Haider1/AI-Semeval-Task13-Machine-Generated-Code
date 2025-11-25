import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout_rate=0.6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout_rate if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(emb)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(combined))
