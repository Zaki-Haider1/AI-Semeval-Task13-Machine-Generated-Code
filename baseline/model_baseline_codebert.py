import torch
import torch.nn as nn
from transformers import AutoModel

class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.fc(self.dropout(pooled_output))
