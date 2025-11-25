import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import os
from transformers import AutoTokenizer

# -------------------------- Utilities --------------------------
def load_parquet(path):
    return pd.read_parquet(path)

def extract_xy(df):
    texts = df["code"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels

# -------------------------- TF-IDF --------------------------
def fit_vectorizer(train_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(train_texts)
    return vectorizer

def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)  # keep sparse for memory

def preprocess_tfidf_pyTorch(train_path, val_path, save_dir="vectorizer", max_features=5000):
    train_texts, train_labels = extract_xy(load_parquet(train_path))
    val_texts, val_labels = extract_xy(load_parquet(val_path))

    vectorizer = fit_vectorizer(train_texts, max_features=max_features)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{save_dir}/tfidf.pkl")

    X_train = transform_texts(vectorizer, train_texts)
    X_val   = transform_texts(vectorizer, val_texts)

    y_train = np.array(train_labels, dtype=np.int64)
    y_val   = np.array(val_labels, dtype=np.int64)

    return (X_train, y_train), (X_val, y_val), vectorizer

# -------------------------- Tokenization --------------------------
class Tokenizer:
    def __init__(self, min_freq=1):
        self.vocab = {"<PAD>":0, "<UNK>":1}
        self.freq = {}
        self.min_freq = min_freq

    def build_vocab(self, texts):
        for t in texts:
            for tok in t.split():
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, count in self.freq.items():
            if count >= self.min_freq and tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self.vocab

class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = torch.tensor([self.tokenizer.vocab.get(tok, self.tokenizer.vocab["<UNK>"]) for tok in self.texts[idx].split()], dtype=torch.long)
        return tokens, self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences, labels

def preprocess_tokenized(train_path, val_path, min_freq=1, batch_size=32):
    train_texts, train_labels = extract_xy(load_parquet(train_path))
    val_texts, val_labels = extract_xy(load_parquet(val_path))

    tokenizer = Tokenizer(min_freq=min_freq)
    tokenizer.build_vocab(train_texts)

    train_dataset = CodeDataset(train_texts, train_labels, tokenizer)
    val_dataset   = CodeDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, tokenizer

# -------------------------- CodeBERT --------------------------
def preprocess_codebert(train_path, val_path, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def load_data(path):
        df = pd.read_parquet(path)
        texts = df["code"].astype(str).tolist()
        labels = torch.tensor(df["label"].astype(int).tolist(), dtype=torch.long)
        encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        return encodings, labels

    train_encodings, y_train = load_data(train_path)
    val_encodings, y_val = load_data(val_path)

    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    val_dataset   = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val)

    return train_dataset, val_dataset
