import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np



# -----------------------------
# Common utilities
# -----------------------------
def load_parquet(path):
    """Reads a parquet file and returns dataframe."""
    return pd.read_parquet(path)

def extract_xy(df):
    """
    df: DataFrame with columns ["code", "label"]
    returns: list(texts), list(labels)
    """
    texts = df["code"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


# =============================
# TF-IDF preprocessing
# FOR SKLEARN
# =============================

def fit_vectorizer(train_texts, max_features=20000):
    """Fits TF-IDF on training texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b"   # handles code tokens better
    )
    vectorizer.fit(train_texts)
    return vectorizer

def transform_texts(vectorizer, texts):
    """Transforms texts → TF-IDF and returns sparse matrix (no dense conversion)."""    
    return vectorizer.transform(texts)


def preprocess_tfidf(train_path, val_path, save_dir="vectorizer", max_features=20000, sample_size=None):
    """
    Load train/val parquet → TF-IDF → NumPy arrays.
    Saves vectorizer for future inference.
    """
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)

    # optionally sample for quick iteration
    if sample_size is not None:
       train_df = train_df.head(sample_size)
       val_df = val_df.head(sample_size)

    train_texts, train_labels = extract_xy(train_df)
    val_texts, val_labels = extract_xy(val_df)

    vectorizer = fit_vectorizer(train_texts, max_features=max_features)

    # Save vectorizer
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{save_dir}/tfidf.pkl")

    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # Return in the ((X_train,y_train),(X_val,y_val), vectorizer) shape
    return (X_train, y_train), (X_val, y_val), vectorizer

# =============================
# TF-IDF preprocessing
# FOR PYTORCHHH
# =============================
'''
def preprocess_tfidf_pyTorch(train_path, val_path, save_dir="vectorizer", max_features=20000, sample_size=None):
    """
    Load train/val parquet → TF-IDF → tensors.
    Saves vectorizer for future inference.
    """
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)

    train_texts, train_labels = extract_xy(train_df)
    val_texts, val_labels = extract_xy(val_df)

    # optionally sample for quick iteration
    if sample_size is not None:
        train_texts = train_texts[:sample_size]
        train_labels = train_labels[:sample_size]
        val_texts = val_texts[:sample_size]
        val_labels = val_labels[:sample_size]

    vectorizer = fit_vectorizer(train_texts, max_features=max_features)

    # Save vectorizer
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{save_dir}/tfidf.pkl")

    X_train = transform_texts(vectorizer, train_texts)
    X_val = transform_texts(vectorizer, val_texts)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    # Return as sparse matrices + numpy labels for the loader to handle
    return (X_train, y_train), (X_val, y_val), vectorizer
'''


# =============================
# Tokenization preprocessing (for TextCNN / LSTM)
# PYTORCH
# =============================
class Tokenizer:
    """Simple word-level tokenizer with vocab building."""
    def __init__(self, min_freq=1):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.freq = {}
        self.min_freq = min_freq

    def build_vocab(self, texts):
        """Build vocabulary from list of strings"""
        for text in texts:
            for tok in text.split():  # simple whitespace tokenizer
                self.freq[tok] = self.freq.get(tok, 0) + 1
        for tok, count in self.freq.items():
            if count >= self.min_freq and tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self.vocab

    def text_to_ids(self, text):
        """Convert a text string to list of token IDs"""
        return torch.tensor([self.vocab.get(tok, self.vocab["<UNK>"]) for tok in text.split()],
                            dtype=torch.long)

def tokenize_and_pad(texts, tokenizer):
    """Tokenize and pad a list of texts"""
    seqs = [tokenizer.text_to_ids(t) for t in texts]
    padded = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.vocab["<PAD>"])
    return padded

def preprocess_tokenized(train_path, val_path, batch_size=32, min_freq=1, sample_size=None):
    """
    Reads parquet train + val files, builds vocab on train, tokenizes sequences and
    returns PyTorch DataLoaders with per-batch padding. Also returns the tokenizer.

    Returns: train_loader, val_loader, tokenizer
    """
    # load dataframes
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)

    train_texts, train_labels = extract_xy(train_df)
    val_texts, val_labels = extract_xy(val_df)

    # optionally sample for quick iteration
    if sample_size is not None:
        train_texts = train_texts[:sample_size]
        train_labels = train_labels[:sample_size]
        val_texts = val_texts[:sample_size]
        val_labels = val_labels[:sample_size]

    tokenizer = Tokenizer(min_freq=min_freq)
    tokenizer.build_vocab(train_texts)

    # dataset that returns variable-length tensors
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            t = self.texts[idx]
            ids = self.tokenizer.text_to_ids(t)
            lbl = int(self.labels[idx])
            return ids, torch.tensor(lbl, dtype=torch.long)

    def collate_fn(batch):
        seqs, labs = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.vocab["<PAD>"])
        labels = torch.stack(labs)
        return padded, labels

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, tokenizer


# CODE BERT TOKENIZATION 
# FOR PYTORCH

def preprocess_codebert(train_path, val_path, batch_size=32, max_length=256, sample_size=None):
    # Lazy import to avoid heavy transformers import at module import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def load_data(path):
        df = pd.read_parquet(path)
        texts = df["code"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        # optionally sample for quick iteration
        if sample_size is not None:
            texts = texts[:sample_size]
            labels = labels[:sample_size]

        # encode and return tensors
        encodings = tokenizer(texts, truncation=True, padding='max_length',
                              max_length=max_length, return_tensors='pt')
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return encodings, labels_tensor

    train_encodings, y_train = load_data(train_path)
    val_encodings, y_val = load_data(val_path)

    # Create TensorDataset
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    val_dataset   = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val)

    # Create DataLoaders so callers get batches (consistent with tokenization loader)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer
