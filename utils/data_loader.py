import torch
from torch.utils.data import DataLoader
from .preprocess import preprocess_tfidf_pyTorch, preprocess_tokenized, preprocess_codebert

def loadData(mode="tfidf-pytorch", batch_size=32, max_features=5000):
    if mode == "tfidf-pytorch":
        train_tuple, val_tuple, vectorizer = preprocess_tfidf_pyTorch(
            "data/task_a_training_Set_1.parquet",
            "data/task_a_validation_set.parquet",
            max_features=max_features
        )
        return train_tuple, val_tuple, vectorizer

    elif mode == "tokenization":
        train_loader, val_loader, tokenizer = preprocess_tokenized(
            "data/task_a_training_Set_1.parquet",
            "data/task_a_validation_set.parquet",
            batch_size=batch_size
        )
        return train_loader, val_loader

    elif mode == "codeBert":
        train_loader, val_loader = preprocess_codebert(
            "data/task_a_training_Set_1.parquet",
            "data/task_a_validation_set.parquet"
        )
        return train_loader, val_loader
