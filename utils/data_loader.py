import logging
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from .preprocess import preprocess_tokenized, preprocess_codebert, preprocess_tfidf

logger = logging.getLogger(__name__)


# ========================= OLD CODE BLOCKS (SAFE, OUTSIDE FUNCTION) =========================

'''
def _safe_preprocess_tfidf(train_path, val_path, max_features, sample_size):
    """Call the TF-IDF preprocess function with kwargs; fall back if signature differs."""
    try:
        return preprocess_tfidf_pyTorch(
            train_path, val_path,
            max_features=max_features,
            sample_size=sample_size
        )
    except TypeError as e:
        logger.warning(f"Signature mismatch: {e}; falling back to preprocess_tfidf")
        try:
            return preprocess_tfidf(
                train_path, val_path,
                max_features=max_features,
                sample_size=sample_size
            )
        except Exception as e2:
            logger.error(f"Fallback preprocess_tfidf failed: {e2}")
            raise
'''

'''
if mode == "tfidf":
    logger.info("Starting TF-IDF (sparse) preprocessing")
    t0 = time.perf_counter()
    (X_train, y_train), (X_val, y_val), vectorizer = _safe_preprocess_tfidf(
        train_path, val_path, max_features, sample_size
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"TF-IDF done in {elapsed:.2f}s")
    return (X_train, y_train), (X_val, y_val), vectorizer
'''

'''
if mode == "tfidf-pytorch":
    logger.info("Starting TF-IDF (PyTorch) preprocessing")
    t0 = time.perf_counter()
    (X_train, y_train), (X_val, y_val), vectorizer = _safe_preprocess_tfidf(
        train_path, val_path, max_features, sample_size
    )
    elapsed = time.perf_counter() - t0

    n_rows, n_cols = X_train.shape
    if n_rows * n_cols > 50_000_000:
        raise MemoryError("Too large to convert to dense!")

    X_train_dense = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_dense = torch.tensor(X_val.toarray(), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_dense, torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(X_val_dense, torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vectorizer
'''

# =============================================================================================



def loadData(mode="tfidf", batch_size=32, max_features=2000, sample_size=None):
    """Load data in several modes:
      - tfidf
      - tfidf-pytorch
      - tokenization
      - codeBert
    """

    train_path = "data/task_a_training_Set_1.parquet"
   #val_path = "data/task_a_validation_set.parquet"
    val_path = "data/task_a_test_set_sample.parquet"

    # ====================== ACTIVE WORKING CODE ======================

    if mode == "tfidf-sklearn":
        (X_train, y_train), (X_val, y_val), vectorizer = preprocess_tfidf(
            train_path, val_path, max_features=max_features, sample_size=sample_size
        )
        return (X_train, y_train), (X_val, y_val), vectorizer

    if mode == "tokenization":
        logger.info("Starting tokenization preprocessing")
        t0 = time.perf_counter()

        train_loader, val_loader, tokenizer = preprocess_tokenized(
            train_path, val_path, batch_size=batch_size, sample_size=sample_size
        )

        logger.info(
            f"Tokenization done in {time.perf_counter()-t0:.2f}s; "
            f"train_size={len(train_loader.dataset)}, "
            f"val_size={len(val_loader.dataset)}, "
            f"vocab={len(tokenizer.vocab)}"
        )

        try:
            train_loader.dataset.tokenizer = tokenizer
            val_loader.dataset.tokenizer = tokenizer
        except Exception:
            pass

        return train_loader, val_loader

    if mode == "codeBert":
        logger.info("Starting CodeBERT preprocessing")
        t0 = time.perf_counter()

        train_loader, val_loader = preprocess_codebert(
            train_path, val_path, batch_size=batch_size, sample_size=sample_size
        )

        logger.info(
            f"CodeBERT preprocessing done in {time.perf_counter()-t0:.2f}s; "
            f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
        )

        return train_loader, val_loader

    raise ValueError(f"Unknown data mode: {mode}")

