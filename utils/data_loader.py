import logging
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from .preprocess import preprocess_tfidf_pyTorch, preprocess_tokenized, preprocess_codebert, preprocess_tfidf

logger = logging.getLogger(__name__)


def loadData(mode="tfidf", batch_size=32, max_features=2000, sample_size=None):
    """Load data in several modes.

    Modes:
      - 'tfidf'         : returns ((X_train,y_train),(X_val,y_val), vectorizer) where X are scipy sparse matrices + numpy labels (for sklearn)
      - 'tfidf-pytorch' : returns (train_loader, val_loader, vectorizer) converting sparse->dense only when safe
      - 'tokenization'  : returns (train_loader, val_loader) for token models (per-batch padding)
      - 'codeBert'      : returns (train_loader, val_loader) for CodeBERT encodings

    Use `sample_size` for quick debugging runs on a subset.
    """

    train_path = "data/task_a_training_Set_1.parquet"
    val_path = "data/task_a_validation_set.parquet"

    def _safe_preprocess_tfidf(train_path, val_path, max_features, sample_size):
        """Call the TF-IDF preprocess function with kwargs; fall back if signature differs."""
        try:
            return preprocess_tfidf_pyTorch(train_path, val_path, max_features=max_features, sample_size=sample_size)
        except TypeError as e:
            logger.warning(f"preprocess_tfidf_pyTorch signature mismatch: {e}; falling back to preprocess_tfidf")
            try:
                return preprocess_tfidf(train_path, val_path, max_features=max_features, sample_size=sample_size)
            except Exception as e2:
                logger.error(f"Fallback preprocess_tfidf failed: {e2}")
                raise

    if mode == "tfidf":
        logger.info("Starting TF-IDF (sparse) preprocessing")
        t0 = time.perf_counter()
        (X_train, y_train), (X_val, y_val), vectorizer = _safe_preprocess_tfidf(
            train_path, val_path, max_features, sample_size
        )
        elapsed = time.perf_counter() - t0
        try:
            nnz = getattr(X_train, 'nnz', None)
            logger.info(f"TF-IDF sparse shapes: train={X_train.shape}, val={X_val.shape}, nnz={nnz}; elapsed={elapsed:.2f}s")
        except Exception:
            logger.info(f"TF-IDF preprocessing done; elapsed={elapsed:.2f}s")
        return (X_train, y_train), (X_val, y_val), vectorizer

    if mode == "tfidf-pytorch":
        logger.info("Starting TF-IDF preprocessing for PyTorch (will densify if safe)")
        t0 = time.perf_counter()
        (X_train, y_train), (X_val, y_val), vectorizer = _safe_preprocess_tfidf(
            train_path, val_path, max_features, sample_size
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"TF-IDF loaded (sparse) in {elapsed:.2f}s; shapes: {X_train.shape} (train), {X_val.shape} (val)")

        # sanity check before densifying
        n_rows, n_cols = X_train.shape
        total_elements = int(n_rows) * int(n_cols)
        # safety threshold (elements). Adjust if needed.
        if total_elements > 50_000_000 and sample_size is None:
            raise MemoryError(
                f"TF-IDF dense conversion would allocate {total_elements} elements. "
                "Reduce `max_features` or use `sample_size` to limit data before converting to dense."
            )

        # convert (sparse) to dense
        t1 = time.perf_counter()
        X_train_dense = torch.tensor(X_train.toarray(), dtype=torch.float32)
        X_val_dense = torch.tensor(X_val.toarray(), dtype=torch.float32)
        logger.info(f"Densified TF-IDF to tensors in {time.perf_counter()-t1:.2f}s")

        train_dataset = TensorDataset(X_train_dense, torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(X_val_dense, torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, vectorizer

    if mode == "tokenization":
        logger.info("Starting tokenization preprocessing")
        t0 = time.perf_counter()
        # preprocess_tokenized already returns DataLoaders and tokenizer
        train_loader, val_loader, tokenizer = preprocess_tokenized(
            train_path, val_path, batch_size=batch_size, sample_size=sample_size
        )
        logger.info(f"Tokenization done in {time.perf_counter()-t0:.2f}s; train_size={len(train_loader.dataset)}, val_size={len(val_loader.dataset)}, vocab={len(tokenizer.vocab)}")
        # attach tokenizer for downstream use (vocab size)
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
        logger.info(f"CodeBERT preprocessing done in {time.perf_counter()-t0:.2f}s; train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        return train_loader, val_loader

    raise ValueError(f"Unknown data mode: {mode}")


