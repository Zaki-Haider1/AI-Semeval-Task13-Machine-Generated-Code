# train.py (Unified Baseline Trainer - Optimized, Timed, Device Info)
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.data_loader import loadData
from utils.evaluate import evaluate_and_plot

from baseline.model_baseline_B import BiLSTMClassifier
from baseline.model_baseline_codebert import CodeBERTClassifier
from baseline.model_baseline_svm import SVMClassifier
from baseline.model_baseline_textcnn import TextCNNClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



from sklearn.metrics import f1_score
import numpy as np





### TIME ESTIMATION UTILITIES ###

def estimate_sklearn_time(model, X_train, y_train):
    """Estimate sklearn model fit+predict time by timing on a small sample and scaling.
    Returns estimated seconds.
    """
    import time
    # Use two sample sizes to estimate a power-law scaling (time ~ n^p).
    # This captures super-linear behavior (SVC) better than a single-point linear extrapolation.
    try:
        n = X_train.shape[0]
    except Exception:
        n = len(y_train)

    if n <= 0:
        return 60.0

    # determine two sample sizes (small and medium), capped to reasonable values
    s1 = min(200, max(20, int(0.001 * n)))
    s2 = min(2000, max(s1 + 1, int(0.01 * n)))
    s1 = max(10, s1)
    s2 = min(n, max(s2, s1 + 1))

    # helper to time fit+predict on a sample using a fresh clone of the model
    def _time_on_sample(m, Xs, ys):
        t0 = time.time()
        try:
            from sklearn.base import clone
            m_clone = clone(m)
        except Exception:
            m_clone = m
        try:
            m_clone.fit(Xs, ys)
            _ = m_clone.predict(Xs)
        except Exception:
            return None
        return time.time() - t0

    # take samples (slice; works for numpy/scipy sparse)
    try:
        X1 = X_train[:s1]
        y1 = y_train[:s1]
        X2 = X_train[:s2]
        y2 = y_train[:s2]
    except Exception:
        # fallback to small single-sample strategy
        try:
            X1 = X_train
            y1 = y_train
        except Exception:
            return 60.0
        t1 = _time_on_sample(model, X1, y1)
        if t1 is None:
            return 60.0
        return t1 * (n / max(1, len(y1))) * 1.2

    t1 = _time_on_sample(model, X1, y1)
    t2 = _time_on_sample(model, X2, y2)

    # if either timing failed, fall back to single-sample linear extrapolation
    if (t1 is None) or (t2 is None) or t1 <= 0 or t2 <= 0:
        # single-sample fallback
        try:
            sample_n = max(20, int(0.01 * n))
            Xs = X_train[:sample_n]
            ys = y_train[:sample_n]
            t0 = time.time()
            from sklearn.base import clone
            m_clone = clone(model)
            m_clone.fit(Xs, ys)
            _ = m_clone.predict(Xs)
            elapsed = time.time() - t0
            est_seconds = elapsed * (n / max(1, sample_n))
            return est_seconds * 1.2
        except Exception:
            return 120.0

    # fit power law t ~ c * n^p  => p = log(t2/t1)/log(n2/n1)
    try:
        import math
        p = math.log(t2 / t1) / math.log(s2 / s1)
    except Exception:
        p = 1.0

    # heuristic: SVC-like solvers often super-linear; clamp p to reasonable range
    if p < 0.5:
        p = 1.0
    if p > 3.0:
        p = 3.0

    # If model appears to be an SVC (slow), bias towards higher exponent if p is small
    try:
        model_name = type(model).__name__
        if 'SVC' in model_name or 'SVR' in model_name:
            p = max(p, 1.5)
    except Exception:
        pass

    # extrapolate from larger sample t2
    est_seconds = t2 * (n / max(1, s2)) ** p

    # conservative buffer
    return est_seconds * 1.25



def estimate_pytorch_time(model_factory, train_loader, device, epochs=1, max_batches=5):
    """Estimate PyTorch training time by running a few batches and scaling.
    model_factory: callable -> model instance already moved to device
    """
    import time
    try:
        it = iter(train_loader)
        batch = next(it)
    except Exception:
        return 60.0

    # Build model and optimizer
    try:
        model = model_factory().to(device)
    except Exception:
        try:
            model = model_factory
            model.to(device)
        except Exception:
            return 60.0

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # run up to max_batches training steps
    batches = [batch]
    for i in range(max_batches - 1):
        try:
            batches.append(next(it))
        except StopIteration:
            break

    t0 = time.time()
    model.train()
    for b in batches:
        try:
            optimizer.zero_grad()
            # unpack batch depending on shape
            if isinstance(b, (list, tuple)) and len(b) >= 2:
                inputs = b[0].to(device)
                labels = b[1].to(device)
            else:
                continue
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        except Exception:
            # if any op fails, return a default
            return 120.0

    elapsed = time.time() - t0

    # estimate total batches
    try:
        total_batches = len(train_loader)
    except Exception:
        # fallback guess
        total_batches = 100

    avg_per_batch = elapsed / max(1, len(batches))
    est_seconds = avg_per_batch * total_batches * epochs
    return est_seconds * 1.05
def _format_seconds(sec: float) -> str:
    if sec is None:
        return "unknown"
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{sec/60:.2f} mins"

### TIME ESTIMATION UTILITIES END###




def find_best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)

    best_f1 = -1
    best_t = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1



# ----------------------- OLD TRAINING LOOP -----------------------

'''

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-3))
        best_f1 = 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch')
            for sequences, labels in pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})







            # Validation
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            macro_f1 = evaluate_and_plot(val_true, val_preds, output_dir, num_labels=num_labels, model_name=model_name)
            logger.info(f"Epoch {epoch} finished. Macro F1: {macro_f1:.4f}. Time: {(time.time()-epoch_start)/60:.2f} mins")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best_model.pth'))

        total_time = time.time() - start_time
        logger.info(f"[{model_name}] Training complete. Best Macro F1: {best_f1:.4f}. Total time: {total_time/60:.2f} mins")
        return best_f1

'''
#-------------------- OLD TRIANIN LOOOOOP ENDS HERE --------------------


# ---------------- MODEL REGISTRY ----------------
MODEL_REGISTRY = {
    'A': LogisticRegression(max_iter=2000, n_jobs=-1,class_weight='balanced'),
    'B': BiLSTMClassifier,
    'C': CodeBERTClassifier,
    'D': MultinomialNB(),
    'E': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'F': SVMClassifier(),
    'G': TextCNNClassifier,
}

# ---------------- UNIFIED TRAIN FUNCTION ----------------
def train_model(model_name, train_loader, val_loader, output_dir, num_labels=None, **kwargs):
    start_time = time.time()
    logger.info(f"\nTraining model: {model_name}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name in ['B', 'G', 'C']:
        logger.info(f"Using device: {device}")

    # ---------------- PyTorch token models: BiLSTM/TextCNN ----------------
    if model_name in ['B', 'G']:
        vocab_size = kwargs.get('vocab_size')
        num_labels = kwargs.get('num_labels', 2)
        num_epochs = kwargs.get('epochs', 5)

        if model_name == 'B':
            model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_dim=128,
                num_layers=2,
                num_classes=num_labels
            ).to(device)
      
      #  elif model_name == 'G':
       #     model = TextCNNClassifier(
        #        vocab_size=vocab_size,
         #       embedding_dim=100,
          #      num_classes=num_labels
           # ).to(device)
        elif model_name == 'G':
            model = TextCNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=128,       # Increased from 100
                num_filters=256,         # Increased from 100
                kernel_sizes=[1,2,3,4,5],# Better for code
                num_classes=num_labels
            ).to(device)



# ------------------- NEW TRAINING LOOP -------------------

# 2. WEIGHTED LOSS (Crucial for Macro F1 on skewed data)
        # Assuming Training data is also skewed. If you don't know the exact count, 
        # a safe bet for Human(0)/Machine(1) is usually to weight the minority class higher.
        # If dataset is 50/50, remove the weight. If it's 75/25, weight=[1.0, 3.0]
        # Let's assume training is similar to test (77% human, 23% machine)
        #class_weights = torch.tensor([1.0, 3.0]).to(device) 
        #criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ---------------- OLD (Weighted) ----------------
        # class_weights = torch.tensor([1.0, 3.35]).to(device) 
        # criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ---------------- NEW (Balanced) ----------------
        # No weights needed!
        criterion = nn.CrossEntropyLoss()   

        # 3. OPTIMIZER & SCHEDULER
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_f1 = 0.0
        early_stop_count = 0
        patience = 4 # Stop if no improvement for 4 epochs

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch')
            
            for sequences, labels in pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Validation
            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    # Get probabilities using Softmax for threshold tuning
                    probs = F.softmax(outputs, dim=1)[:, 1] 
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            val_true = np.array(val_true)
            val_probs = np.array(val_probs)

            # 4. FIND BEST THRESHOLD (Maximizing Macro F1)
            # Instead of default 0.5, we scan for the best split
            thresholds = np.linspace(0.1, 0.9, 50)
            best_epoch_f1 = 0
            best_epoch_thresh = 0.5
            
            for t in thresholds:
                temp_preds = (val_probs >= t).astype(int)
                f1 = f1_score(val_true, temp_preds, average='macro')
                if f1 > best_epoch_f1:
                    best_epoch_f1 = f1
                    best_epoch_thresh = t

            # Log results
            logger.info(f"Epoch {epoch} finished. Best Thresh: {best_epoch_thresh:.2f}, Macro F1: {best_epoch_f1:.4f}")
            
            # Step the scheduler based on F1 score
            scheduler.step(best_epoch_f1)

            if best_epoch_f1 > best_f1:
                best_f1 = best_epoch_f1
                early_stop_count = 0
                torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best_model.pth'))
                # Also save the threshold to use during inference!
                with open(os.path.join(output_dir, f'{model_name}_threshold.txt'), 'w') as f:
                    f.write(str(best_epoch_thresh))
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    logger.info("Early stopping triggered.")
                    break
        
        return best_f1


# ------------------- NEW TRAINING LOOP -------------------



    # ---------------- CodeBERT (C) ----------------
    if model_name == 'C':
        num_labels = kwargs.get('num_labels', 2)
        num_epochs = kwargs.get('epochs', 3)
        model = CodeBERTClassifier(num_classes=num_labels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-5))
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()
            pbar = tqdm(train_loader, desc=f'CodeBERT Epoch {epoch}/{num_epochs}', unit='batch')
            for input_ids, attn, labels in pbar:
                input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attn)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Validation
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for input_ids, attn, labels in val_loader:
                    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attn)
                    _, predicted = torch.max(outputs, 1)
                    preds.extend(predicted.cpu().numpy())
                    trues.extend(labels.cpu().numpy())

            macro_f1 = evaluate_and_plot(trues, preds, output_dir, num_labels=num_labels, model_name="CodeBERT")
            logger.info(f"Epoch {epoch} finished. Macro F1: {macro_f1:.4f}. Time: {(time.time()-epoch_start)/60:.2f} mins")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), os.path.join(output_dir, 'codebert_best_model.pth'))

        return best_f1










    # ---------------- TFIDF ML MODELS ----------------
    # train_loader/val_loader expected as tuples: (X_train, y_train), (X_val, y_val)





    
    if isinstance(train_loader, tuple) and len(train_loader) == 2:
        X_train_vec, y_train_vec = train_loader
        X_val_vec, y_val_vec = val_loader
    else:
        # fallback: try to extract one batch
        try:
            X_train, y_train = next(iter(train_loader))
            X_val, y_val = next(iter(val_loader))
            X_train_vec, X_val_vec = X_train, X_val
            y_train_vec, y_val_vec = y_train, y_val
        except Exception as e:
            raise RuntimeError("Unable to unpack TF-IDF training data") from e





# ***************** THRESHOLD TUNING AND EVALUATION FOR TFIDF MODELS *****************

    model = MODEL_REGISTRY[model_name]
    logger.info(f"Training {model_name} (sparse TF-IDF ML model)...")
    start = time.time()

    # ---- TRAIN ----
    model.fit(X_train_vec, y_train_vec)

    # ---- FIND BEST THRESHOLD ON VALIDATION ----
    if hasattr(model, "predict_proba"):   # LogisticRegression, NB, RF, SVM(prob) etc.
      
      
        best_t, best_f1 = find_best_threshold(model, X_val_vec, y_val_vec)
        logger.info(f"Best threshold={best_t:.3f}  Best macro-F1={best_f1:.4f}")


        from sklearn.metrics import precision_recall_curve

        # ANOTHER WAY TO FIND BEST THRESHOLD
        # This is Giving a worse F1 than the above function

        #probs = model.predict_proba(X_val_vec)[:, 1]
        #precision, recall, thresholds = precision_recall_curve(y_val_vec, probs)
        #f1_scores = 2 * (precision * recall) / (precision + recall)
        #best_idx = np.argmax(f1_scores)
        #best_t = thresholds[best_idx]


        #manually setting threshold for testing
        #best_t = 0.4

        # Apply threshold
        val_probs = model.predict_proba(X_val_vec)[:, 1]
        y_pred = (val_probs >= best_t).astype(int)

    else:
        # models with no .predict_proba
        y_pred = model.predict(X_val_vec)
        best_t = None

    # ---- EVALUATE ----
    final_f1 = evaluate_and_plot(
        y_val_vec, y_pred, output_dir,
        num_labels=len(np.unique(np.concatenate((y_train_vec, y_val_vec)))),
        model_name=model_name
    )

    logger.info(f"[{model_name}] Training complete. Macro F1: {final_f1:.4f}. "
                f"Time: {(time.time()-start)/60:.2f} mins")
    return final_f1, best_t



# ***************** NORMAL EVALUATION FOR TFIDF MODELS *****************
'''
    model = MODEL_REGISTRY[model_name]
    logger.info(f"Training {model_name} (sparse TF-IDF ML model)...")
    start = time.time()
    model.fit(X_train_vec, y_train_vec)
    y_pred = model.predict(X_val_vec)
    final_f1 = evaluate_and_plot(y_val_vec, y_pred, output_dir,
                                 num_labels=len(np.unique(np.concatenate((y_train_vec, y_val_vec)))),
                                 model_name=model_name)
    logger.info(f"[{model_name}] Training complete. Macro F1: {final_f1:.4f}. Time: {(time.time()-start)/60:.2f} mins")
    return final_f1
'''





# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description='Unified Baseline Trainer for SemEval Task13')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--model', type=str, default='B',help="Comma-separated model(s) e.g., B,G,F or 'all'")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_features', type=int, default=30000, help='Max features for TF-IDF vectorizer')
    parser.add_argument('--sample_size', type=int, default=None, help='Optional: run on a small sample of the dataset for quick iteration')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    # Parse multiple models
    models_to_run = [m.strip() for m in args.model.split(',')] if args.model != 'all' else list(MODEL_REGISTRY.keys())

    # Load datasets once
    tfidf_loaded = False
    token_loaded = False
    codebert_loaded = False


# WHENEVER YOU RUN ANY MODEL MAKE SURE THE DATA 
# IS BEING LOADAD ACC TO WHAT YOU WANT BY CHANGING 
# THE MODE IN THE loadData FUNCTION

    # 3 Available Modes:
    # tfidf-sklearn (sparse)
    # tokenization (PyTorch DataLoader (tokenization))
    # codeBert (PyTorch DataLoader(codebert specifiz tokenization))
    train_loader , val_loader, vectorizerORtokenizerORWhatever = loadData(
        "tokenization", batch_size=args.batch_size, max_features=args.max_features, sample_size=args.sample_size
    )

    #for m in models_to_run:
     #   if m in ['A','D','E','F'] and not tfidf_loaded:
      #      logger.info("Loading TF-IDF data (sparse) ...")
       #     train_loader_tfidf, val_loader_tfidf, _ = loadData(
        #        "tfidf-sklearn", batch_size=args.batch_size, max_features=args.max_features, sample_size=args.sample_size
         #   )
          #  tfidf_loaded = True

        #if m in ['B','G'] and not token_loaded:
         #   logger.info("Loading tokenized data ...")
          #  train_loader_token, val_loader_token = loadData(
           #     "tokenization", batch_size=args.batch_size, sample_size=args.sample_size
            #)
            #token_loaded = True

        #if m == 'C' and not codebert_loaded:
         #   logger.info("Loading CodeBERT data ...")
          #  train_loader_codebert, val_loader_codebert = loadData(
           #     "codeBert", batch_size=args.batch_size, sample_size=args.sample_size
            #)
            #codebert_loaded = True

    # Train each model
    for m in models_to_run:
        # determine device for this model
        if m in ['B', 'G', 'C']:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Model {m} will run on device: {device}")
        else:
            device = 'cpu'
            logger.info(f"Model {m} will run on device: cpu")

        # estimate time more accurately per model type
        est_seconds = None
        try:
            if m in ['B', 'G']:
                vocab_size = len(train_loader.dataset.tokenizer.vocab)

                # build a small factory to create the model with same args
                if m == 'B':
                    model_factory = lambda: BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=100,
                                                            hidden_dim=128, num_layers=2, num_classes=2)
                else:
                    model_factory = lambda: TextCNNClassifier(vocab_size=vocab_size, embedding_dim=100, num_classes=2)

                est_seconds = estimate_pytorch_time(model_factory, train_loader, device, epochs=args.epochs)
                logger.info(f"Estimated time for model {m}: ~{_format_seconds(est_seconds)}")

                train_model(
                    m, train_loader, val_loader, args.output_dir,
                    vocab_size=vocab_size, num_labels=2, epochs=args.epochs,
                    learning_rate=args.learning_rate
                )

            elif m == 'C':
                # CodeBERT: device-aware estimate
                model_factory = lambda: CodeBERTClassifier(num_labels=2)
                est_seconds = estimate_pytorch_time(model_factory, train_loader, device, epochs=args.epochs)
                logger.info(f"Estimated time for model {m}: ~{_format_seconds(est_seconds)}")

                train_model(
                    m, train_loader, val_loader, args.output_dir,
                    num_labels=2, epochs=args.epochs, learning_rate=args.learning_rate
                )

            else:
                # sklearn TF-IDF models: extract sparse arrays
                try:
                    (X_train_vec, y_train_vec), _ = train_loader, val_loader
                except Exception:
                    (X_train_vec, y_train_vec) = train_loader

                est_seconds = estimate_sklearn_time(MODEL_REGISTRY[m], X_train_vec, y_train_vec)
                logger.info(f"Estimated time for model {m}: ~{_format_seconds(est_seconds)} (sklearn, cpu)")

                train_model(m, train_loader, val_loader, args.output_dir)

        except Exception as e:
            logger.warning(f"Could not compute precise ETA for model {m}: {e}")
            # still run the model
            if m in ['B','G']:
                vocab_size = len(train_loader.dataset.tokenizer.vocab)
                train_model(
                    m, train_loader, val_loader, args.output_dir,
                    vocab_size=vocab_size, num_labels=2, epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
            elif m == 'C':
                train_model(
                    m, train_loader, val_loader, args.output_dir,
                    num_labels=2, epochs=args.epochs, learning_rate=args.learning_rate
                )
            else:
                train_model(m, train_loader, val_loader, args.output_dir)

if __name__ == '__main__':
    main()
#to activate the virtual environment use
#.\venv_311\Scripts\Activate.ps1