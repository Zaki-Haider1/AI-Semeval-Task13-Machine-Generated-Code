# train.py (Unified Baseline Trainer - Full Dataset, Timer, Global Registry)
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- MODEL REGISTRY ----------------
MODEL_REGISTRY = {
    'A': LogisticRegression(max_iter=2000, n_jobs=-1),
    'B': BiLSTMClassifier,
    'C': CodeBERTClassifier,
    'D': MultinomialNB(),
    'E': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'F': SVMClassifier,
    'G': TextCNNClassifier,
}

# ---------------- UNIFIED TRAIN FUNCTION ----------------
def train_model(model_name, train_loader, val_loader, output_dir, num_labels=None, **kwargs):
    start_time = time.time()
    logger.info(f"\nTraining model: {model_name}")

    # ---------------- PyTorch token models: BiLSTM/TextCNN ----------------
    if model_name in ['B', 'G']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        vocab_size = kwargs.get('vocab_size', None)
        if vocab_size is None:
            raise ValueError("vocab_size must be provided for token-based models")
        num_labels = kwargs.get('num_labels', 2)

        if model_name == 'B':
            model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_dim=128,
                num_layers=2,
                num_classes=num_labels
            ).to(device)
        elif model_name == 'G':
            model = TextCNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=100,
                num_classes=num_labels
            ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-3))
        best_f1 = 0.0
        num_epochs = kwargs.get('epochs', 5)

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

    # ---------------- CodeBERT (C) ----------------
    if model_name == 'C':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CodeBERTClassifier(num_labels=num_labels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-5))
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0
        num_epochs = kwargs.get('epochs', 3)

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
    if isinstance(train_loader, tuple) and len(train_loader) == 2:
        X_train_vec, y_train_vec = train_loader[0], train_loader[1]
        X_val_vec, y_val_vec = val_loader[0], val_loader[1]
    else:
        X_train_vec, y_train_vec = train_loader
        X_val_vec, y_val_vec = val_loader

    model = MODEL_REGISTRY[model_name]
    model.fit(X_train_vec, y_train_vec)
    y_pred = model.predict(X_val_vec)

    final_f1 = evaluate_and_plot(y_val_vec, y_pred, output_dir, num_labels=len(np.unique(np.concatenate((y_train_vec, y_val_vec)))), model_name=model_name)
    logger.info(f"[{model_name}] Training complete. Macro F1: {final_f1:.4f}. Time: {(time.time()-start_time)/60:.2f} mins")
    return final_f1


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description='Unified Baseline Trainer for SemEval Task13')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--model', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'all'], default='B')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    # ---------------- Load data ----------------
    train_loader_tfidf, val_loader_tfidf, _ = loadData("tfidf-pytorch", batch_size=args.batch_size, max_features=2000)
    train_loader_token, val_loader_token = loadData("tokenization", batch_size=args.batch_size)
    train_loader_codebert, val_loader_codebert = loadData("codeBert", batch_size=args.batch_size)

    models_to_run = [args.model] if args.model != 'all' else list(MODEL_REGISTRY.keys())

    for m in models_to_run:
        logger.info(f"Estimated time for model {m}: ~5 mins approx")
        if m in ['B', 'G']:
            vocab_size = len(train_loader_token.dataset.tokenizer.vocab)
            train_model(m, train_loader_token, val_loader_token, args.output_dir, vocab_size=vocab_size, num_labels=2, epochs=args.epochs, learning_rate=args.learning_rate)
        elif m == 'C':
            train_model(m, train_loader_codebert, val_loader_codebert, args.output_dir, num_labels=2, epochs=args.epochs, learning_rate=args.learning_rate)
        else:
            train_model(m, train_loader_tfidf, val_loader_tfidf, args.output_dir)


if __name__ == '__main__':
    main()
#to activate the virtual environment use
#.\venv_311\Scripts\Activate.ps1