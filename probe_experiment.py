"""
Frozen-embedding linear probe: BERT-Small vs ELECTRA-Small (discriminator)

What this tests
----------------
ELECTRA's central claim is that replaced-token-detection pretraining produces
BETTER representations than masked-language-modeling at equal model size and
compute. This script runs a small, independent test of that specific claim:

  1. Take two architecturally IDENTICAL encoders (12 layers, 256 hidden,
     4 attention heads) -- one pretrained with BERT's MLM objective, one
     with ELECTRA's RTD objective.
  2. Freeze both -- no fine-tuning, no gradient updates to either model.
  3. Extract mean-pooled sentence embeddings from each on SST-2 (binary
     sentiment).
  4. Train a simple logistic regression probe on top of each model's frozen
     embeddings, and compare probe accuracy on held-out validation data.

If ELECTRA's representations are genuinely more linearly separable --
i.e. the pretraining objective itself produces better features, not just
"a bigger model" -- the ELECTRA probe should outperform the BERT probe
even though NEITHER model is fine-tuned and the probe is trivial.

Model choice -- why this pairing specifically
----------------------------------------------
google/electra-small-discriminator : 12 layers, hidden=256, heads=4 (~14M params)
google/bert_uncased_L-12_H-256_A-4 : 12 layers, hidden=256, heads=4 (~14M params)

These are size-matched on purpose. Comparing ELECTRA-Small against
bert-base-uncased (110M params) would confound "better objective" with
"bigger model" -- this pairing isolates the pretraining objective as the
only real variable, matching the spirit of the ELECTRA paper's own
BERT-Small vs ELECTRA-Small comparison (Table 1).

Expected runtime: a few minutes on CPU (forward passes only, no training
of the transformers themselves -- only a tiny logistic regression on top).

Usage:
    pip install -r requirements_probe.txt
    python probe_experiment.py
"""

import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# SST-2 loaded directly as parquet from the Hugging Face Hub's auto-generated
# dataset mirror -- NOT via the `datasets` library. As of this writing, the
# `datasets` library depends on `dill` for its caching fingerprint, and `dill`
# is not yet compatible with Python 3.14's changed pickle.Pickler internals
# (a new required argument was added to _batch_setitems in 3.14). Loading
# parquet directly with pandas avoids that dependency chain entirely.
SST2_TRAIN_URL = (
    "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/sst2/train-00000-of-00001.parquet"
)
SST2_VAL_URL = (
    "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/sst2/validation-00000-of-00001.parquet"
)

SEED = 42
TRAIN_SAMPLES = 2000     # subset of SST-2 train split used to fit the probe
BATCH_SIZE = 32
MAX_LENGTH = 64

MODELS = {
    "BERT-Small": "google/bert_uncased_L-12_H-256_A-4",
    "ELECTRA-Small": "google/electra-small-discriminator",
}


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings into one sentence embedding, respecting padding.

    Used for BOTH models identically. BERT's own pooler_output is skipped on
    purpose: it includes an extra NSP-trained dense+tanh layer that ELECTRA
    has no equivalent of, which would make the comparison unfair. Mean-pooling
    the raw last_hidden_state keeps the comparison apples-to-apples.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Run frozen forward passes and return mean-pooled embeddings as a numpy array."""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embeddings.append(pooled.numpy())
    return np.concatenate(all_embeddings, axis=0)


def run_probe(model_name: str, hf_id: str, train_texts, train_labels, val_texts, val_labels):
    print(f"\n--- {model_name} ({hf_id}) ---")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)

    print("Extracting frozen embeddings (train)...")
    X_train = embed_texts(train_texts, tokenizer, model)
    print("Extracting frozen embeddings (validation)...")
    X_val = embed_texts(val_texts, tokenizer, model)

    print("Fitting logistic regression probe...")
    probe = LogisticRegression(max_iter=2000, random_state=SEED)
    probe.fit(X_train, train_labels)

    val_preds = probe.predict(X_val)
    acc = accuracy_score(val_labels, val_preds)

    elapsed = time.time() - t0
    print(f"{model_name} probe accuracy: {acc:.4f}  ({elapsed:.1f}s)")
    return acc


def main():
    print("Loading SST-2 (GLUE) directly as parquet...")
    train_df = pd.read_parquet(SST2_TRAIN_URL)
    val_df = pd.read_parquet(SST2_VAL_URL)  # full validation set, already small (~872 examples)

    train_df = train_df.sample(n=TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)

    train_texts = train_df["sentence"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["sentence"].tolist()
    val_labels = val_df["label"].tolist()

    print(f"Train subset: {len(train_texts)} examples | Validation: {len(val_texts)} examples")

    results = {}
    for model_name, hf_id in MODELS.items():
        results[model_name] = run_probe(
            model_name, hf_id, train_texts, train_labels, val_texts, val_labels
        )

    print("\n=== Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    diff = results["ELECTRA-Small"] - results["BERT-Small"]
    print(f"\nELECTRA-Small vs BERT-Small (same architecture, frozen): {diff:+.4f}")

    # --- bar chart for the portfolio ---
    names = list(results.keys())
    accs = [results[n] for n in names]
    colors = ["#6b7280", "#22c55e"]  # grey for BERT, green for ELECTRA -- matches demo's color language

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel("SST-2 validation accuracy (frozen linear probe)")
    ax.set_title("Frozen-Embedding Linear Probe\nBERT-Small vs ELECTRA-Small (same architecture)")
    ax.set_ylim(0.5, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("probe_comparison.png", dpi=150)
    print("\nSaved chart to probe_comparison.png")


if __name__ == "__main__":
    main()