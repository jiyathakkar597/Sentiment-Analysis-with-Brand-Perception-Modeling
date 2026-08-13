# Sentiment Analysis with Brand Perception Modeling

A hybrid sentiment classification system that predicts whether customer reviews are **Positive**, **Negative**, or **Neutral** — combining review text with brand perception metadata rather than relying on text alone.

## Overview

Traditional sentiment models look only at what customers wrote. This project goes a step further by fusing review text with structured brand-perception scores — **Awareness, Novelty, Personality, Dynamism, Connectivity, and Actuation** — to capture not just *what* was said, but the brand context behind it.

## How It Works

1. **Text Processing** — Reviews are cleaned and tokenized using a `DistilBERT` tokenizer.
2. **Feature Engineering** — Brand perception metadata columns are scaled using `StandardScaler`.
3. **Model** — A fine-tuned `DistilBERT`-based transformer extracts contextual embeddings from the review text, which are fused with the scaled metadata features in a hybrid architecture.
4. **Training** — The model is trained on labeled review data with sentiment classes encoded via `LabelEncoder`.
5. **Evaluation** — Performance is measured using accuracy, a full classification report, and a confusion matrix.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers (DistilBERT)
- scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn (visualizations)

## Repository Contents

| File | Description |
|---|---|
| `Project2_Model.ipynb` | Main notebook — data loading, preprocessing, model training, and evaluation |
| `Sample_Review_Data.csv` | Sample review dataset used for training/testing |
| `Advertising_Data.csv` | Supplementary advertising/brand metric data |
