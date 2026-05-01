# Hypothesis Testing using Language Models

### Computational Psycholinguistics: Assignment 7

Assignment exploring the relationship between language model probability scores and human sentence acceptability judgements.

## Files

| File/Folder      | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `code.ipynb`     | Main notebook — training, scoring, and analysis      |
| `ratings-lm.pdf` | Assignment instructions                              |
| `Report.pdf`     | Written report with results and discussion           |
| `bnc.csv`        | English BNC dataset with human acceptability ratings |
| `browndata/`     | Brown corpus splits (train / dev / test)             |
| `ngram`          | SRILM ngram binary (scoring)                         |
| `ngram-count`    | SRILM ngram-count binary (training)                  |

## Setup

The notebook runs on Google Colab. Mount your Drive, set the `DRIVE_BASE` path in Cell 3, and run all cells in order.

**Dependencies** (auto-installed in notebook): `scipy`, `pandas`, `matplotlib`, `numpy`, `scikit-learn`, `transformers`, `torch`

## Overview

- **Q1** — Trains bigram and trigram LMs (Kneser-Ney) on Brown corpus; computes total log-prob, avg log-prob, and SLOR for each BNC sentence
- **Q2** — Spearman correlation between LM scores and human ratings
- **Q3** — Word length, frequency, and information content analysis across the combined corpus
- **Extra Credit** — Repeats Q1/Q2 using GPT-2
