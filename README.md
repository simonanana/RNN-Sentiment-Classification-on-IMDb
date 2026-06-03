# RNN Sentiment Classification on IMDb

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A systematic empirical study of **recurrent and non-recurrent neural architectures** for binary sentiment classification on the IMDb movie-review dataset, implemented in PyTorch.


## Project Overview

This project benchmarks six deep-learning architectures across three controlled experiments to answer the following research questions:

1. **Optimisers** — Does the choice of SGD, Adam, or Adagrad matter when the model is weak?  
2. **Epochs** — Does training a vanilla RNN longer improve performance?  
3. **Architectures** — How do FFN, CNN, LSTM, and Bi-LSTM compare on long-document sentiment?


## Repository Structure

```
rnn-sentiment-classification/
├── rnn_sentiment_classification.ipynb   # Main experiment notebook
├── experiment_results.png               # Result visualisation (auto-generated)
├── results_optimizer.csv                # Experiment 1 results
├── results_epochs.csv                   # Experiment 2 results
├── results_architectures.csv            # Experiment 3 results
└── README.md
```


## Experimental Setup

| Hyperparameter | Value |
|---|---|
| Dataset | IMDb (25k train / 25k test) |
| Train / Validation split | 70 / 30 of original training set |
| Vocabulary size | 10,000 most frequent tokens |
| Sequence length | 500 tokens (pad / truncate) |
| Embedding dimension | 128 (randomly initialised, trained jointly) |
| Batch size | 32 |
| Hidden dimension | 256 |
| Dropout | 0.3 |
| Loss | Binary Cross-Entropy (BCE) with sigmoid output |
| Gradient clipping | max_norm = 1.0 |
| Device | GPU (CUDA) when available |

Three experiment groups:

- **Warm-up** — Baseline vanilla RNN, SGD, 5 epochs  
- **Experiment 1** — SGD vs. Adam vs. Adagrad (5 epochs, Baseline RNN)  
- **Experiment 2** — 5 / 10 / 20 / 50 epochs (Adam, Baseline RNN)  
- **Experiment 3** — FFN-1/2/3, CNN, LSTM, Bi-LSTM (Adam, 50 epochs)


## Results Summary

### Experiment 1 — Optimizer Comparison (5 epochs, Baseline RNN)

| Optimizer | Test Loss | Test Accuracy |
|-----------|-----------|---------------|
| SGD       | 0.6934    | 50.36%        |
| Adam      | 0.6938    | **50.65%**    |
| Adagrad   | 0.6932    | 50.42%        |

All three optimisers cluster around ~50% (random-guess level), confirming that the vanilla RNN architecture is the bottleneck — not the optimiser.

### Experiment 2 — Epoch Comparison (Adam, Baseline RNN)

| Epochs | Test Loss | Test Accuracy |
|--------|-----------|---------------|
| 5      | 0.6955    | 50.62%        |
| 10     | 0.6962    | 50.34%        |
| 20     | 0.7148    | 50.80%        |
| 50     | 0.7607    | 50.74%        |

Accuracy remains flat while test loss increases past 20 epochs — characteristic of overfitting compounded by the vanishing gradient problem inherent in vanilla RNNs over long sequences.

### Experiment 3 — Architecture Comparison (Adam, 50 epochs)

| Model    | Parameters | Test Loss | Test Accuracy |
|----------|-----------|-----------|---------------|
| FFN-1    | 17,664,513 | 2.1329    | 81.06%        |
| FFN-2    | 17,697,281 | 3.0262    | 81.40%        |
| FFN-3    | 17,705,473 | 2.3577    | 81.75%        |
| CNN      | 1,357,401  | 2.6631    | 86.28%        |
| **LSTM** | **1,675,521** | **0.6804** | **88.17%** |
| Bi-LSTM  | 2,071,041  | 0.8533    | 87.60%        |

Key observations:
- **LSTM** achieves the highest accuracy (88.17%) with a moderate parameter budget.
- **CNN** is the most parameter-efficient model (~15.7k parameters per 1% accuracy).
- **FFNs** are the least efficient (>17M parameters, ~81% accuracy) — flattening discards sequential structure.
- **Bi-LSTM** slightly underperforms LSTM under these hyperparameters, likely due to overfitting from the larger parameter count.


## Key Insights

### 1. Architecture Is the Dominant Factor
Across all experiments, switching from a vanilla RNN to an LSTM yields a **~38 percentage-point** accuracy jump (50% → 88%), dwarfing any gain from changing the optimiser or training longer.

### 2. Vanishing Gradients Explain the RNN Ceiling
The vanilla RNN cannot propagate meaningful gradients over 500 time-steps. LSTM gating mechanisms directly address this, enabling the model to credit or blame words far from the classification decision.

### 3. Parameter Efficiency of CNNs and LSTMs
CNN and LSTM achieve strong results with compact models, while FFNs are more than an order of magnitude less parameter-efficient. This demonstrates that **inductive bias** (locality for CNN, sequential order for LSTM) matters more than raw parameter count.

### 4. Pre-trained Embeddings Would Further Improve Performance
All experiments use randomly initialised embeddings trained from scratch. Substituting pre-trained GloVe or Word2Vec embeddings would be expected to accelerate convergence and improve accuracy, especially for data-scarce settings.


## Getting Started

### Requirements

```bash
pip install torch torchvision tensorflow numpy pandas matplotlib seaborn
```

### Run

Open `rnn_sentiment_classification.ipynb` in Jupyter Lab / Google Colab and run all cells.  
A GPU is strongly recommended for Experiment 3 (50-epoch training of 6 models).


## License

This project is released under the [MIT License](LICENSE).
