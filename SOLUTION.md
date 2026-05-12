# SMILES-2026 Hallucination Detection , Solution

## Reproducibility

Tested on Python 3.10 with the packages in `requirements.txt`. To run:

```bash
git clone https://github.com/carlosDev8/SMILES-2026-Hallucination-Detection---Carlos-MA.git
cd SMILES-2026-Hallucination-Detection---Carlos-MA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

This will generate both `results.json` and `predictions.csv` in the root directory. The first run downloads Qwen2.5-0.5B (~1GB). I ran this on a Colab T4 GPU, which took around 10-15 minutes total. Running on CPU is possible but noticeably slower for the hidden state extraction part.

The random seed is set to 42 everywhere (splits, PCA, network init), so results should be identical on any machine.

---

## What I did and why

I modified three files: `aggregation.py`, `probe.py`, and `splitting.py`.

### aggregation.py

The default implementation just takes the last token of the last transformer layer , a single 896-dimensional vector per sample. My first thought was that this throws away a lot of information. The model has 24 layers and all of them are doing something; only looking at the end felt like reading the last sentence of a book and calling it a summary.

I ended up with two types of features combined:

**Hidden state features.** I take the last 4 transformer layers and compute the mean over all real (non-padding) tokens for each one. Mean pooling instead of last-token is more robust , the last token is sometimes punctuation or a formatting artifact and its representation is noisy. I also keep the last-token vector of the final layer separately since in autoregressive models that position has attended to the full sequence. In total this gives 5 × 896 = 4480 values.

**Geometric features.** I also compute some scalar statistics over all 25 layers (embedding + 24 transformer layers):
- The L2 norm of the mean representation at each layer (25 values). I noticed that in some hallucinated responses the norm grows unusually in the mid-layers, as if the model is "pushing" a representation it's not confident about.
- Cosine similarity between consecutive layer representations (24 values). When the representation changes a lot from one layer to the next, it can indicate the model is recalibrating , which seems to happen more in hallucinated answers.
- Normalised sequence length (1 value). Longer responses were slightly more hallucinated on average in the training set, so I included it.

This gives 4530 features per sample, which get compressed by PCA inside the probe.

### probe.py

The main issue with the original probe was that 4530 features and ~460 training samples per fold is a recipe for overfitting. The network would memorise training examples almost perfectly while generalising poorly.

I added PCA after the StandardScaler to compress down to 256 dimensions before the network sees anything. This helped a lot , without it the training accuracy would go to ~100% while validation stayed around 65%.

For the network I went from the original one-hidden-layer MLP to:

```
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.35)
Linear(128 → 64)  → BatchNorm → ReLU → Dropout(0.20)
Linear(64 → 1)
```

The BatchNorm layers help stabilise training after PCA rotation. The dropout is fairly aggressive (especially the 0.35 in the first layer) because the dataset is small and without it the network overfits quickly.

For training I kept Adam but added weight decay (1e-4) and a cosine annealing schedule that decays the learning rate from 1e-3 down to 1e-5 over 500 epochs. The schedule helps because with a fixed learning rate the loss would oscillate towards the end instead of converging cleanly.

### splitting.py

I replaced the single train/val/test split with 5-fold stratified cross-validation. With only 689 labelled samples, a single test set of ~100 examples has too much variance , a different random seed could shift the reported accuracy by 5+ points in either direction. Five folds average over different test partitions which gives a better picture of how the model actually generalises.

From each fold's training pool I carve out 15% as validation for threshold tuning (the `fit_hyperparameters` step).

The final probe that generates `predictions.csv` is trained on all labelled data, so k-fold only affects the reported metrics in `results.json`, not the actual submission.

---

## What didn't work

**Using all 25 layers.** My first attempt concatenated the mean-pooled representation from every layer (25 × 896 = 22400 features). It didn't improve over using just the last 4 and made PCA much slower, so I dropped it. The early layers seem to carry mostly syntactic/positional information that doesn't add much for this task.

**Deeper network.** I tried adding a third hidden layer (256 → 128 → 64 → 32 → 1). It consistently made validation F1 slightly worse. With this few training examples, more parameters just means more overfitting surface.

**Logistic regression.** Out of curiosity I tried replacing the MLP entirely with sklearn's LogisticRegression after the same preprocessing. Results were comparable on validation but a bit worse on test folds. I kept the MLP mainly because it was already there and the difference wasn't large enough to justify switching.

**Attention-weighted pooling.** I wanted to weight each token's hidden state by its attention score instead of doing a flat mean. The intuition was that the tokens the model "focuses on" should carry more signal. In practice the attention patterns from the last layer were either nearly uniform (short answers) or very peaked on just 2-3 tokens (longer answers), and the features weren't more useful than the simple mean. I also would have needed to change `solution.py` to return the attention weights, which I wanted to avoid.

**PCA with 512 components.** More components didn't help and slowed things down. With ~460 training samples 256 components already captures the important variance.
