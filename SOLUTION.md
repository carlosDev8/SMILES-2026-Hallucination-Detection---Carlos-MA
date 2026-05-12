# SMILES-2026 Hallucination Detection

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

This will generate both `results.json` and `predictions.csv` in the root directory. The first run downloads Qwen2.5-0.5B (~1GB). I ran this on a Colab T4 GPU, which took around 15 minutes total. Running on CPU works but the hidden state extraction is much slower.

The random seed is fixed to 42 everywhere (splits, PCA, network init), so results should be identical on any machine.

---

## What I did and why

I modified three files: `aggregation.py`, `probe.py`, and `splitting.py`.

### aggregation.py

The default implementation just takes the last token of the last transformer layer, a single 896-dimensional vector per sample. My first thought was that this throws away a lot of information. The model has 24 layers and the representation evolves across all of them, so only looking at the very end felt limiting.

I ended up with two types of features combined:

**Hidden state features.** I take the last 4 transformer layers and compute the mean over all real (non-padding) tokens for each one. Mean pooling instead of last-token is more robust, since the last token is sometimes punctuation or a formatting artifact. I also keep the last-token vector of the final layer separately, since in autoregressive models that position has attended to the full context. In total this gives 5 x 896 = 4480 values.

**Geometric features.** I also compute scalar statistics over all 25 layers (embedding + 24 transformer layers):
- The L2 norm of the mean representation at each layer (25 values). In some hallucinated responses the norm grows unusually in the mid-layers, as if the model is forcing a representation it has no factual grounding for.
- Cosine similarity between consecutive layer representations (24 values). A drop in similarity between two consecutive layers means the representation changed a lot in one step, which seems to happen more in hallucinated answers.
- Normalised sequence length (1 value). Longer responses were slightly more hallucinated on average in the training set.

This gives 4530 features per sample, which get compressed by PCA inside the probe.

### probe.py

With only ~460 training samples per fold and 4530 raw features, overfitting is the main concern. I first tried a deeper MLP with BatchNorm and Dropout, but the training AUROC went to 100% while test AUROC stayed around 61%, which is basically the model memorising the training set. More details in the failed attempts section below.

The final approach is much simpler: StandardScaler, then PCA down to 64 components, then a single linear layer. This is essentially regularised logistic regression. With this few samples, a linear decision boundary turns out to generalise better than any nonlinear model I tried. The weight decay is set to 0.1, which is fairly strong, to prevent even the linear model from overfitting.

The `fit_hyperparameters` step tunes the decision threshold on the validation set to maximise F1, which matters here because the classes are imbalanced (roughly 70% hallucinated).

### splitting.py

I replaced the single train/val/test split with 5-fold stratified cross-validation. With only 689 labelled samples, a single test set of around 100 examples has a lot of variance; a different random seed can shift the reported accuracy by 5 points or more. Five folds give a more honest picture of generalisation.

From each fold's training pool I carve out 15% as validation for threshold tuning.

The final probe that generates `predictions.csv` is trained on all labelled data, so k-fold only affects the reported metrics in `results.json`, not the actual submission.

---

## Results

The final results averaged over 5 folds:

- Majority-class baseline accuracy: 70.10%
- Probe test accuracy: 69.81%
- Probe test AUROC: 62.86%

The accuracy is essentially at baseline level, which means the threshold-tuned predictions are not much better than always predicting hallucination. The AUROC of ~63% tells a slightly different story: the model does rank hallucinated responses above truthful ones with some reliability (50% would be random), but not enough to translate into clean accuracy improvements.

This is partly a dataset size problem. With 468 training samples per fold and a 70/30 class split, there is not much room to learn a robust decision boundary from 4530-dimensional features. The signal is there (train AUROC ~79%) but it does not generalise cleanly.

---

## What didn't work

**MLP with BatchNorm and Dropout.** My first probe was a two-hidden-layer network (256 -> 128 -> 64 -> 1) with BatchNorm and Dropout, PCA to 256 components, and cosine LR annealing. Train AUROC hit 100% immediately while val and test stayed around 61%. The network was memorising the training set despite the regularisation. The fundamental issue is that 468 samples is not enough for a nonlinear model with this feature dimensionality.

**Using all 25 layers.** Concatenating the mean-pooled representation from every layer (25 x 896 = 22400 features) did not improve over using just the last 4. The early layers seem to carry mostly syntactic and positional information that does not add much signal for this task, and the extra dimensions just increased PCA fitting time.

**Deeper MLP (3 hidden layers).** Adding a third hidden layer made things worse. More parameters with this few training examples just adds more overfitting surface.

**Attention-weighted pooling.** I wanted to weight each token's hidden state by its attention score instead of doing a flat mean. The attention patterns from the last layer were either nearly uniform for short answers or very peaked on just a few tokens for longer ones, and the resulting features were not more useful than the simple mean. It would also have required modifying `solution.py` to return attention weights, which I wanted to avoid.

**PCA with more components (128, 256, 512).** More components consistently led to more overfitting without improving test AUROC. 64 components was the sweet spot for this dataset size.
