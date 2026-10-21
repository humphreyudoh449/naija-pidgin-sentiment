# Sentiment Analysis for Nigerian Pidgin: TF-IDF Baseline vs XLM-R

A small, reproducible study of tweet sentiment classification in Nigerian Pidgin (NaijaSenti, `pcm`). It compares a classical baseline with a fine-tuned multilingual transformer, checks for data leakage, and analyses where both models fail.

**Author:** Humphrey Udoh | udohsamhumphrey@gmail.com | https://humphreyudoh449.github.io

## Research question
How much does a fine-tuned multilingual transformer improve over a TF-IDF + logistic regression baseline on low-resource Nigerian Pidgin sentiment, and where do both fail?

## Key findings
- **The transformer did not clearly beat the baseline.** Test macro-F1 was 0.433 for TF-IDF + logistic regression and 0.422 ± 0.015 for XLM-R base (3 seeds). The gap is within seed-to-seed variation.
- **Neither model predicts the neutral class.** Only 66 of 4,609 training tweets are neutral (1.4%), yet 429 of 3,026 test tweets (14%) are. Neutral F1 is about 0 for both models.
- **Both models lean negative.** In one XLM-R run, 456 of 1,025 positive test tweets (44%) were predicted negative.
- **The data had leakage.** 926 tweets appeared in both train and test. Removing duplicates and overlaps cut the test set from 4,154 to 3,026 tweets.
- **Neutral tweets contained more Pidgin marker words** (mean rate 0.224 vs about 0.16 for positive and negative). This uses a crude word list and is a pattern, not a proven cause.

## Results (test set, after cleaning)

| Model | Macro-F1 | Positive F1 | Neutral F1 | Negative F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.433 | 0.57 | 0.01 | 0.72 |
| XLM-R base (mean of 3 seeds) | 0.422 ± 0.015 | n/a | n/a | n/a |
| XLM-R base (single run, seed 1) | 0.446 | 0.59 | 0.00 | 0.75 |

Per-seed XLM-R macro-F1: 0.443, 0.407, 0.416.

## Data
- **Source:** [NaijaSenti-Twitter](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter), Pidgin config `pcm` (Muhammad et al., 2022). Licence: CC BY-NC-SA 4.0.
- **Labels:** 0 = positive, 1 = neutral, 2 = negative.

| Split | Original | After cleaning |
|---|---|---|
| Train | 5,121 | 4,609 |
| Validation | 1,281 | 996 |
| Test | 4,154 | 3,026 |

**Cleaning steps:** URLs replaced with `<url>`, mentions with `<user>`, text lowercased, exact duplicates removed within each split, and validation/test tweets that also appear in train removed. No tweet had conflicting labels.

## Methods
- **Baseline:** TF-IDF (unigrams and bigrams, `min_df=2`, sublinear TF) + logistic regression with balanced class weights.
- **Transformer:** `xlm-roberta-base`, 3 epochs, learning rate 2e-5, batch size 16, max length 128, weight decay 0.01, no class weighting. Seeds 1, 2 and 3.
- **Metric:** macro-F1, so the small neutral class counts as much as the large ones.
- **Hardware:** Google Colab, T4 GPU. Each XLM-R run takes about 1.5 minutes.

## Error analysis
50 random misclassified test tweets (XLM-R, one run) were reviewed by hand by the author, a Nigerian Pidgin speaker. Main error types (single annotator, indicative only):

| Error type | Count (of 50) |
|---|---|
| Question or plain statement, no clear emotion | 17 |
| Slang or idiom | 12 |
| Sarcasm or joke | 8 |
| Banter or excitement | 6 |
| Dataset label looks wrong | 3 |
| Unsure | 4 |

Three tweets labelled positive in the dataset were judged not positive on manual review (one neutral, two negative), which suggests some label noise. These three were picked because they looked doubtful, so this does not estimate the noise rate. See `WRITEUP.md` for details.

## Limitations
- Tweets only, one dataset, one language variety.
- The baseline was run once (it is close to deterministic); only XLM-R has multiple seeds.
- The neutral class is too small in training to learn from, which limits any conclusion about neutral.
- The Pidgin marker word list is informal and may include words that are also standard English.
- Labels may contain noise; a few examples looked doubtful on manual review.
- No hyperparameter tuning or class-imbalance handling for the transformer.

## How to run
1. Open `Sentiment_analysis_naija_pidgin.ipynb` in Google Colab and set the runtime to **T4 GPU**.
2. Install the dependencies. This dataset uses a loading script, so pin an older `datasets` version:
   ```
   pip install -q "datasets<4" transformers evaluate scikit-learn accelerate seaborn
   ```
3. Restart the session, then run the cells in order. If the script loader fails, the notebook falls back to the Hub's parquet copy (`revision="refs/convert/parquet"`).

## Repo contents
- `Sentiment_analysis_naija_pidgin.ipynb`: full notebook
- `requirements.txt`: dependencies
- `WRITEUP.md`: two-page write-up
- `error_analysis.csv`: the 50 reviewed errors, with error types

## Future work
- Handle class imbalance (class-weighted loss, resampling, or merging classes).
- Try AfroXLMR or other African-language models.
- Test on messages closer to business use, such as customer chats, with consent.

## Citation
Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis. LREC 2022.
