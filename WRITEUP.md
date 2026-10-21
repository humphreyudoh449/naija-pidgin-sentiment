# Sentiment Analysis for Nigerian Pidgin: A Classical Baseline vs a Fine-Tuned Transformer

**Humphrey Udoh** | udohsamhumphrey@gmail.com | Website: https://humphreyudoh449.github.io | Code: https://github.com/humphreyudoh449/naija-pidgin-sentiment

## 1. Motivation
Nigerian Pidgin is widely spoken but under-resourced in NLP. Sentiment analysis for it matters for tasks such as understanding customer feedback and social media opinion. Most NLP tools are built for English, and Pidgin text is informal, code-mixed with English, and full of spelling variation. This project tests how far standard methods go on such text. It is a first technical step towards my planned doctoral research on how people without technical training can use language technology to analyse their own customer messages, and how we can verify that it works for them.

## 2. Research question
How much does a fine-tuned multilingual transformer (XLM-R) improve over a TF-IDF + logistic regression baseline on Nigerian Pidgin sentiment classification, and where do both fail?

## 3. Data
I used the Pidgin (`pcm`) subset of NaijaSenti-Twitter (Muhammad et al., 2022), with three labels: positive, neutral and negative. The original splits had 5,121 train, 1,281 validation and 4,154 test tweets.

**Data leakage.** 512 duplicate tweets were found in train, and 926 tweets appeared in both train and test. I removed duplicates within each split and dropped validation and test tweets that also appeared in train. No tweet had conflicting labels. The cleaned splits contain 4,609 train, 996 validation and 3,026 test tweets.

**Class balance (after cleaning).**

| Split | Positive | Neutral | Negative |
|---|---|---|---|
| Train | 1,540 | 66 | 3,003 |
| Test | 1,025 | 429 | 1,572 |

Neutral tweets are 1.4% of training data but 14% of the test data. This mismatch shapes the results below.

## 4. Method
**Preprocessing:** URLs and mentions were replaced with placeholders and text was lowercased. I deliberately did not normalise spelling or translate Pidgin.

**Baseline:** TF-IDF (unigrams and bigrams, minimum document frequency 2, sublinear TF) with logistic regression and balanced class weights.

**Transformer:** `xlm-roberta-base` fine-tuned for 3 epochs (learning rate 2e-5, batch size 16, maximum length 128, weight decay 0.01), without class weighting, using seeds 1, 2 and 3.

**Evaluation:** macro-F1 on the cleaned test set, chosen because the neutral class is small. XLM-R is reported as mean ± standard deviation over 3 seeds. The baseline was run once because logistic regression is close to deterministic.

## 5. Results

| Model | Test macro-F1 |
|---|---|
| TF-IDF + logistic regression | 0.433 |
| XLM-R base, 3 seeds (0.443, 0.407, 0.416) | 0.422 ± 0.015 |

Per-class F1 (baseline): positive 0.57, neutral 0.01, negative 0.72. For one XLM-R run (seed 1): positive 0.59, neutral 0.00, negative 0.75.

The transformer did not outperform the baseline. Its mean is slightly lower, and the difference is smaller than its variation across seeds. On this cleaned data, a large multilingual model gave no clear advantage over TF-IDF.

**Confusion pattern (XLM-R, one run, 1,093 errors of 3,026).**

| True label | Predicted negative | Predicted positive | Predicted neutral |
|---|---|---|---|
| Negative | 1,364 | 208 | 0 |
| Positive | 456 | 569 | 0 |
| Neutral | 292 | 137 | 0 |

The model never predicted neutral. All 429 neutral test tweets were errors, about 39% of all mistakes. It also leaned negative: 44% of positive tweets were predicted negative.

## 6. Error analysis
I read 50 randomly sampled misclassified test tweets (XLM-R, one run) and assigned each one a main error type from my own knowledge of Nigerian Pidgin. This is single-annotator and subjective, so the counts are indicative.

| Error type | Count (of 50) | Example |
|---|---|---|
| Question or plain statement, no clear emotion | 17 | "na only him waka come" |
| Slang or idiom | 12 | "the other guy na sabi man" |
| Sarcasm or joke | 8 | "na so so order e dey order" |
| Banter or excitement | 6 | "omg i swear tears just dey drop from my eyes..." |
| Dataset label looks wrong | 3 | "presido abeg na im not feeling fine" |
| Unsure | 4 | "na so e dey look like now o" |

The biggest group is text with no clear emotional signal, which fits the finding that the model never predicts neutral. Slang, idioms, sarcasm and jokes together account for 20 of the 50 errors (40%), cases where the surface words point one way and the intended meaning points another. Banter and excitement written in strong Pidgin (for example, football talk) was often read as negative.

**Label check.** For three tweets I compared the dataset label with my own reading, and these are counted in their own row above. All three were labelled positive, and I judged none of them positive: one neutral ("i dey warri oooooo who dey warri") and two negative ("presido abeg na im not feeling fine"; "where the million naira wrist watch wen dem say he buy that year"). These three were chosen because they looked doubtful, so this is evidence that some label noise exists, not an estimate of how much. In these cases the model's "negative" prediction may have been closer to my reading than the dataset label.

Neutral tweets also contained more Pidgin marker words (mean rate 0.224 vs about 0.16 for positive and negative), measured with a crude word list. This suggests the most Pidgin-heavy text was the hardest, but it is a pattern, not a demonstrated cause.

## 7. Limitations
- One dataset, tweets only; results may not transfer to other domains such as customer chats.
- The neutral class has too few training examples (66) to learn from, so conclusions about neutral are limited.
- No hyperparameter tuning or imbalance handling for the transformer.
- The baseline was run once; only XLM-R has multiple seeds.
- Some labels may be noisy, and the error categories come from one annotator (me) on 50 errors, so they are subjective and approximate.
- The Pidgin marker word list is informal.

## 8. Future work
- Address class imbalance (weighted loss, resampling, or merging classes) and re-test.
- Compare with models trained for African languages, such as AfroXLMR.
- Test on business-style messages such as customer chats, with consent, which links to my planned doctoral work on tools that non-technical SME owners can use and verify.

## References
- Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment Analysis. LREC 2022.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020.
