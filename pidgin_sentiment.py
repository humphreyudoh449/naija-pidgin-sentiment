"""
Nigerian Pidgin Sentiment Analysis: TF-IDF baseline vs fine-tuned transformer.

Run in Google Colab (Runtime > Change runtime type > GPU).
Install first:
    pip install -q datasets transformers evaluate scikit-learn accelerate seaborn

NOTE: Check the dataset name/config on Hugging Face before running.
Search "NaijaSenti" on huggingface.co/datasets and confirm the config
for Pidgin (commonly "pcm") and the column names (commonly "tweet", "label").
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
DATASET = "tasksource/naijasenti"   # confirm exact name on Hugging Face
LANG = "pcm"                         # Nigerian Pidgin

ds = load_dataset(DATASET, LANG)
train_df = ds["train"].to_pandas()
val_df = ds["validation"].to_pandas()
test_df = ds["test"].to_pandas()

TEXT, LABEL = "tweet", "label"       # adjust if column names differ

print(train_df.shape, val_df.shape, test_df.shape)
print(train_df[LABEL].value_counts(normalize=True))
print("Mean words per tweet:", train_df[TEXT].str.split().str.len().mean())

# ---------------------------------------------------------------
# 2. PREPROCESSING (document every choice in your write-up)
# ---------------------------------------------------------------
def clean(text):
    text = re.sub(r"http\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)
    return text.strip().lower()

for df in (train_df, val_df, test_df):
    df["clean"] = df[TEXT].apply(clean)

# ---------------------------------------------------------------
# 3. BASELINE: TF-IDF + LOGISTIC REGRESSION
# ---------------------------------------------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(train_df["clean"])   # fit on train only
X_val = tfidf.transform(val_df["clean"])
X_test = tfidf.transform(test_df["clean"])

baseline_scores = []
for seed in [1, 2, 3]:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed)
    clf.fit(X_train, train_df[LABEL])
    preds = clf.predict(X_test)
    baseline_scores.append(f1_score(test_df[LABEL], preds, average="macro"))

print(f"Baseline macro-F1: {np.mean(baseline_scores):.3f} "
      f"+/- {np.std(baseline_scores):.3f}")
print(classification_report(test_df[LABEL], preds))

cm = confusion_matrix(test_df[LABEL], preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Baseline confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ---------------------------------------------------------------
# 4. TRANSFORMER FINE-TUNING (XLM-R or AfroXLMR)
# ---------------------------------------------------------------
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

MODEL = "xlm-roberta-base"           # try "Davlan/afro-xlmr-base" as a second run
num_labels = train_df[LABEL].nunique()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch["clean"], truncation=True, max_length=128)

def to_hf(df):
    d = Dataset.from_pandas(df[["clean", LABEL]].rename(columns={LABEL: "labels"}))
    return d.map(tokenize, batched=True)

hf_train, hf_val, hf_test = to_hf(train_df), to_hf(val_df), to_hf(test_df)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

transformer_scores = []
for seed in [1, 2, 3]:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=num_labels)
    args = TrainingArguments(
        output_dir=f"out_seed{seed}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=hf_train,
                      eval_dataset=hf_val, tokenizer=tokenizer,
                      compute_metrics=compute_metrics)
    trainer.train()
    out = trainer.predict(hf_test)
    transformer_scores.append(out.metrics["test_macro_f1"])

print(f"Transformer macro-F1: {np.mean(transformer_scores):.3f} "
      f"+/- {np.std(transformer_scores):.3f}")

# ---------------------------------------------------------------
# 5. ERROR ANALYSIS: export 50 misclassified examples to label by hand
# ---------------------------------------------------------------
t_preds = np.argmax(out.predictions, axis=-1)
errors = test_df.copy()
errors["pred"] = t_preds
errors = errors[errors[LABEL] != errors["pred"]].sample(
    n=min(50, (errors[LABEL] != errors["pred"]).sum()), random_state=42)
errors["error_type"] = ""   # fill by hand: sarcasm / code-mixing / slang / negation / label noise
errors[[TEXT, LABEL, "pred", "error_type"]].to_csv("error_analysis.csv", index=False)
print("Saved error_analysis.csv: open it and categorize each error.")
