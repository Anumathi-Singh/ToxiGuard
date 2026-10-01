# ToxiGuard

## Cyberbullying and Toxic Comment Detection using Machine Learning

ToxiGuard is an MCA minor-project prototype for identifying potentially toxic
online comments. It is designed as moderation support: it flags patterns for
human review rather than making an automatic moderation decision.

The project follows one reproducible path:

CSV dataset -> text cleaning -> TF-IDF features -> six-model comparison ->
measured model selection -> saved artifacts -> Streamlit dashboard

## What the project demonstrates

- Cleaning a labelled toxic-comment CSV.
- Exploratory data analysis for an imbalanced dataset.
- TF-IDF unigram and bigram feature extraction.
- A fair comparison of six conventional machine-learning models.
- Evaluation using accuracy, toxic-class precision, recall, F1 score, ROC-AUC,
  classification report and a confusion matrix.
- Selection and saving of the best measured model and TF-IDF vectorizer.
- A professional Streamlit interface for single-comment and batch analysis.

## Models compared

| Model | Why it is included |
|---|---|
| Multinomial Naive Bayes | Fast, explainable baseline for text features. |
| Complement Naive Bayes | A useful alternative when toxic comments are the minority class. |
| Logistic Regression | Strong linear probability model for high-dimensional TF-IDF data. |
| Linear SVM (calibrated) | Margin-based linear model; calibration enables probability output. |
| SGD Classifier | Efficient incremental linear classifier for large text collections. |
| Random Forest | Non-linear tree ensemble used as a contrasting model family. |

All candidates use the same stratified 80:20 train/test split and the same
TF-IDF feature matrix. The selection rule is:

1. Highest toxic-class F1 score
2. Then highest toxic recall
3. Then highest accuracy

This avoids choosing a model only because it performs well on the larger
non-toxic class.

## Current measured result

The included run uses a 30,000-row stratified experiment subset from the
cleaned Jigsaw CSV and evaluates 6,000 held-out comments.

| Rank | Model | Accuracy | Precision | Recall | Toxic F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.932 | 0.610 | 0.814 | 0.697 | 0.958 |
| 2 | SGD Classifier | 0.932 | 0.613 | 0.795 | 0.692 | 0.953 |
| 3 | Linear SVM (Calibrated) | 0.950 | 0.855 | 0.574 | 0.687 | 0.956 |
| 4 | Multinomial Naive Bayes | 0.945 | 0.926 | 0.459 | 0.614 | 0.945 |
| 5 | Complement Naive Bayes | 0.892 | 0.467 | 0.864 | 0.606 | 0.945 |
| 6 | Random Forest | 0.917 | 0.558 | 0.628 | 0.591 | 0.914 |

Logistic Regression is currently deployed because it produced the highest
toxic-class F1 score. It was selected from these results, not assumed to be
the winner beforehand.

## Project layout

    app.py                  Streamlit dashboard
    train_model.py          Canonical training, evaluation and export pipeline
    train.py                Backward-compatible wrapper for train_model.py
    src/text_utils.py       Shared cleaning and binary-label helpers
    data/                   Source CSV and cleaned CSV
    models/                 Saved selected model and TF-IDF vectorizer
    outputs/                Metrics, predictions, figures and metadata
    notebooks/              Guided training notebook
    report/                 Report-ready project documentation

The dashboard and training script both import the same clean_text function
from src/text_utils.py. This prevents training/inference preprocessing drift.

## Run locally

The raw and cleaned Jigsaw CSV files are not stored in this repository. Obtain the source data from the [official Jigsaw competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) under its usage terms, or use `python download_dataset.py` to fetch the public training CSV. The saved model, vectorizer, compact evaluation results, and latest internship report are included, so the dashboard can run after installing dependencies. Running `python train_model.py` with the source CSV regenerates the cleaned dataset and all model and evaluation outputs.

1. Create and activate a virtual environment if desired.
2. Install dependencies.

       python -m pip install -r requirements.txt

3. Put the Jigsaw train.csv file at:

       data/toxic_comments_dataset.csv

   Or download and decompress the public CSV directly to that location:

       python download_dataset.py

4. To reproduce the experiment, train the models and regenerate all artifacts.

       python train_model.py

5. Start the dashboard. The included saved artifacts also allow this step directly after installing dependencies.

       python -m streamlit run app.py

If PowerShell says that streamlit is not recognised, use the final command
above. Running Streamlit through Python uses the installed package from the
active environment.

## Training pipeline in simple terms

1. Read comment_text and toxic from the CSV.
2. Remove missing, duplicate and empty comments.
3. Lowercase text, remove URLs and Wikipedia-style markup, and normalise
   spacing. Useful words and negation are retained.
4. Fit TF-IDF only on training text, using up to 15,000 unigram/bigram
   features.
5. Fit all six models with the same input features.
6. Calculate test metrics and rank results by the documented selection rule.
7. Save the selected model, vectorizer, comparison table, classification
   report, confusion matrix, prediction export and project metadata.

## Dashboard pages

- Overview: project purpose, real dataset totals, selection rule and workflow.
- Analyze comment: one comment, predicted class, toxic probability, review
  priority, preprocessing and available linear-model feature evidence.
- Batch analysis: CSV upload or pasted comments, results table and CSV export.
- Model performance: comparison table, metric chart, confusion matrix,
  classification report, algorithm guide and reproducibility settings.
- Dataset insights: class balance, label totals, text-length patterns and
  frequent toxic-label terms.
- About the project: methodology, TF-IDF explanation, all six algorithms,
  measured conclusion and limitations.

## Responsible use

Toxicity depends on context, sarcasm, quotations, conversation history and
local language. ToxiGuard is an English-language binary classification
prototype and must not be the sole basis for punitive or safety-critical
action.
