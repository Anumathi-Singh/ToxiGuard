"""Reproducible training pipeline for ToxiGuard.

Run: python train_model.py
The script expects data/toxic_comments_dataset.csv.  It creates all derived
datasets, model files, charts and evaluation outputs used by the dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from src.text_utils import clean_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
RANDOM_STATE = 42
# Keeps the project fast enough for an ordinary MCA-project laptop while still
# selecting a representative, stratified subset from the source CSV.
MAX_TRAINING_ROWS = 30000


def stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_TRAINING_ROWS:
        return frame.copy()
    sampled, _ = train_test_split(
        frame, train_size=MAX_TRAINING_ROWS, stratify=frame["toxic"], random_state=RANDOM_STATE
    )
    return sampled.reset_index(drop=True)


def make_eda_plots(frame: pd.DataFrame) -> None:
    labels = [
        col
        for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        if col in frame.columns
    ]
    display_names = [x.replace("_", " ").title() for x in labels]
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    palette = ["#93C5FD", "#F97316"]

    counts = frame["toxic"].value_counts().reindex([0, 1], fill_value=0)
    axes[0, 0].bar(["Non-toxic", "Toxic"], counts.values, color=palette)
    axes[0, 0].set_title("Binary Target Distribution")
    axes[0, 0].set_ylabel("Comments")

    if labels:
        axes[0, 1].bar(display_names, frame[labels].sum().values, color="#6366F1")
        axes[0, 1].tick_params(axis="x", rotation=30)
        axes[0, 1].set_title("Available Toxicity Labels")
        axes[0, 1].set_ylabel("Positive labels")

    for value, color, label in [(0, "#60A5FA", "Non-toxic"), (1, "#FB923C", "Toxic")]:
        values = frame.loc[frame["toxic"] == value, "comment_length"]
        axes[1, 0].hist(values, bins=40, alpha=0.65, color=color, label=label)
    axes[1, 0].set_title("Comment Length Distribution")
    axes[1, 0].set_xlabel("Words")
    axes[1, 0].legend()

    average_lengths = frame.groupby("toxic")["comment_length"].mean().reindex([0, 1])
    axes[1, 1].bar(["Non-toxic", "Toxic"], average_lengths.values, color=palette)
    axes[1, 1].set_title("Average Comment Length")
    axes[1, 1].set_ylabel("Average words")
    fig.suptitle("ToxiGuard: Exploratory Data Analysis", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "eda_plots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_top_words(frame: pd.DataFrame) -> None:
    """Save an interpretable list of frequent words in toxic comments."""
    toxic_text = frame.loc[frame["toxic"] == 1, "clean_text"]
    counter = CountVectorizer(stop_words="english", max_features=5000)
    matrix = counter.fit_transform(toxic_text)
    top = (
        pd.DataFrame({"term": counter.get_feature_names_out(), "count": matrix.sum(axis=0).A1})
        .sort_values("count", ascending=False)
        .head(15)
        .sort_values("count")
    )
    top.to_csv(OUTPUTS / "top_toxic_words.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["term"], top["count"], color="#ef4444")
    ax.set_title("Most Frequent Terms in Toxic Comments")
    ax.set_xlabel("Occurrences")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "top_toxic_words.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    source = DATA / "toxic_comments_dataset.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Download the Jigsaw CSV and place it in data/ before training."
        )
    raw = pd.read_csv(source)
    required = {"comment_text", "toxic"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset must contain {sorted(required)}; missing {sorted(missing)}")

    dataset = raw.dropna(subset=["comment_text", "toxic"]).drop_duplicates(subset=["comment_text"]).copy()
    dataset["toxic"] = dataset["toxic"].astype(int)
    dataset["clean_text"] = dataset["comment_text"].map(clean_text)
    dataset = dataset[dataset["clean_text"].str.len() > 1].copy()
    dataset["comment_length"] = dataset["clean_text"].str.split().str.len()
    cleaned_path = DATA / "cleaned_toxic_comments.csv"
    dataset.to_csv(cleaned_path, index=False)
    make_eda_plots(dataset)
    make_top_words(dataset)

    sampled = stratified_sample(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        sampled["clean_text"], sampled["toxic"], test_size=0.20,
        stratify=sampled["toxic"], random_state=RANDOM_STATE,
    )
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        ),
        "SGD Classifier": SGDClassifier(loss="modified_huber", class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }
    fitted_models, rows, predictions, probabilities_by_model = {}, [], {}, {}
    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        fitted_models[name], predictions[name] = model, pred
        probabilities = model.predict_proba(X_test_tfidf)[:, 1]
        probabilities_by_model[name] = probabilities
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })

    results = pd.DataFrame(rows).sort_values(
        ["F1 Score", "Recall", "Accuracy"], ascending=False
    ).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    results.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    results.to_csv(OUTPUTS / "model_comparison_results.csv", index=False)
    best_name = results.loc[0, "Model"]
    best_model = fitted_models[best_name]
    best_pred = predictions[best_name]
    best_probabilities = probabilities_by_model[best_name]
    joblib.dump(best_model, MODELS / "toxiguard_model.pkl")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")

    result_frame = pd.DataFrame({
        "comment_text": X_test.index.map(dataset["comment_text"]),
        "clean_text": X_test.values,
        "actual_label": y_test.values,
        "predicted_label": best_pred,
        "toxicity_probability": best_probabilities,
    })
    result_frame["actual_class"] = result_frame["actual_label"].map({0: "Non-toxic", 1: "Toxic"})
    result_frame["predicted_class"] = result_frame["predicted_label"].map({0: "Non-toxic", 1: "Toxic"})
    result_frame["result"] = np.where(result_frame.actual_label == result_frame.predicted_label, "Correct", "Incorrect")
    result_frame.to_csv(OUTPUTS / "predictions_output.csv", index=False)
    report = pd.DataFrame(classification_report(y_test, best_pred, target_names=["Non-toxic", "Toxic"], output_dict=True)).T
    report.to_csv(OUTPUTS / "classification_report.csv")

    matrix = confusion_matrix(y_test, best_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Non-toxic", "Toxic"], yticklabels=["Non-toxic", "Toxic"])
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    melted = results.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                          var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=24)
    ax.set_title("Model Evaluation Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "project": "ToxiGuard – Cyberbullying and Toxic Comment Detection",
        "dataset_rows_after_cleaning": int(len(dataset)),
        "rows_used_for_training": int(len(sampled)),
        "test_rows": int(len(y_test)),
        "feature_count": int(len(vectorizer.get_feature_names_out())),
        "toxic_comments": int((dataset["toxic"] == 1).sum()),
        "non_toxic_comments": int((dataset["toxic"] == 0).sum()),
        "models_trained": int(len(candidates)),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "tfidf_settings": {
            "max_features": 15000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
        },
        "best_model": best_name,
        "confidence_method": (
            "Calibrated probability (sigmoid calibration with 3-fold cross-validation)."
            if "Calibrated" in best_name
            else "Native probability returned by the selected classifier's predict_proba method."
        ),
        "selection_rule": "Highest toxic-class F1; then recall; then accuracy.",
        "best_metrics": {key: round(float(value), 4) for key, value in results.iloc[0].to_dict().items() if key not in {"Model", "Rank"}},
    }
    (OUTPUTS / "project_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
