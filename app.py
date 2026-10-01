"""ToxiGuard Streamlit dashboard.

This file intentionally keeps dashboard logic close to the screen it supports.
The training process lives in train_model.py; this file only reads its saved
artifacts and presents the measured results.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.text_utils import clean_text


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
NAVIGATION = (
    "Overview",
    "Analyze comment",
    "Batch analysis",
    "Model performance",
    "Dataset insights",
    "About the project",
)

MODEL_GUIDE = {
    "Multinomial Naive Bayes": {
        "family": "Probabilistic baseline",
        "how": "Estimates how often each TF-IDF term appears in toxic and non-toxic comments, then applies Bayes' rule.",
        "trade_off": "Fast and easy to explain, but it assumes features are independent and can miss interactions between phrases.",
    },
    "Complement Naive Bayes": {
        "family": "Imbalance-aware probabilistic model",
        "how": "Uses evidence from the opposite class when estimating term weights, which often helps when toxic comments are less common.",
        "trade_off": "It recovered more toxic comments in this run, but its lower precision produced more false alerts.",
    },
    "Logistic Regression": {
        "family": "Linear discriminative classifier",
        "how": "Learns a weight for every TF-IDF word or phrase and converts the weighted sum into a class probability with a sigmoid function.",
        "trade_off": "It gave the best toxic-class F1 score in the measured experiment and its native probability is suitable for the dashboard.",
    },
    "Linear SVM (Calibrated)": {
        "family": "Margin-based linear classifier",
        "how": "Finds a separating boundary with the largest possible margin between classes. Sigmoid calibration converts its score into a probability.",
        "trade_off": "It achieved the highest accuracy here, but its toxic recall was lower, so it missed more toxic comments than the selected model.",
    },
    "SGD Classifier": {
        "family": "Incremental linear classifier",
        "how": "Optimises a linear decision function one training example at a time using stochastic gradient descent.",
        "trade_off": "It was close to the selected model while being efficient for larger text collections.",
    },
    "Random Forest": {
        "family": "Tree ensemble",
        "how": "Combines predictions from many decision trees trained on different random views of the TF-IDF features.",
        "trade_off": "It adds a non-linear comparison point, but sparse high-dimensional text is usually a stronger fit for linear models.",
    },
}


def inject_css() -> None:
    """Apply a restrained visual system without changing Streamlit behaviour."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #10233f;
            --muted: #5d6f86;
            --canvas: #f5f7fb;
            --line: #dce5ef;
            --blue: #1f5ea8;
            --teal: #167d79;
            --danger: #b42318;
            --warning: #a15c00;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
        .block-container { max-width: 1420px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1f3b 0%, #102d50 55%, #0b1f3b 100%);
            border-right: 1px solid #21456e;
        }
        [data-testid="stSidebar"] * { color: #edf5ff; }
        [data-testid="stSidebar"] .stRadio > label { display: none; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: .24rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: transparent; border: 1px solid transparent; border-radius: 8px;
            padding: .38rem .55rem; margin: 0; min-height: 0;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.14);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
        [data-testid="stSidebar"] [role="radiogroup"] label p { font-size: .94rem; font-weight: 560; }
        .brand-lockup { display:flex; align-items:center; gap:.7rem; margin: .4rem 0 .25rem; }
        .brand-mark {
            align-items:center; background:#41b4a7; border-radius:9px; color:#08213c !important;
            display:flex; font-size:.78rem; font-weight:800; height:30px; justify-content:center; width:30px;
        }
        .brand-name { color:#fff; font-size:1.2rem; font-weight:760; letter-spacing:-.025em; }
        .side-caption { color:#aec4dd !important; font-size:.82rem; line-height:1.45; margin-bottom:1.2rem; }
        .side-model {
            background: rgba(65,180,167,.15); border:1px solid rgba(115,225,213,.26);
            border-radius:10px; color:#eafffb !important; font-size:.9rem; font-weight:650; padding:.7rem .75rem;
        }
        .hero {
            background: linear-gradient(114deg, #10335e 0%, #164e83 63%, #137a78 160%);
            border-radius:18px; box-shadow: 0 12px 30px rgba(17,52,92,.12);
            color:#fff; margin-bottom:1.7rem; overflow:hidden; padding:1.9rem 2.1rem; position:relative;
        }
        .hero:after {
            border:1px solid rgba(255,255,255,.15); border-radius:50%; content:"";
            height:190px; position:absolute; right:-40px; top:-95px; width:190px;
        }
        .eyebrow { color:#bde9e1; font-size:.78rem; font-weight:750; letter-spacing:.1em; margin-bottom:.38rem; text-transform:uppercase; }
        .hero h1 { color:#fff; font-size:2.15rem; letter-spacing:-.045em; line-height:1.08; margin:0; }
        .hero p { color:#dcecff; font-size:1rem; margin:.65rem 0 0; max-width:780px; }
        .page-kicker { color:var(--blue); font-size:.78rem; font-weight:780; letter-spacing:.09em; text-transform:uppercase; }
        h2 { color:var(--ink); letter-spacing:-.025em; margin-top:.15rem; }
        h3 { color:var(--ink); letter-spacing:-.015em; }
        .lead { color:var(--muted); font-size:1rem; line-height:1.6; margin:0 0 1.25rem; max-width:920px; }
        .stat-card, .info-card, .result-card {
            background:#fff; border:1px solid var(--line); border-radius:13px; box-sizing:border-box;
        }
        .stat-card { min-height:125px; padding:1.05rem 1.1rem; }
        .stat-label { color:var(--muted); font-size:.75rem; font-weight:760; letter-spacing:.075em; text-transform:uppercase; }
        .stat-value { color:var(--ink); font-size:1.72rem; font-weight:780; letter-spacing:-.04em; line-height:1.25; margin:.35rem 0; }
        .stat-note { color:var(--muted); font-size:.86rem; line-height:1.35; }
        .info-card { min-height:142px; padding:1.1rem 1.15rem; }
        .info-card h4 { color:var(--ink); font-size:1rem; margin:0 0 .45rem; }
        .info-card p { color:var(--muted); font-size:.9rem; line-height:1.5; margin:0; }
        .step-number {
            color:var(--teal); font-size:.76rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
        }
        .result-card { border-left:5px solid var(--accent); margin:1rem 0 1.1rem; padding:1.2rem 1.3rem; }
        .result-card h3 { margin:0 0 .25rem; }
        .result-card p { color:var(--muted); line-height:1.5; margin:0; }
        .note {
            background:#edf5fc; border-left:4px solid #4a86c5; border-radius:7px;
            color:#26435f; line-height:1.5; margin:.75rem 0 1.1rem; padding:.75rem .9rem;
        }
        .warning-note {
            background:#fff8e7; border-left:4px solid #d9911a; border-radius:7px;
            color:#62430b; line-height:1.5; margin:1rem 0; padding:.75rem .9rem;
        }
        .algorithm-card {
            background:#fff; border:1px solid var(--line); border-radius:12px; margin-bottom:.75rem; padding:1rem 1.1rem;
        }
        .algorithm-card h4 { color:var(--ink); margin:0 0 .2rem; }
        .algorithm-card .family { color:var(--teal); font-size:.78rem; font-weight:750; text-transform:uppercase; }
        .algorithm-card p { color:var(--muted); line-height:1.48; margin:.46rem 0 0; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
        .stButton > button, .stDownloadButton > button {
            background:#1f5ea8; border:1px solid #1f5ea8; border-radius:8px; color:#fff;
            font-weight:680; min-height:42px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { background:#174b87; border-color:#174b87; color:#fff; }
        .stTextArea textarea, .stTextInput input {
            background:#fff !important; border:1px solid #b9c9db !important; border-radius:8px !important; color:#10233f !important;
        }
        [data-testid="stMetric"] {
            background:#fff; border:1px solid var(--line); border-radius:11px; padding:.8rem .9rem;
        }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--ink); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_project_assets() -> tuple[object, object, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only artifacts produced by the canonical training script."""
    required = {
        "model": MODELS / "toxiguard_model.pkl",
        "vectorizer": MODELS / "tfidf_vectorizer.pkl",
        "metadata": OUTPUTS / "project_metadata.json",
        "comparison": OUTPUTS / "model_comparison_results.csv",
        "report": OUTPUTS / "classification_report.csv",
        "top_words": OUTPUTS / "top_toxic_words.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    model = joblib.load(required["model"])
    vectorizer = joblib.load(required["vectorizer"])
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(required["comparison"]).sort_values("Rank").reset_index(drop=True)
    report = pd.read_csv(required["report"], index_col=0)
    top_words = pd.read_csv(required["top_words"])
    return model, vectorizer, metadata, comparison, report, top_words


def positive_probability(model: object, features: object) -> float:
    """Return the probability of label 1 without assuming class ordering."""
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return float(probabilities[positive_index])


def assess_comment(comment: str, model: object, vectorizer: object) -> dict[str, object]:
    """Clean, vectorise and classify one non-empty comment."""
    cleaned = clean_text(comment)
    if not cleaned:
        raise ValueError("Enter a comment that contains words or numbers.")

    probability = positive_probability(model, vectorizer.transform([cleaned]))
    if probability >= 0.75:
        risk, accent = "High review priority", "#b42318"
        message = "The model found a strong toxic-language pattern. A moderator should inspect the surrounding context."
    elif probability >= 0.45:
        risk, accent = "Review recommended", "#a15c00"
        message = "The model found mixed signals. Read the comment in context before making a moderation decision."
    else:
        risk, accent = "Lower review priority", "#167d79"
        message = "No strong toxic-language pattern was found. Context can still change the appropriate decision."

    return {
        "comment": comment,
        "cleaned_text": cleaned,
        "prediction": "Toxic" if probability >= 0.50 else "Non-toxic",
        "toxicity_probability": probability,
        "confidence": max(probability, 1 - probability),
        "review_priority": risk,
        "message": message,
        "accent": accent,
    }


def feature_evidence(model: object, vectorizer: object, cleaned_text: str) -> list[str]:
    """Return present, positively weighted features when a linear model exposes them."""
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return []

    row = vectorizer.transform([cleaned_text]).tocoo()
    terms = vectorizer.get_feature_names_out()
    contributions = [
        (terms[column], float(value * coefficients[0][column]))
        for column, value in zip(row.col, row.data)
        if value * coefficients[0][column] > 0
    ]
    contributions.sort(key=lambda item: item[1], reverse=True)
    return [term for term, _ in contributions[:5]]


def add_to_history(records: list[dict[str, object]]) -> None:
    """Keep the current browser session's latest predictions only."""
    st.session_state.history = (records + st.session_state.get("history", []))[:20]


def stat_card(label: str, value: str, note: str) -> str:
    return (
        "<div class='stat-card'>"
        f"<div class='stat-label'>{label}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-note'>{note}</div></div>"
    )


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f"<div class='page-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f"<p class='lead'>{description}</p>", unsafe_allow_html=True)


def percentage(value: float) -> str:
    return f"{value:.1%}"


def show_overview(metadata: dict, comparison: pd.DataFrame) -> None:
    page_header(
        "Project dashboard",
        "Moderation support, grounded in measured results",
        "ToxiGuard is a compact machine-learning project for flagging potentially toxic online comments. "
        "It supports review; it does not make final moderation decisions.",
    )
    winner = comparison.iloc[0]
    cards = (
        ("Cleaned comments", f"{metadata['dataset_rows_after_cleaning']:,}", "Jigsaw toxic-comment CSV"),
        ("Toxic comments", f"{metadata['toxic_comments']:,}", f"{metadata['non_toxic_comments']:,} non-toxic comments"),
        ("Models compared", str(metadata["models_trained"]), "One shared TF-IDF test setup"),
        ("Selected model", str(metadata["best_model"]).replace(" (Calibrated)", ""), "Chosen from held-out metrics"),
    )
    for column, card in zip(st.columns(4), cards):
        column.markdown(stat_card(*card), unsafe_allow_html=True)

    st.markdown("### Why this model was selected")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            "<div class='note'><strong>Selection rule:</strong> rank models by toxic-class F1 score; "
            "use recall, then accuracy, only as tie-breakers. This prevents a high non-toxic majority "
            "from making accuracy look better than the toxic-detection result.</div>",
            unsafe_allow_html=True,
        )
        st.write(
            f"**{winner['Model']}** had the best balance of toxic precision and recall in the real test run: "
            f"F1 = **{percentage(winner['F1 Score'])}**, recall = **{percentage(winner['Recall'])}**, "
            f"precision = **{percentage(winner['Precision'])}**."
        )
    with right:
        st.metric("Toxic-class F1", percentage(winner["F1 Score"]))
        st.caption("The primary selection metric for this imbalanced binary task.")

    st.markdown("### End-to-end workflow")
    steps = (
        ("01", "Prepare the CSV", "Remove empty and duplicate comments; retain the original labels for traceability."),
        ("02", "Clean text", "Lowercase, remove URLs and markup, then normalise spacing without deleting negation."),
        ("03", "Create features", "Use TF-IDF unigrams and bigrams to convert text into numeric feature weights."),
        ("04", "Compare six models", "Train every candidate on the same stratified train/test split."),
        ("05", "Select and save", "Save the measured winner, vectorizer, metrics, predictions and figures."),
    )
    for column, step in zip(st.columns(5), steps):
        number, title, text = step
        column.markdown(
            f"<div class='info-card'><div class='step-number'>{number}</div><h4>{title}</h4><p>{text}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### What you can demonstrate")
    first, second, third = st.columns(3)
    first.markdown(
        "<div class='info-card'><h4>Single-comment analysis</h4><p>Enter a comment and see the predicted class, "
        "toxic probability, review priority and the exact cleaning applied.</p></div>",
        unsafe_allow_html=True,
    )
    second.markdown(
        "<div class='info-card'><h4>Batch analysis</h4><p>Upload a CSV or paste multiple comments, then export "
        "the resulting predictions for review.</p></div>",
        unsafe_allow_html=True,
    )
    third.markdown(
        "<div class='info-card'><h4>Evidence for the viva</h4><p>Open model performance to explain the metrics, "
        "confusion matrix, algorithm choices and selection rule.</p></div>",
        unsafe_allow_html=True,
    )


def show_analyze_comment(model: object, vectorizer: object, metadata: dict) -> None:
    page_header(
        "Single-comment check",
        "Analyze a comment",
        "Use this page to inspect one comment. The displayed score is the model's toxic-class probability, "
        "not a replacement for human judgement.",
    )
    examples = {
        "Write my own comment": "",
        "Constructive comment": "Thank you for taking the time to explain this clearly.",
        "Potentially toxic comment": "You are useless and nobody wants to hear you.",
        "Context-sensitive comment": "That comment was a bad idea and needs to be corrected.",
    }
    left, right = st.columns([1.55, 0.85])
    with left:
        choice = st.selectbox("Start with an example", list(examples))
        comment = st.text_area(
            "Comment to analyze",
            value=examples[choice],
            height=190,
            placeholder="Type or paste an online comment here...",
        )
        run_analysis = st.button("Analyze comment", type="primary", width="stretch")
    with right:
        st.markdown("#### How to read the result")
        st.markdown(
            "<div class='info-card'><h4>Prediction threshold</h4><p>At 50%, the model labels the comment "
            "toxic; below 50%, non-toxic. The review priority uses separate bands to make follow-up clearer.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='info-card'><h4>Deployed model</h4><p>"
            f"{metadata['best_model']}. The dashboard uses {metadata['confidence_method'].lower()}</p></div>",
            unsafe_allow_html=True,
        )

    if run_analysis:
        try:
            result = assess_comment(comment, model, vectorizer)
            add_to_history([result])
            title = "Potentially toxic pattern detected" if result["prediction"] == "Toxic" else "No strong toxic pattern detected"
            st.markdown(
                f"<div class='result-card' style='--accent:{result['accent']};'><h3>{title}</h3>"
                f"<p>{result['message']}</p></div>",
                unsafe_allow_html=True,
            )
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Prediction", str(result["prediction"]))
            metric_two.metric("Toxic probability", percentage(float(result["toxicity_probability"])))
            metric_three.metric("Model confidence", percentage(float(result["confidence"])))
            metric_four.metric("Review priority", str(result["review_priority"]))
            st.progress(float(result["toxicity_probability"]), text="Toxic-class probability")

            with st.expander("See the preprocessing and model explanation"):
                st.write(f"**Cleaned text:** {result['cleaned_text']}")
                st.write(f"**Model used:** {metadata['best_model']}")
                st.write(f"**Probability method:** {metadata['confidence_method']}")
                evidence = feature_evidence(model, vectorizer, str(result["cleaned_text"]))
                if evidence:
                    st.write(
                        "**Present features with positive model weight:** "
                        + ", ".join(f"'{term}'" for term in evidence)
                    )
                    st.caption("These are learned feature contributions for this prediction, not fixed moderation rules.")
                else:
                    st.caption(
                        "This model does not expose direct feature coefficients. Its prediction is still based on "
                        "the learned TF-IDF feature pattern."
                    )
        except ValueError as error:
            st.warning(str(error))

    history = st.session_state.get("history", [])
    if history:
        st.markdown("### Recent checks in this session")
        history_frame = pd.DataFrame(history)[
            ["comment", "prediction", "toxicity_probability", "review_priority"]
        ].copy()
        history_frame["toxicity_probability"] = history_frame["toxicity_probability"].map(percentage)
        history_frame.columns = ["Comment", "Prediction", "Toxic probability", "Review priority"]
        st.dataframe(history_frame.head(8), hide_index=True, width="stretch")


def show_batch_analysis(model: object, vectorizer: object) -> None:
    page_header(
        "Batch workflow",
        "Analyze several comments",
        "Use a CSV with a text column or paste one comment per line. The app processes at most 1,000 comments "
        "at a time so the demonstration remains responsive.",
    )
    input_mode = st.radio("Input method", ("Upload CSV", "Paste comments"), horizontal=True)
    comments: list[str] = []
    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded is not None:
            try:
                batch = pd.read_csv(uploaded)
                if batch.empty:
                    st.warning("The uploaded CSV has no rows.")
                else:
                    default_index = list(batch.columns).index("comment_text") if "comment_text" in batch.columns else 0
                    column = st.selectbox("Choose the column containing comments", batch.columns, index=default_index)
                    comments = batch[column].dropna().astype(str).tolist()
                    st.caption(f"{len(comments):,} non-empty values found in the '{column}' column.")
            except Exception as error:
                st.error(f"Could not read the CSV: {error}")
    else:
        pasted = st.text_area(
            "One comment per line",
            height=220,
            placeholder="First comment...\nSecond comment...",
        )
        comments = [line.strip() for line in pasted.splitlines() if line.strip()]

    if st.button("Run batch analysis", type="primary"):
        if not comments:
            st.warning("Add at least one comment first.")
        else:
            records = []
            skipped = 0
            for comment in comments[:1000]:
                try:
                    records.append(assess_comment(comment, model, vectorizer))
                except ValueError:
                    skipped += 1
            if not records:
                st.warning("No usable comments were found.")
            else:
                add_to_history(records)
                result_frame = pd.DataFrame(records)[
                    ["comment", "prediction", "toxicity_probability", "confidence", "review_priority"]
                ].copy()
                result_frame["toxicity_probability"] = result_frame["toxicity_probability"].round(4)
                result_frame["confidence"] = result_frame["confidence"].round(4)
                st.success(f"Analyzed {len(result_frame):,} comments." + (f" Skipped {skipped} empty comment(s)." if skipped else ""))
                st.dataframe(result_frame, hide_index=True, width="stretch")
                st.download_button(
                    "Download predictions as CSV",
                    data=result_frame.to_csv(index=False).encode("utf-8"),
                    file_name="toxiguard_batch_predictions.csv",
                    mime="text/csv",
                )


def format_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Format a copy for display without altering the values used in charts."""
    visible = comparison.copy()
    for metric in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"):
        visible[metric] = visible[metric].map(percentage)
    return visible


def show_model_performance(metadata: dict, comparison: pd.DataFrame, report: pd.DataFrame) -> None:
    page_header(
        "Measured experiment",
        "Model performance",
        "All six candidates used the same cleaned-text TF-IDF features and the same stratified held-out test set. "
        "The project selects a model from results rather than assuming a model will win.",
    )
    winner = comparison.iloc[0]
    first, second, third, fourth = st.columns(4)
    first.metric("Final model", str(winner["Model"]).replace(" (Calibrated)", ""))
    second.metric("Toxic F1", percentage(winner["F1 Score"]))
    third.metric("Toxic recall", percentage(winner["Recall"]))
    fourth.metric("ROC-AUC", percentage(winner["ROC-AUC"]))

    st.markdown(
        "<div class='note'><strong>Interpretation:</strong> accuracy measures all correct predictions; precision "
        "asks how many flagged comments were actually toxic; recall asks how many toxic comments were found; "
        "F1 balances precision and recall. ROC-AUC measures how well the probability scores separate both classes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Results table")
    st.dataframe(format_comparison(comparison), hide_index=True, width="stretch")

    st.markdown("### Compare one metric at a time")
    selected_metric = st.selectbox(
        "Metric",
        ("F1 Score", "Recall", "Precision", "Accuracy", "ROC-AUC"),
        format_func=lambda value: value.replace(" Score", ""),
    )
    chart_data = comparison.sort_values(selected_metric, ascending=False).set_index("Model")[[selected_metric]]
    st.bar_chart(chart_data, height=300, width="stretch")

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("### Deployed model confusion matrix")
        matrix_path = OUTPUTS / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption=f"Actual versus predicted labels — {metadata['best_model']}")
        st.caption(
            "Rows are actual labels and columns are predicted labels. The bottom-left cell represents toxic "
            "comments the model missed; the top-right cell represents non-toxic comments incorrectly flagged."
        )
    with right:
        st.markdown("### Classification report")
        st.dataframe(report.round(3), width="stretch")
        st.markdown(
            "<div class='warning-note'><strong>Important trade-off:</strong> Complement Naive Bayes had the "
            "highest recall in this run, but Logistic Regression had a stronger F1 score. The chosen model "
            "therefore reduces the overall imbalance between missed toxic comments and false alerts.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Algorithm guide")
    selected_model = st.selectbox("Choose a model to explain", list(comparison["Model"]))
    guide = MODEL_GUIDE[selected_model]
    metrics = comparison.loc[comparison["Model"] == selected_model].iloc[0]
    st.markdown(
        f"<div class='algorithm-card'><div class='family'>{guide['family']}</div><h4>{selected_model}</h4>"
        f"<p><strong>How it works:</strong> {guide['how']}</p>"
        f"<p><strong>What this experiment showed:</strong> {guide['trade_off']}</p></div>",
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", percentage(metrics["Accuracy"]))
    b.metric("Precision", percentage(metrics["Precision"]))
    c.metric("Recall", percentage(metrics["Recall"]))
    d.metric("F1 score", percentage(metrics["F1 Score"]))

    with st.expander("Experiment design and reproducibility"):
        st.markdown(
            f"- **Source data after cleaning:** {metadata['dataset_rows_after_cleaning']:,} comments\n"
            f"- **Stratified experiment subset:** {metadata['rows_used_for_training']:,} comments\n"
            f"- **Held-out test set:** {metadata['test_rows']:,} comments\n"
            f"- **TF-IDF features:** {metadata['feature_count']:,} unigrams and bigrams\n"
            "- **Split:** 80% training / 20% testing, random state 42\n"
            f"- **Selection rule:** {metadata['selection_rule']}"
        )


def show_dataset_insights(metadata: dict, top_words: pd.DataFrame) -> None:
    page_header(
        "Data understanding",
        "Dataset insights",
        "These figures are generated from the cleaned Jigsaw toxic-comment CSV before selecting the smaller "
        "stratified experiment subset.",
    )
    metrics = (
        ("Total comments", f"{metadata['dataset_rows_after_cleaning']:,}", "After removing empty and duplicate rows"),
        ("Toxic", f"{metadata['toxic_comments']:,}", "Binary target value: 1"),
        ("Non-toxic", f"{metadata['non_toxic_comments']:,}", "Binary target value: 0"),
    )
    for column, metric in zip(st.columns(3), metrics):
        column.markdown(stat_card(*metric), unsafe_allow_html=True)

    st.markdown("### Exploratory analysis")
    eda_path = OUTPUTS / "eda_plots.png"
    if eda_path.exists():
        st.image(str(eda_path), caption="Class balance, available label totals and text-length patterns")
    st.markdown(
        "<div class='note'><strong>What the imbalance means:</strong> non-toxic comments are much more common. "
        "A model can therefore look strong on accuracy while missing toxic comments. This is why toxic-class "
        "precision, recall and F1 are reported separately.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        word_path = OUTPUTS / "top_toxic_words.png"
        if word_path.exists():
            st.image(str(word_path), caption="Frequently occurring terms among toxic-labelled comments")
    with right:
        st.markdown("### How to explain this page")
        st.markdown(
            "- The project predicts the binary toxic field, not every available label.\n"
            "- severe_toxic, obscene, threat, insult and identity_hate remain useful for EDA.\n"
            "- Frequent terms reveal dataset patterns only; they are not a list of fixed moderation rules.\n"
            "- TF-IDF gives more weight to terms that are distinctive, not merely frequent."
        )
        st.markdown("#### Frequent terms in toxic comments")
        st.dataframe(top_words.sort_values("count", ascending=False).head(10), hide_index=True, width="stretch")


def show_about_project(metadata: dict, comparison: pd.DataFrame) -> None:
    page_header(
        "Documentation",
        "About ToxiGuard",
        "A clear project narrative for an MCA minor-project demonstration: the problem, data preparation, "
        "features, algorithms, measured selection and responsible-use limits.",
    )
    winner = comparison.iloc[0]
    left, right = st.columns(2)
    with left:
        st.markdown("### Problem and objective")
        st.write(
            "Online discussion spaces can contain harassment, insults and harmful language. ToxiGuard uses "
            "supervised machine learning to flag text that may deserve a moderator's attention."
        )
        st.markdown("### Dataset and target")
        st.write(
            "The project uses the Jigsaw Toxic Comment Classification CSV. comment_text is the input and "
            "the binary toxic field is the target: 1 for toxic and 0 for non-toxic."
        )
        st.markdown("### Preprocessing")
        st.markdown(
            "1. Remove rows with missing comments and duplicate comments.\n"
            "2. Lowercase text.\n"
            "3. Remove URLs and Wikipedia-style markup.\n"
            "4. Keep words, numbers, contractions and negation; normalise spacing.\n"
            "5. Remove comments that become empty after cleaning."
        )
    with right:
        st.markdown("### TF-IDF in simple terms")
        st.write(
            "TF-IDF turns each comment into a numeric vector. A term receives more weight when it is important "
            "within one comment but not common across every comment. This project uses single words and two-word phrases."
        )
        st.markdown("### Evaluation")
        st.write(
            "The data is split into stratified training and test sets. Each model sees the same training features "
            "and is evaluated on the same unseen test comments using accuracy, toxic precision, toxic recall, F1 and ROC-AUC."
        )
        st.markdown("### Measured conclusion")
        st.write(
            f"{winner['Model']} was selected because it achieved the highest toxic-class F1 score "
            f"({percentage(winner['F1 Score'])}) in the real experiment. It was not chosen in advance."
        )

    st.markdown("### The six algorithms")
    for model_name in comparison["Model"]:
        guide = MODEL_GUIDE[model_name]
        st.markdown(
            f"<div class='algorithm-card'><div class='family'>{guide['family']}</div><h4>{model_name}</h4>"
            f"<p><strong>Core idea:</strong> {guide['how']}</p><p><strong>Reason to include it:</strong> "
            f"{guide['trade_off']}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Scope, limitations and responsible use")
    st.markdown(
        "<div class='warning-note'><strong>Use ToxiGuard as decision support only.</strong> Toxicity can depend "
        "on sarcasm, quotations, conversation history, spelling variation and local language. The model is "
        "English-language binary classification and should not be the sole basis for punitive or safety-critical action.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Possible future extensions: multi-label prediction for threat and insult, multilingual training data, "
        "context-aware transformer models, threshold tuning with moderator feedback, and a human-in-the-loop review workflow."
    )


def main() -> None:
    st.set_page_config(
        page_title="ToxiGuard | Moderation Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    try:
        model, vectorizer, metadata, comparison, report, top_words = load_project_assets()
    except FileNotFoundError as error:
        st.error(
            "Training artifacts are missing: "
            f"{error}. Run python train_model.py from the project folder, then restart Streamlit."
        )
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown(
            "<div class='brand-lockup'><div class='brand-mark'>TG</div><div class='brand-name'>ToxiGuard</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='side-caption'>Toxic-comment detection<br>for moderation support</div>", unsafe_allow_html=True)
        section = st.radio("Pages", NAVIGATION, label_visibility="collapsed")
        st.divider()
        st.caption("Deployed model")
        st.markdown(f"<div class='side-model'>{metadata['best_model']}</div>", unsafe_allow_html=True)
        st.caption("Chosen by toxic-class F1 on the held-out test set.")

    st.markdown(
        "<section class='hero'><div class='eyebrow'>Machine-learning moderation prototype</div>"
        "<h1>ToxiGuard</h1><p>Cyberbullying and toxic-comment detection using a transparent, "
        "measured traditional machine-learning workflow.</p></section>",
        unsafe_allow_html=True,
    )
    if section == "Overview":
        show_overview(metadata, comparison)
    elif section == "Analyze comment":
        show_analyze_comment(model, vectorizer, metadata)
    elif section == "Batch analysis":
        show_batch_analysis(model, vectorizer)
    elif section == "Model performance":
        show_model_performance(metadata, comparison, report)
    elif section == "Dataset insights":
        show_dataset_insights(metadata, top_words)
    else:
        show_about_project(metadata, comparison)


if __name__ == "__main__":
    main()
