"""
backend/registry/nodes/bci.py

Brain-Computer Interface (BCI) node descriptors — Tier 5.

Pipeline flow:
  epochs (2+ classes) → compute_csp           → classify_lda → plot_roc_curve
  epochs (2+ classes) → extract_epoch_features → classify_lda → plot_roc_curve

The `features` handle type carries a dict with:
  - "X": list (n_epochs x n_features feature matrix, JSON-serializable)
  - "labels": list (class labels per epoch)
  - "label_names": dict (event_id → name mapping)

Soft dependency on scikit-learn: server starts without it, BCI nodes raise
a clear error at execution time.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)

# ---------------------------------------------------------------------------
# Soft sklearn import (same pattern as connectivity.py with mne-connectivity)
# ---------------------------------------------------------------------------

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, roc_curve, auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _require_sklearn():
    """Raise a helpful error if scikit-learn is not installed."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for BCI nodes. "
            "Install it with: pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _figure_to_base64_png(fig: plt.Figure) -> str:
    """Converts a Matplotlib Figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Node 1: compute_csp
# ---------------------------------------------------------------------------

def _execute_compute_csp(
    epochs: "mne.Epochs",
    params: dict,
) -> dict:
    """Extract Common Spatial Pattern features from epochs."""
    _require_sklearn()
    import mne.decoding

    n_components = int(params.get("n_components", 4))

    labels = epochs.events[:, -1]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"CSP requires at least 2 event classes, found {len(unique_labels)}. "
            "Use event_id in Epoch by Events to select 2+ event types."
        )

    csp = mne.decoding.CSP(
        n_components=n_components, log=True, norm_trace=False,
    )
    X = csp.fit_transform(epochs.get_data(), labels)

    label_names = (
        {int(v): k for k, v in epochs.event_id.items()}
        if epochs.event_id else {}
    )

    return {
        "X": X.tolist(),
        "labels": labels.tolist(),
        "label_names": label_names,
        "n_components": n_components,
        "n_epochs": len(labels),
    }


COMPUTE_CSP = NodeDescriptor(
    node_type="compute_csp",
    display_name="Common Spatial Patterns (CSP)",
    category="BCI",
    description=(
        "Extracts Common Spatial Pattern features from multi-class epochs. "
        "CSP finds spatial filters that maximize variance for one class while "
        "minimizing it for the other — the standard feature extraction method "
        "for motor imagery BCI. Requires epochs with at least 2 event classes. "
        "Output is a feature matrix for downstream classification."
    ),
    tags=["csp", "bci", "motor-imagery", "spatial-filter", "features", "classification"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs (2+ classes)"),
    ],
    outputs=[
        HandleSchema(id="features_out", type="features", label="CSP Features"),
    ],
    parameters=[
        ParameterSchema(
            name="n_components",
            label="Components",
            type="int",
            default=4,
            min=2,
            max=20,
            description=(
                "Number of CSP components to extract. Uses the top and bottom "
                "components (most discriminative spatial filters)."
            ),
        ),
    ],
    execute_fn=_execute_compute_csp,
    code_template=lambda p: f'csp = mne.decoding.CSP(n_components={p.get("n_components", 4)}, log=True, norm_trace=False)\nX = csp.fit_transform(epochs.get_data(), labels)',
    methods_template=lambda p: f'Common Spatial Pattern (CSP) features ({p.get("n_components", 4)} components) were extracted to maximize class discriminability (Ramoser et al., 2000; MNE-Python).',
    docs_url="https://mne.tools/stable/generated/mne.decoding.CSP.html",
)


# ---------------------------------------------------------------------------
# Node 2: extract_epoch_features
# ---------------------------------------------------------------------------

def _execute_extract_epoch_features(
    epochs: "mne.Epochs",
    params: dict,
) -> dict:
    """Extract bandpower, variance, and Hjorth parameters per epoch."""
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    labels = epochs.events[:, -1]
    label_names = (
        {int(v): k for k, v in epochs.event_id.items()}
        if epochs.event_id else {}
    )

    features_list = []
    for epoch_data in data:
        feats = []
        for ch_data in epoch_data:
            # Variance
            var0 = float(np.var(ch_data))
            feats.append(var0)

            # Hjorth mobility and complexity
            diff1 = np.diff(ch_data)
            diff2 = np.diff(diff1)
            var1 = float(np.var(diff1))
            var2 = float(np.var(diff2))
            mobility = float(np.sqrt(var1 / (var0 + 1e-30)))
            complexity = float(
                np.sqrt(var2 / (var1 + 1e-30)) / (mobility + 1e-30)
            )
            feats.extend([mobility, complexity])
        features_list.append(feats)

    X = np.array(features_list)  # (n_epochs, n_ch * 3)

    return {
        "X": X.tolist(),
        "labels": labels.tolist(),
        "label_names": label_names,
        "n_features": int(X.shape[1]),
        "n_epochs": int(X.shape[0]),
        "feature_names": "variance, hjorth_mobility, hjorth_complexity per channel",
    }


EXTRACT_EPOCH_FEATURES = NodeDescriptor(
    node_type="extract_epoch_features",
    display_name="Extract Epoch Features",
    category="BCI",
    description=(
        "Extracts time-domain features from each epoch for BCI classification. "
        "Computes per-channel: variance, Hjorth mobility (frequency content), "
        "and Hjorth complexity (bandwidth). Output is a feature matrix with "
        "n_channels x 3 features per epoch. Requires epochs with 2+ event "
        "classes for downstream classification."
    ),
    tags=["features", "bci", "hjorth", "variance", "epoch", "time-domain"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs (2+ classes)"),
    ],
    outputs=[
        HandleSchema(id="features_out", type="features", label="Epoch Features"),
    ],
    parameters=[],
    execute_fn=_execute_extract_epoch_features,
    code_template=lambda p: '# Per-channel features: variance, Hjorth mobility, Hjorth complexity\ndata = epochs.get_data()  # (n_epochs, n_ch, n_times)\n# ... compute features per epoch per channel',
    methods_template=lambda p: "Time-domain features (variance, Hjorth mobility, and Hjorth complexity) were extracted per channel per epoch for BCI classification.",
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html",
)


# ---------------------------------------------------------------------------
# Node 3: classify_lda
# ---------------------------------------------------------------------------

def _execute_classify_lda(
    features_dict: dict,
    params: dict,
) -> dict:
    """LDA classification with stratified cross-validation."""
    _require_sklearn()

    X = np.array(features_dict["X"])
    y = np.array(features_dict["labels"])
    n_folds = int(params.get("n_folds", 5))

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "Classification requires at least 2 classes in the labels."
        )
    if len(y) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} samples for {n_folds}-fold CV, "
            f"but only have {len(y)}."
        )

    lda = LinearDiscriminantAnalysis()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Predictions via cross-validation
    y_pred = cross_val_predict(lda, X, y, cv=skf, method="predict")
    accuracy = float(accuracy_score(y, y_pred))

    # ROC curve (binary classification only)
    roc_data = None
    if len(unique_classes) == 2:
        y_scores = cross_val_predict(
            lda, X, y, cv=skf, method="decision_function",
        )
        fpr, tpr, _ = roc_curve(y, y_scores, pos_label=unique_classes[1])
        roc_auc = float(auc(fpr, tpr))
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(roc_auc, 4),
        }

    label_names = features_dict.get("label_names", {})

    return {
        "accuracy": round(accuracy, 4),
        "n_folds": n_folds,
        "n_samples": len(y),
        "n_features": int(X.shape[1]),
        "n_classes": len(unique_classes),
        "class_labels": [
            label_names.get(int(c), str(int(c))) for c in unique_classes
        ],
        "roc": roc_data,
    }


CLASSIFY_LDA = NodeDescriptor(
    node_type="classify_lda",
    display_name="Classify (LDA)",
    category="BCI",
    description=(
        "Trains a Linear Discriminant Analysis (LDA) classifier with "
        "stratified k-fold cross-validation. Reports accuracy and, for "
        "binary classification, computes ROC curve data (FPR, TPR, AUC). "
        "Input must be a features dict from CSP or Extract Epoch Features. "
        "Output is a metrics dict suitable for reporting or plotting."
    ),
    tags=["lda", "classify", "bci", "cross-validation", "accuracy", "classification"],
    inputs=[
        HandleSchema(id="features_in", type="features", label="Features"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Classification Results"),
    ],
    parameters=[
        ParameterSchema(
            name="n_folds",
            label="CV Folds",
            type="int",
            default=5,
            min=2,
            max=20,
            description=(
                "Number of folds for stratified k-fold cross-validation. "
                "Higher values give more reliable estimates but take longer."
            ),
        ),
    ],
    execute_fn=_execute_classify_lda,
    code_template=lambda p: f'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.model_selection import StratifiedKFold, cross_val_predict\nlda = LinearDiscriminantAnalysis()\nskf = StratifiedKFold(n_splits={p.get("n_folds", 5)}, shuffle=True, random_state=42)\ny_pred = cross_val_predict(lda, X, y, cv=skf)',
    methods_template=lambda p: f'Classification was performed using Linear Discriminant Analysis with {p.get("n_folds", 5)}-fold stratified cross-validation (scikit-learn; Pedregosa et al., 2011).',
    docs_url="https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html",
)


# ---------------------------------------------------------------------------
# Node 4: plot_roc_curve
# ---------------------------------------------------------------------------

def _execute_plot_roc_curve(
    metrics: dict,
    params: dict,
) -> str:
    """Plot ROC curve from classify_lda output."""
    roc = metrics.get("roc")
    if roc is None:
        raise ValueError(
            "ROC data not available. plot_roc_curve requires binary "
            "classification results. Ensure classify_lda ran with exactly "
            "2 event classes."
        )

    fpr = roc["fpr"]
    tpr = roc["tpr"]
    roc_auc = roc["auc"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve \u2014 AUC = {roc_auc:.3f}")
    ax.legend(loc="lower right")

    # Dark theme matching existing visualization nodes
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white")

    return _figure_to_base64_png(fig)


PLOT_ROC_CURVE = NodeDescriptor(
    node_type="plot_roc_curve",
    display_name="Plot ROC Curve",
    category="Visualization",
    description=(
        "Plots the Receiver Operating Characteristic (ROC) curve with AUC "
        "from binary classification results. Visualizes the trade-off between "
        "true positive rate and false positive rate. Requires binary (2-class) "
        "classification output from Classify (LDA)."
    ),
    tags=["roc", "auc", "bci", "classification", "plot", "visualization", "curve"],
    inputs=[
        HandleSchema(id="metrics_in", type="metrics", label="Classification Metrics"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="ROC Curve"),
    ],
    parameters=[],
    execute_fn=_execute_plot_roc_curve,
    code_template=lambda p: 'from sklearn.metrics import roc_curve, auc\nfpr, tpr, _ = roc_curve(y, y_scores)\nroc_auc = auc(fpr, tpr)\nfig, ax = plt.subplots()\nax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")',
    methods_template=None,
    docs_url="https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html",
)
