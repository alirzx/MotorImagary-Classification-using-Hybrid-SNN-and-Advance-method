# app_st.py
import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support

# ----------------------------
# Configuration
# ----------------------------
API_BASE_URL = "http://localhost:8080"
SUBJECT_IDS = list(range(1, 10))
EEG_SHAPE = (22, 1000)

# ----------------------------
# Helper Functions
# ----------------------------
def generate_dummy_input():
    """Generate a random dummy EEG sample."""
    return np.random.randn(*EEG_SHAPE).tolist()

def call_predict_subject(subject_id: int, eeg_data):
    url = f"{API_BASE_URL}/predict/{subject_id}"
    payload = {"data": eeg_data}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

def call_predict_all_subjects(eeg_data):
    url = f"{API_BASE_URL}/predict_all"
    payload = {"data": eeg_data}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

def fetch_real_data(subject_id=None):
    """Fetch real EEG data automatically from the API."""
    if subject_id is not None:
        url = f"{API_BASE_URL}/fetch_subject/{subject_id}"
    else:
        url = f"{API_BASE_URL}/fetch_all"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        st.error(f"API Error fetching real data: {r.status_code} - {r.text}")
        return None

# ----------------------------
# New: batch predict via API (one call per subject)
# ----------------------------
def call_predict_batch_subject(subject_id: int, windows_list):
    """Windows_list: list of windows each shaped (22,1000)"""
    url = f"{API_BASE_URL}/predict_batch/{subject_id}"
    payload = {"data": windows_list}
    start = time.time()
    r = requests.post(url, json=payload, timeout=60)  # allow timeout
    elapsed = time.time() - start
    if r.status_code == 200:
        res = r.json()
        res["_http_time_s"] = elapsed
        return res
    else:
        st.error(f"Batch API Error: {r.status_code} - {r.text}")
        return None


def call_dummy_predict_all():
    url = f"{API_BASE_URL}/dummy_predict_all"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Dummy API Error: {response.status_code} - {response.text}")
        return None

def call_dummy_predict_subject(subject_id: int):
    """Call the dummy_predict_subject endpoint for a single subject."""
    url = f"{API_BASE_URL}/dummy_predict/{subject_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Dummy API Error: {response.status_code} - {response.text}")
        return None

    

# ----------------------------
# Visualization Helpers
# ----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pandas as pd

# ----------------------------
# Raw EEG Signal Plot
# ----------------------------
def plot_raw_eeg_signal(eeg_2d, channels_to_plot=16, title="Raw EEG Signal"):
    """
    eeg_2d: (num_channels, num_samples) numpy array
    channels_to_plot: number of top channels to show
    """
    num_channels = min(channels_to_plot, eeg_2d.shape[0])
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(num_channels):
        ax.plot(eeg_2d[i], label=f"Ch {i+1}")
    ax.set_title(f"{title} — first {num_channels} channels", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    st.pyplot(fig)


# ----------------------------
# Single Subject Probability Bar
# ----------------------------
def plot_probabilities_single(prob_list, subject_id, class_labels=["Class 0", "Class 1"]):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("Set2", len(prob_list))
    ax.bar(class_labels, prob_list, color=colors, edgecolor="black")
    ax.set_ylim(0, 1)
    for i, p in enumerate(prob_list):
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title(f"Prediction Probabilities — Subject {subject_id}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=12)
    st.pyplot(fig)


# ----------------------------
# Multi-subject Probability Bar
# ----------------------------
def plot_multisubject_probabilities(results_dict, class_labels=["Class 0", "Class 1"]):
    subjects = list(results_dict.keys())
    probs = np.array([results_dict[s]["probabilities"] for s in subjects])
    
    x = np.arange(len(subjects))
    width = 0.3
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("Set2", len(class_labels))
    
    for i, cls in enumerate(class_labels):
        ax.bar(x + i*width - width/2, probs[:, i], width=width, label=cls, color=colors[i], edgecolor="black")
        for j, p in enumerate(probs[:, i]):
            ax.text(x[j] + i*width - width/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=10, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects], rotation=45)
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Probabilities Across All Subjects", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)


# ----------------------------
# Heatmap of Probabilities
# ----------------------------
def plot_prob_heatmap(results_dict, class_labels=["Class 0", "Class 1"], title="Subject-wise Probability Heatmap"):
    subjects = list(results_dict.keys())
    all_probs = np.array([results_dict[s]["probabilities"] for s in subjects])
    df = pd.DataFrame(all_probs, index=[f"Subject {s}" for s in subjects], columns=class_labels)
    
    fig, ax = plt.subplots(figsize=(len(class_labels)*2.5, len(subjects)*0.6 + 2))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5,
                annot_kws={"fontsize":10, "weight":"bold"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Classes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Subjects", fontsize=12, fontweight="bold")
    st.pyplot(fig)


# ----------------------------
# Radar Plot (Class 1 Confidence)
# ----------------------------
def plot_radar_multisubject(results_dict, class_index=1, title="Class 1 Confidence Radar Plot"):
    subjects = list(results_dict.keys())
    probs = [results_dict[s]["probabilities"][class_index] for s in subjects]
    
    # Radar requires circular closure
    probs += probs[:1]
    angles = np.linspace(0, 2*np.pi, len(subjects), endpoint=False).tolist()
    angles += angles[:1]
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, probs, marker='o', linewidth=2)
    ax.fill(angles, probs, alpha=0.25)
    
    ax.set_xticks(np.linspace(0, 2*np.pi, len(subjects), endpoint=False))
    ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.grid(True)
    st.pyplot(fig)


# ----------------------------
# Binary Prediction Indicator
# ----------------------------
def display_binary_indicator(pred, title="Prediction Result"):
    st.markdown(f"### {title}")
    if pred == 1:
        st.success("**Prediction: Class 1**")
        st.progress(100)
    else:
        st.warning("**Prediction: Class 0**")
        st.progress(0)


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", normalize=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"
    
    num_classes = cm.shape[0]
    # Dynamically scale figure: width = number of classes * factor
    fig_width = max(6, num_classes * 1.2)
    fig_height = max(4, num_classes * 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(cm_display, annot=False, fmt=fmt, cmap="Blues", cbar=True,
                square=False, linewidths=0.5, ax=ax)
    
    # Overlay numbers manually to guarantee visibility
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text_color = "white" if value > cm_display.max() / 2 else "black"
            ax.text(j + 0.5, i + 0.5, f"{value:{fmt}}",
                    ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")
    
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, rotation=0, fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)


# ----------------------------
# Streamlit Layout
# ----------------------------
st.set_page_config(page_title="BCI Motor Imagery Classifier", layout="wide")
st.title("🧠 BCI Motor Imagery Classification Dashboard")
st.markdown(
    "A clean, interactive interface powered by **FastAPI + Streamlit**. Fetch real EEG automatically or test with dummy data."
)

tab1, tab2 = st.tabs(["Dummy Input", "Real EEG Data"])

# ====================================================
# 🔹 DUMMY INPUT TAB
# ====================================================
with tab1:
    st.subheader("Generate and Test Dummy EEG Input")

    if st.button("Generate Dummy EEG"):
        dummy_input = generate_dummy_input()
        st.session_state["dummy_input"] = dummy_input
        st.success("Dummy EEG generated!")
        st.write(f"Shape: {np.array(dummy_input).shape}")
        st.dataframe(pd.DataFrame(dummy_input), height=300)
        plot_raw_eeg_signal(np.array(dummy_input))

    if "dummy_input" in st.session_state:
        st.markdown("### Run Inference")
        subject_id = st.selectbox("Select Subject", SUBJECT_IDS, key="dummy_subject_select")
        col1, col2 = st.columns(2)

        # Single subject
        with col1:
            if st.button("Predict Single Subject (Dummy)"):
                result = call_dummy_predict_subject(subject_id)
                if result:
                    st.json(result)
                    plot_probabilities_single(result["probabilities"], subject_id)
                    display_binary_indicator(result["prediction"])

        # All subjects
        with col2:
            if st.button("Predict All Subjects (Dummy)"):
                results = call_dummy_predict_all()
                if results:
                    st.json(results)
                    plot_multisubject_probabilities(results)
                    plot_prob_heatmap(results)
                    plot_radar_multisubject(results)
# ====================================================
# 🔹 REAL EEG DATA TAB (Merged fetch + predict + evaluate + dummy-style plots)
# ====================================================
with tab2:
    st.subheader("Fetch Real EEG Data Automatically (and evaluate)")

    # ---------------- SINGLE SUBJECT ----------------
    subject_id = st.selectbox("Select Subject to Fetch", SUBJECT_IDS, key="real_subject_select")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch & Predict Single Subject"):
            # Fetch subject windows
            data = fetch_real_data(subject_id)
            if not data or "data" not in data:
                st.error("Failed to fetch subject data.")
            else:
                X = np.array(data["data"])  # shape (N,22,1000)
                Y = np.array(data["labels"])
                st.session_state["real_input"] = X.tolist()

                st.success(f"Fetched real EEG: subject {subject_id}, windows={X.shape[0]}")
                # Flatten for display only
                eeg_display = X.reshape(X.shape[0], -1)
                st.dataframe(pd.DataFrame(eeg_display), height=300)

                # For raw plot, show representative first window only (avoid massive images)
                rep_window = X[0]
                plot_raw_eeg_signal(rep_window)

                # Call batch predict endpoint (one HTTP call)
                res = call_predict_batch_subject(subject_id, X.tolist())
                if not res:
                    st.error("Prediction failed.")
                else:
                    preds = np.array(res["predictions"])
                    probs = np.array(res["probabilities"])
                    if probs.ndim == 1:
                        probs = np.vstack([np.array(p) for p in res["probabilities"]])

                   
                    # Summary
                    acc = (preds == Y).mean() if len(Y) > 0 else None
                    st.markdown(f"### Subject {subject_id} results — accuracy: **{acc:.3f}** (count={res['count']})")
                    st.write(f"API batch time: {res.get('batch_infer_time_s', 0):.3f}s, per-window avg: {res.get('per_window_avg_ms', 0):.2f}ms, cache_hits: {res.get('cache_hits', 0)}")
                    
                    # show first few predictions
                    df_small = pd.DataFrame({
                        "true": Y.tolist(),
                        "pred": preds.tolist(),
                        "prob_class0": probs[:, 0].tolist(),
                        "prob_class1": probs[:, 1].tolist()
                    })
                    st.dataframe(df_small.head(50))

                    # Confusion matrix
                    cm = confusion_matrix(Y, preds)
                    plot_confusion_matrix(cm, title=f"Confusion Matrix Subject {subject_id}")

                    # Per-class metrics
                    prec, rec, f1, supp = precision_recall_fscore_support(Y, preds, zero_division=0, average=None)
                    per_class_df = pd.DataFrame({
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "support": supp
                    }, index=["class0", "class1"])
                    st.markdown("**Per-class metrics**")
                    st.dataframe(per_class_df)

                    # ROC (needs prob of positive class)
                    probs_class1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    try:
                        fpr, tpr, _ = roc_curve(Y, probs_class1)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                        ax.plot([0, 1], [0, 1], linestyle="--")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve")
                        ax.legend()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"ROC could not be computed: {e}")

                    # Probability distribution
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(probs_class1, kde=True, ax=ax)
                    ax.set_title("Predicted Class-1 Probability Distribution")
                    st.pyplot(fig)

                    # Error distribution (which windows wrong)
                    correctness = (preds == Y).astype(int)
                    fig, ax = plt.subplots(figsize=(12, 2))
                    ax.plot(correctness, marker="o", linestyle="None")
                    ax.set_title("Correct (1) vs Incorrect (0) (window index)")
                    st.pyplot(fig)

                     # ----------------- DUMMY-STYLE PLOTS FOR REAL -----------------
                    probs_mean = probs.mean(axis=0).tolist()
                    plot_probabilities_single(probs_mean, subject_id)
                    results_dict_single = {subject_id: {"probabilities": probs_mean}}
                    plot_prob_heatmap(results_dict_single)
                    plot_radar_multisubject(results_dict_single)
                    pred_majority = int(np.round(preds.mean()))
                    display_binary_indicator(pred_majority, title=f"Binary Indicator — Subject {subject_id}")


    # ---------------- ALL SUBJECTS ----------------
    with colB:
        if st.button("Fetch & Predict All Subjects"):
            data = fetch_real_data()
            if not data:
                st.error("Failed to fetch all subjects.")
            else:
                overall_preds = []
                overall_labels = []
                subj_metrics = {}
                timing_info = {}
                results = {}

                for subj, subj_data in data.items():
                    X = np.array(subj_data["data"])  # (N, 22, 1000)
                    Y = np.array(subj_data["labels"])
                    st.markdown(f"## Subject {subj} — windows {X.shape[0]}")

                    # flatten for display
                    eeg_display = X.reshape(X.shape[0], -1)
                    st.dataframe(pd.DataFrame(eeg_display).head(50), height=200)

                    # representative raw window
                    rep_window = X[0]
                    plot_raw_eeg_signal(rep_window)

                    # call batch endpoint
                    res = call_predict_batch_subject(subj, X.tolist())
                    if not res:
                        st.error(f"Prediction failed for subject {subj}")
                        continue

                    preds = np.array(res["predictions"])
                    probs = np.array(res["probabilities"])
                    if probs.ndim == 1:
                        probs = np.vstack([np.array(p) for p in res["probabilities"]])

                    # ----------------- DUMMY-STYLE PLOTS PER SUBJECT -----------------
                    probs_mean = probs.mean(axis=0).tolist()
                    results_dict_single = {subj: {"probabilities": probs_mean}}
                    plot_probabilities_single(probs_mean, subj)
                    
                    plot_prob_heatmap(results_dict_single)
                    plot_radar_multisubject(results_dict_single)
                    pred_majority = int(np.round(preds.mean()))
                    display_binary_indicator(pred_majority, title=f"Binary Indicator — Subject {subj}")

                    # per-subject metrics
                    acc = (preds == Y).mean() if len(Y) > 0 else None
                    subj_metrics[subj] = {"accuracy": float(acc), "count": int(res["count"]), "cache_hits": int(res["cache_hits"])}
                    timing_info[subj] = {"batch_time_s": res.get("batch_infer_time_s", 0), "per_window_ms": res.get("per_window_avg_ms", 0), "http_time_s": res.get("_http_time_s", 0)}

                    # collect for combined metrics
                    overall_preds.extend(preds.tolist())
                    overall_labels.extend(Y.tolist())

                    # quick confusion matrix per subject
                    cm = confusion_matrix(Y, preds)
                    plot_confusion_matrix(cm, title=f"Confusion Matrix Subject {subj}")

                    # store result
                    results[subj] = {
                        "preds": preds.tolist(),
                        "probs": probs.tolist(),
                        "labels": Y.tolist()
                    }

                # Combined evaluation
                overall_preds = np.array(overall_preds)
                overall_labels = np.array(overall_labels)
                if overall_preds.size > 0:
                    combined_acc = (overall_preds == overall_labels).mean()
                    st.subheader(f"Combined accuracy across fetched subjects: {combined_acc:.3f}")

                    cm_all = confusion_matrix(overall_labels, overall_preds)
                    plot_confusion_matrix(cm_all, title=f"Combined Confusion Matrix")

                    # combined ROC using all positive probs (approx)
                    all_probs_class1 = []
                    for subj, r in results.items():
                        arr = np.array(r["probs"])
                        if arr.size and arr.shape[1] > 1:
                            all_probs_class1.extend(arr[:, 1].tolist())
                    try:
                        if len(all_probs_class1) == overall_labels.size:
                            fpr, tpr, _ = roc_curve(overall_labels, np.array(all_probs_class1))
                            roc_auc = auc(fpr, tpr)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                            ax.plot([0, 1], [0, 1], linestyle="--")
                            ax.set_title("Combined ROC")
                            ax.legend()
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Combined ROC error: {e}")

                    # Combined dummy-style multi-subject plots
                    combined_results_dict = {s: {"probabilities": np.array(r["probs"]).mean(axis=0).tolist()} for s, r in results.items()}
                    plot_multisubject_probabilities(combined_results_dict)
                    plot_prob_heatmap(combined_results_dict)
                    plot_radar_multisubject(combined_results_dict)

                    # show per-subject summary table
                    df_subj = pd.DataFrame.from_dict(subj_metrics, orient="index")
                    st.markdown("### Per-subject summary")
                    st.dataframe(df_subj)

                    # response time summary
                    df_time = pd.DataFrame.from_dict(timing_info, orient="index")
                    st.markdown("### Response time summary (per-subject)")
                    st.dataframe(df_time)

                    # error distribution across all windows
                    correctness = (overall_preds == overall_labels).astype(int)
                    fig, ax = plt.subplots(figsize=(12, 2))
                    ax.plot(correctness, marker="o", linestyle="None")
                    ax.set_title("Combined Correct (1) vs Incorrect (0) across windows")
                    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("✨ Built with FastAPI + Streamlit for real-time BCI inference.")
