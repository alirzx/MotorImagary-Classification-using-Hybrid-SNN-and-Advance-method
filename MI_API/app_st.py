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
# Visualization Helpers (Plotly)
# ----------------------------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------
# Raw EEG Signal Plot
# ----------------------------
def plot_raw_eeg_signal_plotly(eeg_2d, channels_to_plot=16):
    num_channels = min(channels_to_plot, eeg_2d.shape[0])
    fig = go.Figure()
    for i in range(num_channels):
        fig.add_trace(go.Scatter(y=eeg_2d[i], mode='lines', name=f"Ch {i+1}"))
    fig.update_layout(
        title=f"Raw EEG Signal — first {num_channels} channels",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Single Subject Probability Bar
# ----------------------------
def plot_probabilities_single(prob_list, subject_id, class_labels=["Class 0", "Class 1"]):
    fig = go.Figure()
    for i, p in enumerate(prob_list):
        fig.add_trace(go.Bar(
            x=[class_labels[i]],
            y=[p],
            text=[f"{p:.2f}"],
            textposition="outside",
            marker_color=px.colors.qualitative.Set2[i]
        ))
    fig.update_layout(
        title=f"Prediction Probabilities — Subject {subject_id}",
        yaxis=dict(range=[0, 1], title="Probability"),
        template="plotly_white",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Multi-subject Probability Bar
# ----------------------------
def plot_multisubject_probabilities(results_dict, class_labels=["Class 0", "Class 1"]):
    subjects = list(results_dict.keys())
    probs = np.array([results_dict[s]["probabilities"] for s in subjects])
    
    fig = go.Figure()
    for i, cls in enumerate(class_labels):
        fig.add_trace(go.Bar(
            x=[f"S{s}" for s in subjects],
            y=probs[:, i],
            name=cls,
            text=[f"{v:.2f}" for v in probs[:, i]],
            textposition="outside",
            marker_color=px.colors.qualitative.Set2[i]
        ))
    fig.update_layout(
        title="Prediction Probabilities Across All Subjects",
        yaxis=dict(range=[0, 1], title="Probability"),
        barmode='group',
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Heatmap of Probabilities
# ----------------------------
def plot_prob_heatmap(results_dict, class_labels=["Class 0", "Class 1"], title="Subject-wise Probability Heatmap"):
    subjects = list(results_dict.keys())
    all_probs = np.array([results_dict[s]["probabilities"] for s in subjects])
    df = pd.DataFrame(all_probs, index=[f"Subject {s}" for s in subjects], columns=class_labels)
    
    fig = px.imshow(
        df,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        labels=dict(x="Classes", y="Subjects", color="Probability"),
        aspect="auto"
    )
    fig.update_layout(title=title, template="plotly_white", height=300 + len(subjects)*20)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Radar Plot (Class 1 Confidence)
# ----------------------------
def plot_radar_multisubject(results_dict, class_index=1, title="Class 1 Confidence Radar Plot"):
    subjects = list(results_dict.keys())
    probs = [results_dict[s]["probabilities"][class_index] for s in subjects]
    
    # Radar requires circular closure
    probs += probs[:1]
    labels = [f"S{s}" for s in subjects]
    labels += labels[:1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs,
        theta=labels,
        fill='toself',
        name=f"Class {class_index}"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,1])),
        showlegend=False,
        title=title,
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


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


# ----------------------------
# Confusion Matrix
# ----------------------------
def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", normalize=False):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        z_text = np.round(cm_display, 2)
    else:
        cm_display = cm
        z_text = cm_display
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        text=z_text,
        texttemplate="%{text}",
        colorscale='Blues',
        showscale=True
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import numpy as np

# ----------------------------
# ROC Curve
# ----------------------------
def plot_roc_curve_plotly(y_true, probs_class1, title="ROC Curve"):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs_class1)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.3f}", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ROC could not be computed: {e}")


# ----------------------------
# Probability Distribution (Class 1)
# ----------------------------
def plot_prob_distribution_plotly(probs_class1, title="Predicted Class-1 Probability Distribution"):
    fig = px.histogram(
        x=probs_class1,
        nbins=30,
        marginal="box",
        histnorm="probability",
        labels={"x": "Probability", "y": "Density"},
        title=title
    )
    fig.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Error Distribution (Correct vs Incorrect)
# ----------------------------
def plot_error_distribution_plotly(preds, y_true, title="Correct (1) vs Incorrect (0) (window index)"):
    correctness = (preds == y_true).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=correctness,
        mode='markers',
        marker=dict(color=correctness, colorscale=["red", "green"], size=8),
        name="Correctness"
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Correct (1) / Incorrect (0)", range=[-0.1, 1.1]),
        xaxis=dict(title="Window Index"),
        template="plotly_white",
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)




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
        plot_raw_eeg_signal_plotly(np.array(dummy_input))

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
                plot_raw_eeg_signal_plotly(rep_window)

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

                    # Define class-1 probabilities first
                    probs_class1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

                    # Now call the new Plotly functions
                    plot_roc_curve_plotly(Y, probs_class1)
                    plot_prob_distribution_plotly(probs_class1)
                    plot_error_distribution_plotly(preds, Y)

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
                    plot_raw_eeg_signal_plotly(rep_window)

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

                    # # combined ROC using all positive probs (approx)
                    # First, collect all positive-class probabilities as before
                    all_probs_class1 = []
                    for subj, r in results.items():
                        arr = np.array(r["probs"])
                        if arr.size and arr.shape[1] > 1:
                            all_probs_class1.extend(arr[:, 1].tolist())

                    # Ensure array and lengths match
                    all_probs_class1 = np.array(all_probs_class1)

                    if len(all_probs_class1) == overall_labels.size:
                        # Use the same Plotly ROC function
                        plot_roc_curve_plotly(overall_labels, all_probs_class1, title="Combined ROC")


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
