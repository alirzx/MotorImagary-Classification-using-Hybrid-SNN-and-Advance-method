import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ======================================================
# 1. Data
# ======================================================
data = {
    "subject": [1,2,3,4,5,6,7,8,9],
    "accuracy": [0.979166667,0.805555556,0.951388889,0.868055556,0.701388889,0.777777778,0.826388889,0.881944444,0.972222222],
    "precision": [0.985915493,0.768292683,0.933333333,0.920634921,0.698630137,0.785714286,0.850746269,0.898550725,1.0],
    "recall": [0.972222222,0.875,0.972222222,0.805555556,0.708333333,0.763888889,0.791666667,0.861111111,0.944444444],
    "f1_score": [0.979020979,0.818181818,0.952380952,0.859259259,0.703448276,0.774647887,0.820143885,0.879432624,0.971428571],
    "kappa": [0.958333333,0.611111111,0.902777778,0.736111111,0.402777778,0.555555556,0.652777778,0.763888889,0.944444444],
    "specificity": [0.986111111,0.736111111,0.930555556,0.930555556,0.694444444,0.791666667,0.861111111,0.902777778,1.0]
}

df = pd.DataFrame(data)

# ======================================================
# 2. Accuracy vs Subject ID (Main Figure)
# ======================================================
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(
    x=df["subject"],
    y=df["accuracy"],
    mode="lines+markers",
    marker=dict(size=9),
    line=dict(width=2),
    name="Accuracy"
))
fig_acc.update_layout(
    title="Accuracy vs Subject ID",
    xaxis_title="Subject ID",
    yaxis_title="Accuracy",
    yaxis_range=[0.65, 1.0],
    template="plotly_white",
    width=900,
    height=450
)
fig_acc.show()

# ======================================================
# 3. Histogram — Accuracy Distribution (Variability)
# ======================================================
fig_hist = px.histogram(
    df,
    x="accuracy",
    nbins=8,
    title="Distribution of Accuracy Across Subjects",
    template="plotly_white"
)
fig_hist.update_layout(
    xaxis_title="Accuracy",
    yaxis_title="Number of Subjects",
    width=900,
    height=450
)
fig_hist.show()

# ======================================================
# 4. Grouped Bar Chart — All Metrics per Subject
# ======================================================
df_long = df.melt(
    id_vars="subject",
    value_vars=["accuracy","precision","recall","f1_score","kappa","specificity"],
    var_name="metric",
    value_name="value"
)

fig_bar = px.bar(
    df_long,
    x="subject",
    y="value",
    color="metric",
    barmode="group",
    title="Subject-wise Performance Across Evaluation Metrics",
    template="plotly_white"
)

fig_bar.update_layout(
    xaxis_title="Subject ID",
    yaxis_title="Score",
    yaxis_range=[0.35, 1.0],
    width=1200,
    height=500,
    legend_title_text="Metric"
)
fig_bar.show()

# ======================================================
# 5. Radar Plot — Best vs Worst Subject
# ======================================================
best_subject = df.loc[df["accuracy"].idxmax()]
worst_subject = df.loc[df["accuracy"].idxmin()]

metrics = ["accuracy","precision","recall","f1_score","kappa","specificity"]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[best_subject[m] for m in metrics],
    theta=metrics,
    fill="toself",
    name=f"Best Subject (S{int(best_subject.subject)})"
))

fig_radar.add_trace(go.Scatterpolar(
    r=[worst_subject[m] for m in metrics],
    theta=metrics,
    fill="toself",
    name=f"Worst Subject (S{int(worst_subject.subject)})"
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0.3,1.0])),
    title="Radar Comparison: Best vs Worst Subject",
    template="plotly_white",
    width=700,
    height=600
)
fig_radar.show()

# ======================================================
# 6. Scatter — Accuracy vs Kappa (Agreement vs Performance)
# ======================================================
fig_scatter = px.scatter(
    df,
    x="accuracy",
    y="kappa",
    text="subject",
    title="Accuracy vs Cohen’s Kappa",
    template="plotly_white"
)

fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
fig_scatter.update_layout(
    xaxis_title="Accuracy",
    yaxis_title="Cohen’s Kappa",
    width=900,
    height=450
)
fig_scatter.show()

# ======================================================
# 7. Mean ± Std Bar Plot (Robustness Summary)
# ======================================================
metric_means = df[metrics].mean()
metric_stds = df[metrics].std()

fig_summary = go.Figure()
fig_summary.add_trace(go.Bar(
    x=metrics,
    y=metric_means,
    error_y=dict(type="data", array=metric_stds),
    name="Mean ± Std"
))

fig_summary.update_layout(
    title="Overall Performance Robustness (Mean ± Std)",
    yaxis_title="Score",
    yaxis_range=[0.3, 1.0],
    template="plotly_white",
    width=900,
    height=450
)
fig_summary.show()
