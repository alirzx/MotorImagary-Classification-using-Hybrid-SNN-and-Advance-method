import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# =====================================================
# DATA
# =====================================================
df = pd.DataFrame({
    "subject": [1,2,3,4,5,6,7,8,9],
    "accuracy": [0.979,0.806,0.951,0.868,0.701,0.778,0.826,0.882,0.972],
    "precision": [0.986,0.768,0.933,0.921,0.699,0.786,0.851,0.899,1.000],
    "recall": [0.972,0.875,0.972,0.806,0.708,0.764,0.792,0.861,0.944],
    "f1": [0.979,0.818,0.952,0.859,0.703,0.775,0.820,0.879,0.971],
    "kappa": [0.958,0.611,0.903,0.736,0.403,0.556,0.653,0.764,0.944],
    "specificity": [0.986,0.736,0.931,0.931,0.694,0.792,0.861,0.903,1.000]
})

metrics = ["accuracy","precision","recall","f1","kappa","specificity"]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b"
]

EXPORT_KWARGS = dict(
    scale=3,          # HIGH resolution (≈600 DPI)
    width=1200,
    height=600
)

# =====================================================
# FIG 1 — Accuracy vs Subject
# =====================================================
fig1 = go.Figure(go.Scatter(
    x=df.subject,
    y=df.accuracy,
    mode="lines+markers",
    line=dict(color=COLORS[0], width=2),
    marker=dict(size=7)
))
fig1.update_layout(
    title="Accuracy vs Subject ID",
    xaxis_title="Subject ID",
    yaxis_title="Accuracy",
    yaxis_range=[0.65, 1.0],
    template="plotly_white"
)
fig1.write_image("Fig1_Accuracy_vs_Subject.pdf", **EXPORT_KWARGS)
fig1.write_image("Fig1_Accuracy_vs_Subject.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 2 — Box Plot (Metric Distributions)
# =====================================================
fig2 = go.Figure()
for m, c in zip(metrics, COLORS):
    fig2.add_trace(go.Box(
        y=df[m],
        name=m.capitalize(),
        marker_color=c
    ))
fig2.update_layout(
    title="Distribution of Performance Metrics Across Subjects",
    yaxis_title="Score",
    template="plotly_white"
)
fig2.write_image("Fig2_Box_Metrics.pdf", **EXPORT_KWARGS)
fig2.write_image("Fig2_Box_Metrics.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 3 — Histogram (Accuracy)
# =====================================================
fig3 = px.histogram(
    df,
    x="accuracy",
    nbins=8,
    color_discrete_sequence=[COLORS[0]],
    template="plotly_white",
    title="Accuracy Distribution Across Subjects"
)
fig3.write_image("Fig3_Hist_Accuracy.pdf", **EXPORT_KWARGS)
fig3.write_image("Fig3_Hist_Accuracy.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 4 — Grouped Bar (Metrics × Subjects)
# =====================================================
df_long = df.melt(
    id_vars="subject",
    var_name="metric",
    value_name="value"
)

fig4 = px.bar(
    df_long,
    x="subject",
    y="value",
    color="metric",
    barmode="group",
    color_discrete_sequence=COLORS,
    template="plotly_white",
    title="Subject-wise Multi-metric Performance"
)
fig4.write_image("Fig4_GroupedBars.pdf", **EXPORT_KWARGS)
fig4.write_image("Fig4_GroupedBars.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 5 — Radar Plot (ALL Subjects)
# =====================================================
# =====================================================
# FIG 5 — Radar Plot (ALL Subjects, CLOSED & POLISHED)
# =====================================================

# Metric order (fixed & explicit)
metrics = ["accuracy", "precision", "recall", "f1", "kappa", "specificity"]

# Close the loop by repeating the first metric
metrics_closed = metrics + [metrics[0]]

fig5 = go.Figure()

for _, row in df.iterrows():
    values = [row[m] for m in metrics]
    values_closed = values + [values[0]]  # 🔒 CLOSE POLYGON

    fig5.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=metrics_closed,
        name=f"S{int(row.subject)}",
        mode="lines",
        line=dict(width=1),
        opacity=0.45
    ))

# Optional: add MEAN profile for interpretability (recommended)
mean_profile = df[metrics].mean().tolist()
mean_profile_closed = mean_profile + [mean_profile[0]]

fig5.add_trace(go.Scatterpolar(
    r=mean_profile_closed,
    theta=metrics_closed,
    name="Mean Profile",
    mode="lines",
    line=dict(width=3, color="black"),
))

fig5.update_layout(
    title="Radar Plot of Performance Profiles Across Subjects",
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.3, 1.0],
            tickfont=dict(size=11),
            gridcolor="lightgray"
        ),
        angularaxis=dict(
            tickfont=dict(size=12)
        )
    ),
    template="plotly_white",
    legend=dict(
        orientation="h",
        y=-0.15,
        font=dict(size=11)
    ),
    width=900,
    height=750
)

fig5.write_image("Fig5_Radar_AllSubjects.pdf", **EXPORT_KWARGS)
fig5.write_image("Fig5_Radar_AllSubjects.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 6 — Accuracy vs Kappa
# =====================================================
fig6 = px.scatter(
    df,
    x="accuracy",
    y="kappa",
    text="subject",
    color_discrete_sequence=[COLORS[3]],
    template="plotly_white",
    title="Accuracy vs Cohen’s Kappa"
)
fig6.update_traces(textposition="top center")
fig6.write_image("Fig6_Acc_vs_Kappa.pdf", **EXPORT_KWARGS)
fig6.write_image("Fig6_Acc_vs_Kappa.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 7 — Accuracy Deviation (Burst-style)
# =====================================================
mean_acc = df.accuracy.mean()
fig7 = go.Figure(go.Bar(
    x=df.subject,
    y=df.accuracy - mean_acc,
    marker_color=COLORS[4]
))
fig7.update_layout(
    title="Accuracy Deviation from Mean (Subject Difficulty Indicator)",
    xaxis_title="Subject ID",
    yaxis_title="Δ Accuracy",
    template="plotly_white"
)
fig7.write_image("Fig7_Deviation.pdf", **EXPORT_KWARGS)
fig7.write_image("Fig7_Deviation.svg", **EXPORT_KWARGS)

# =====================================================
# FIG 8 — Mean ± Std (Robustness)
# =====================================================
means = df[metrics].mean()
stds = df[metrics].std()

fig8 = go.Figure(go.Bar(
    x=metrics,
    y=means,
    error_y=dict(type="data", array=stds),
    marker_color=COLORS
))
fig8.update_layout(
    title="Overall Robustness (Mean ± Std Across Subjects)",
    yaxis_title="Score",
    template="plotly_white"
)
fig8.write_image("Fig8_MeanStd.pdf", **EXPORT_KWARGS)
fig8.write_image("Fig8_MeanStd.svg", **EXPORT_KWARGS)

print("✅ All figures exported successfully (PDF + SVG)")
