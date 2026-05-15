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
