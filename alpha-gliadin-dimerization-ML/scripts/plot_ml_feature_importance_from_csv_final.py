import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12

# ============================================================
# Input CSV files generated from ML analysis
# ============================================================

METRICS_FILE = "ml_model_metrics.csv"
LR_FILE = "logistic_regression_coefficients.csv"
RF_FILE = "random_forest_feature_importance.csv"
XGB_FILE = "xgboost_feature_importance.csv"

OUT_PNG = "Figure_11_ML_feature_importance_final.png"
OUT_PDF = "Figure_11_ML_feature_importance_final.pdf"

# ============================================================
# Load data
# ============================================================

metrics = pd.read_csv(METRICS_FILE)
lr = pd.read_csv(LR_FILE)
rf = pd.read_csv(RF_FILE)
xgb = pd.read_csv(XGB_FILE)

# ============================================================
# Clean model names
# ============================================================

model_labels = {
    "LogisticRegression": "Logistic\nregression",
    "RandomForest": "Random\nforest",
    "XGBoost": "XGBoost"
}

metrics["Model_label"] = metrics["Model"].map(model_labels).fillna(metrics["Model"])

# ============================================================
# Feature labels
# ============================================================

feature_labels = {
    "mean_CP": "Contact persistence",
    "mean_COM": "COM distance",
    "mean_HB": "Interchain H-bonds",
    "Rg_nm": "Radius of gyration",
    "Total_SASA_nm2": "Total SASA",
    "Hydrophobic_SASA_nm2": "Hydrophobic SASA",
    "hotspot_sasa_nm2": "Hotspot SASA",
    "water_hotspot_count": "Water near hotspots",
    "ethanol_hotspot_count": "Ethanol near hotspots",
    "water_ethanol_ratio": "Water/Ethanol ratio",
    "interchain_residue_pair_contacts": "Residue-pair contacts",
    "interface_residue_count": "Interface residues",
    "hotspot_any_contact_pairs": "Hotspot contacts",
    "hotspot_hotspot_contact_pairs": "Hotspot-hotspot contacts",
    "salt_bridge_count": "Salt bridges"
}

def relabel_feature_table(df):
    df = df.copy()
    df["label"] = df["feature"].map(feature_labels).fillna(df["feature"])
    return df

lr = relabel_feature_table(lr)
rf = relabel_feature_table(rf)
xgb = relabel_feature_table(xgb)

# ============================================================
# Sort all feature tables
# ============================================================

lr_plot = lr.sort_values("coefficient", ascending=True)
rf_plot = rf.sort_values("importance", ascending=True)
xgb_plot = xgb.sort_values("importance", ascending=True)

# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, wspace=0.48, hspace=0.42)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# ============================================================
# Panel A: performance metrics
# ============================================================

models = metrics["Model_label"].tolist()
x = np.arange(len(models))
width = 0.24

acc = metrics["Accuracy"].values
f1 = metrics["F1"].values
auc = metrics["ROC_AUC"].values

axA.bar(
    x - width,
    acc,
    width,
    label="Accuracy",
    edgecolor="black",
    linewidth=0.9
)

axA.bar(
    x,
    f1,
    width,
    label="F1-score",
    edgecolor="black",
    linewidth=0.9
)

axA.bar(
    x + width,
    auc,
    width,
    label="ROC-AUC",
    edgecolor="black",
    linewidth=0.9
)

axA.set_xticks(x)
axA.set_xticklabels(
    models,
    rotation=0,
    ha="center",
    fontweight="bold"
)

axA.set_ylim(0.70, 1.06)
axA.set_ylabel("Performance score", fontsize=13, fontweight="bold")

leg = axA.legend(
    frameon=False,
    fontsize=10,
    loc="upper left"
)

for t in leg.get_texts():
    t.set_fontweight("bold")

for xpos, vals in zip(
    [x - width, x, x + width],
    [acc, f1, auc]
):
    for xi, v in zip(xpos, vals):
        axA.text(
            xi,
            v + 0.012,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )

# ============================================================
# Panel B: Logistic regression coefficients
# ============================================================

bar_colors = [
    "#D55E00" if v > 0 else "#0072B2"
    for v in lr_plot["coefficient"]
]

axB.barh(
    lr_plot["label"],
    lr_plot["coefficient"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.8
)

axB.axvline(
    0,
    color="black",
    linewidth=1.2
)

axB.set_xlabel("Coefficient", fontsize=13, fontweight="bold")

# ============================================================
# Panel C: Random forest feature importance
# ============================================================

axC.barh(
    rf_plot["label"],
    rf_plot["importance"],
    edgecolor="black",
    linewidth=0.8
)

axC.set_xlabel("Feature importance", fontsize=13, fontweight="bold")

# ============================================================
# Panel D: XGBoost feature importance
# ============================================================

axD.barh(
    xgb_plot["label"],
    xgb_plot["importance"],
    edgecolor="black",
    linewidth=0.8
)

axD.set_xlabel("Feature importance", fontsize=13, fontweight="bold")

# ============================================================
# Styling
# ============================================================

titles = [
    "Model performance",
    "Logistic regression",
    "Random forest",
    "XGBoost"
]

axes = [axA, axB, axC, axD]
panel_labels = ["A", "B", "C", "D"]

for ax, panel, title in zip(axes, panel_labels, titles):
    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=8
    )

    ax.text(
        -0.14,
        1.07,
        panel,
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)

    ax.tick_params(
        width=1.2,
        length=4
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

# Smaller y-labels for feature panels
for ax in [axB, axC, axD]:
    ax.tick_params(axis="y", labelsize=10)

# ============================================================
# Save
# ============================================================

plt.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

print("\nSaved:")
print(OUT_PNG)
print(OUT_PDF)
