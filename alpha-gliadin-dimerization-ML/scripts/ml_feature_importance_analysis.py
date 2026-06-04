import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("updated_PCA_scores.csv")

df["target"] = (
    df["state"] == "Compact-associated"
).astype(int)

features = [
    "mean_CP",
    "mean_COM",
    "mean_HB",
    "Rg_nm",
    "Total_SASA_nm2",
    "Hydrophobic_SASA_nm2",
    "hotspot_sasa_nm2",
    "water_hotspot_count",
    "ethanol_hotspot_count",
    "water_ethanol_ratio",
    "interchain_residue_pair_contacts",
    "interface_residue_count",
    "hotspot_any_contact_pairs",
    "hotspot_hotspot_contact_pairs",
    "salt_bridge_count"
]

X = df[features]
y = df["target"]

# ==========================
# CV EVALUATION
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "LogisticRegression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",
             LogisticRegression(
                 max_iter=5000
             ))
        ]),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=500,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
}

results = []

for name, model in models.items():

    accs = []
    f1s = []
    aucs = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        prob = model.predict_proba(X_test)[:,1]

        accs.append(
            accuracy_score(y_test, pred)
        )

        f1s.append(
            f1_score(y_test, pred)
        )

        aucs.append(
            roc_auc_score(y_test, prob)
        )

    results.append([
        name,
        np.mean(accs),
        np.mean(f1s),
        np.mean(aucs)
    ])

metrics_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1",
        "ROC_AUC"
    ]
)

metrics_df.to_csv(
    "ml_model_metrics.csv",
    index=False
)

print(metrics_df)

# ==========================
# FIT FULL DATASET
# ==========================

# Logistic Regression
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model",
     LogisticRegression(max_iter=5000))
])

lr.fit(X, y)

coef = (
    lr.named_steps["model"]
    .coef_[0]
)

coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": coef
}).sort_values(
    "coefficient",
    key=np.abs,
    ascending=False
)

coef_df.to_csv(
    "logistic_regression_coefficients.csv",
    index=False
)

# RF
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(X, y)

rf_df = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

rf_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

# XGB
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X, y)

xgb_df = pd.DataFrame({
    "feature": features,
    "importance": xgb.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

xgb_df.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

# ==========================
# FIGURE 11
# ==========================

fig = plt.figure(
    figsize=(14,10)
)

# A
ax1 = plt.subplot(221)

plot_df = metrics_df.set_index("Model")

plot_df["ROC_AUC"].plot(
    kind="bar",
    ax=ax1
)

ax1.set_title(
    "Model Performance"
)

# B
ax2 = plt.subplot(222)

coef_df.head(10).iloc[::-1].plot(
    x="feature",
    y="coefficient",
    kind="barh",
    legend=False,
    ax=ax2
)

ax2.set_title(
    "Logistic Regression"
)

# C
ax3 = plt.subplot(223)

rf_df.head(10).iloc[::-1].plot(
    x="feature",
    y="importance",
    kind="barh",
    legend=False,
    ax=ax3
)

ax3.set_title(
    "Random Forest"
)

# D
ax4 = plt.subplot(224)

xgb_df.head(10).iloc[::-1].plot(
    x="feature",
    y="importance",
    kind="barh",
    legend=False,
    ax=ax4
)

ax4.set_title(
    "XGBoost"
)

plt.tight_layout()

plt.savefig(
    "Figure_11_ML_feature_importance.png",
    dpi=600
)

plt.savefig(
    "Figure_11_ML_feature_importance.pdf"
)

print("\nSaved:")
print("ml_model_metrics.csv")
print("logistic_regression_coefficients.csv")
print("random_forest_feature_importance.csv")
print("xgboost_feature_importance.csv")
print("Figure_11_ML_feature_importance.png")
