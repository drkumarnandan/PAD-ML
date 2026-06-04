import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ============================================================
# Paths
# ============================================================

REPO_DIR = "alpha-gliadin-dimerization-ML"
DATA_FILE = os.path.join(REPO_DIR, "data", "updated_PCA_scores.csv")
MODEL_DIR = os.path.join(REPO_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_FILE)

df["target"] = (df["state"] == "Compact-associated").astype(int)

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

# ============================================================
# Models
# ============================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

random_forest_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

xgboost_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# ============================================================
# Train on full descriptor table
# ============================================================

print("Training Logistic Regression...")
logistic_model.fit(X, y)

print("Training Random Forest...")
random_forest_model.fit(X, y)

print("Training XGBoost...")
xgboost_model.fit(X, y)

# ============================================================
# Save models
# ============================================================

joblib.dump(
    logistic_model,
    os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
)

joblib.dump(
    random_forest_model,
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

joblib.dump(
    xgboost_model,
    os.path.join(MODEL_DIR, "xgboost_model.pkl")
)

# Save feature list too
joblib.dump(
    features,
    os.path.join(MODEL_DIR, "feature_list.pkl")
)

print("\nSaved models:")
print(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
print(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
print(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
print(os.path.join(MODEL_DIR, "feature_list.pkl"))

print("\nDone.")
