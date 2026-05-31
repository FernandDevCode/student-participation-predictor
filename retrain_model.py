"""
retrain_model.py
Rebuilds best_random_forest_model.joblib and preprocessor.joblib
using the current scikit-learn version (1.8.x) so the Streamlit
app can load them without version-mismatch errors.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "RIT_Opportunity_Wise_Data_weeek2_cleaned.csv"
MODEL_OUT  = BASE_DIR / "model" / "best_random_forest_model.joblib"
PREP_OUT   = BASE_DIR / "model" / "preprocessor.joblib"

# ─── Features (same as original app) ─────────────────────────────────────────
NUMERICAL   = ["Student_Age", "Opportunity_Duration", "Opportunity_Quarter",
               "Apply_to_Start_Days", "Signup_to_Start_Days"]
CATEGORICAL = ["Opportunity Name", "Opportunity Category", "Gender",
               "Country", "Major_Cleaned", "Sectors_of_Intended_Major"]
TARGET      = "Participation_status"

FEATURES = NUMERICAL + CATEGORICAL

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)

# Drop rows with missing target or features
df = df.dropna(subset=[TARGET] + FEATURES)

X = df[FEATURES]
y = df[TARGET]

print(f"  Dataset shape : {X.shape}")
print(f"  Target balance:\n{y.value_counts()}")

# ─── Preprocessor ─────────────────────────────────────────────────────────────
print("\nBuilding preprocessor …")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ]
)

# ─── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Fit preprocessor ────────────────────────────────────────────────────────
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

# ─── Train Random Forest ──────────────────────────────────────────────────────
print("Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train_prep, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_prep)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.4f}")
print(classification_report(y_test, y_pred))

# ─── Save ─────────────────────────────────────────────────────────────────────
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model,        MODEL_OUT)
joblib.dump(preprocessor, PREP_OUT)
print(f"\n✅  Model saved      → {MODEL_OUT}")
print(f"✅  Preprocessor saved → {PREP_OUT}")
