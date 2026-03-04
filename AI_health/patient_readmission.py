import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE


# -------------------- LOAD DATASET ---------------------------
df = pd.read_csv(r"C:\Users\akono\OneDrive\Desktop\AI_health\patient_readmission_data.csv")

print("\nColumns in dataset:")
print(df.columns.tolist())


# -------------------- TARGET COLUMN --------------------------
target_col = "Readmitted30Days"

if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found!")

print(f"\nUsing target column: {target_col}")


# -------------------- CONVERT TARGET TO NUMERIC ----------------
# Convert Yes/No to 1/0
df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

# Check conversion worked
if df[target_col].isnull().any():
    raise ValueError("❌ Target column contains unexpected values. Check unique values!")

print("Target mapping: Yes → 1, No → 0")


# -------------------- HANDLE MISSING VALUES ------------------

# Numeric columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Categorical columns (explicit str avoids warning)
cat_cols = df.select_dtypes(include=["object", "string"]).columns

# Fill numeric NaNs with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical NaNs with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# -------------------- FEATURE / TARGET SPLIT ----------------
drop_cols = ["PatientID"] if "PatientID" in df.columns else []

X = df.drop(columns=drop_cols + [target_col])
y = df[target_col]


# -------------------- ENCODE CATEGORICAL DATA ----------------
X = pd.get_dummies(X, drop_first=True)


# -------------------- TRAIN-TEST SPLIT -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------- HANDLE CLASS IMBALANCE -----------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


# -------------------- TRAIN MODEL ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_sm, y_train_sm)


# -------------------- PREDICT PROBABILITIES ------------------
y_prob = rf_model.predict_proba(X_test)[:, 1]


# -------------------- ROC–AUC CALCULATION --------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)


# -------------------- PLOT ROC CURVE -------------------------
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Patient Readmission Prediction")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------- PRINT RESULT ---------------------------
print("====================================")
print(f"ROC–AUC Score: {auc_score:.3f}")
print("====================================")
