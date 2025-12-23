import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


FEATURES_CSV = "features/features.csv"


print("Loading features...")
df = pd.read_csv(FEATURES_CSV)

print("Original rows:", len(df))


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Rows after cleaning:", len(df))


X = df.drop("scanner", axis=1)
y = df["scanner"]


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)


print("Training SVM...")
svm = SVC(
    kernel="rbf",
    C=60,
    gamma="scale"
)

svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)


print("\n================ RESULTS ================\n")

print(f"Random Forest Accuracy : {rf_acc:.4f}")
print(f"SVM Accuracy           : {svm_acc:.4f}")

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))
