# src/visualize_results.py
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Load model and test data
model_data = joblib.load("models/rf_at_risk.joblib")
model = model_data["model"]
features = model_data["features"]
X_train, X_test = model_data["X_train"], model_data["X_test"]
y_train, y_test = model_data["y_train"], model_data["y_test"]

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print("Train Accuracy:", acc_train)
print("Test Accuracy:", acc_test)

report_train = classification_report(y_train, y_pred_train, output_dict=True)
report_test = classification_report(y_test, y_pred_test, output_dict=True)

# --- CONFUSION MATRIX FOR TEST SET ---
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Test Set")
plt.show()

# --- TRAIN VS TEST PERFORMANCE CHART ---
metrics = ["precision", "recall", "f1-score"]
classes = ["0", "1"]

train_scores = {m: [report_train[c][m] for c in classes] for m in metrics}
test_scores = {m: [report_test[c][m] for c in classes] for m in metrics}

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    df = pd.DataFrame({
        "Train": train_scores[metric],
        "Test": test_scores[metric]
    }, index=classes)
    df.plot(kind="bar", ax=axs[i])
    axs[i].set_title(f"{metric.capitalize()} Comparison")
    axs[i].set_ylim(0, 1)
    axs[i].grid(True)

plt.suptitle("Train vs Test Performance by Class")
plt.tight_layout()
plt.show()
