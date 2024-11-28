import os
import pandas as pd
import numpy as np
import wfdb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

data_path = "../data/processed_data/100Hz"
metadata_path = "../data/metadata.csv"

metadata = pd.read_csv(metadata_path)
metadata['labels'] = metadata['labels'].apply(lambda x: x.split(","))

def load_data(set_name, metadata, data_path):
    subset = metadata[metadata['set'] == set_name]
    signals = []
    labels = []

    for _, row in subset.iterrows():
        base_name = os.path.basename(row['file_path_lr'])
        dat_path = os.path.join(data_path, set_name, base_name + ".dat")
        hea_path = os.path.join(data_path, set_name, base_name + ".hea")

        if not os.path.exists(dat_path):
            print(f"Missing .dat file: {dat_path}")
            continue
        if not os.path.exists(hea_path):
            print(f"Missing .hea file: {hea_path}")
            continue

        try:
            record = wfdb.rdsamp(dat_path.replace('.dat', ''))
            signal = record[0].flatten()
            signal = (signal - np.mean(signal)) / np.std(signal)
            signals.append(signal)
            labels.append(row['labels'])
        except Exception as e:
            print(f"Error loading file {dat_path}: {e}")

    return np.array(signals), labels

print("Loading data...")
X_train, y_train = load_data("train", metadata, data_path)
X_val, y_val = load_data("validation", metadata, data_path)
X_test, y_test = load_data("test", metadata, data_path)
print("Data loading completed.")

mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)

class_weights_list = []
for i in range(y_train_bin.shape[1]):
    class_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train_bin[:, i]
    )
    class_weights_list.append({0: class_weight[0], 1: class_weight[1]})

print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weights_list
)
rf.fit(X_train, y_train_bin)

print("Evaluating on validation set...")
y_val_pred = rf.predict(X_val)
val_accuracy = accuracy_score(y_val_bin, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Evaluating on test set...")
y_test_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test_bin, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report (Test Set):")
print(classification_report(y_test_bin, y_test_pred, target_names=mlb.classes_))

output_path = '../results'
os.makedirs(output_path, exist_ok=True)
results_file = os.path.join(output_path, 'RF_model_results.xlsx')

accuracy = test_accuracy
classification_rep = classification_report(y_test_bin, y_test_pred, target_names = mlb.classes_, output_dict = True)
classification_df = pd.DataFrame(classification_rep).transpose()

conf_matrix = confusion_matrix(
    y_test_bin.argmax(axis=1),
    y_test_pred.argmax(axis=1)
)
conf_matrix_df = pd.DataFrame(conf_matrix, index = mlb.classes_, columns = mlb.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_df, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = mlb.classes_, yticklabels = mlb.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
confusion_matrix_image_path = os.path.join(output_path, 'RF_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path, dpi = 300, bbox_inches = 'tight')
plt.close()

correlation_matrix = np.corrcoef(y_test_pred.T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
plt.title("Correlation Matrix of Predicted Labels")
correlation_matrix_image_path = os.path.join(output_path, 'RF_correlation_matrix.png')
plt.savefig(correlation_matrix_image_path, dpi = 300, bbox_inches = 'tight')
plt.close()

def save_results_to_file(output_path, accuracy, classification_report_df):
    os.makedirs(output_path, exist_ok=True)
    
    results_file = os.path.join(output_path, "RF_model_results.xlsx")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report_df.to_string(index=True))

    print(f"Results saved to {results_file}")

save_results_to_file(output_path, accuracy, classification_df)