import os
import pandas as pd
import numpy as np
import wfdb
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

    max_len = max(len(signal) for signal in signals)
    signals_padded = np.array([np.pad(signal, (0, max_len - len(signal))) for signal in signals])

    return signals_padded, labels

print("Loading data...")
X_train, y_train = load_data("train", metadata, data_path)
X_val, y_val = load_data("validation", metadata, data_path)
X_test, y_test = load_data("test", metadata, data_path)
print("Data loading completed.")

mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        focal_loss_value = alpha * (1 - pt)**gamma * bce
        return tf.reduce_mean(focal_loss_value)
    return loss
    
def create_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=focal_loss(), metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], 1)
num_classes = y_train_bin.shape[1]
model = create_cnn_lstm_model(input_shape, num_classes)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=np.argmax(y_train_bin, axis=1)
)
class_weights_dict = {i: weight for i,weight in enumerate(class_weight)}

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Training CNN+LSTM model...")
history = model.fit(
    X_train, y_train_bin,
    validation_data=(X_val, y_val_bin),
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict
    # callbacks=[early_stopping]
)

print("Evaluating on validation set...")
val_loss, val_accuracy = model.evaluate(X_val, y_val_bin)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_bin)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report (Test Set):")
threshold = 0.3
y_test_pred = model.predict(X_test) > threshold
print(classification_report(y_test_bin, y_test_pred, target_names=mlb.classes_))

output_path = '../results'
os.makedirs(output_path, exist_ok=True)
results_file = os.path.join(output_path, 'CNN_model_results.xlsx')

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
confusion_matrix_image_path = os.path.join(output_path, 'CNN_confusion_matrix.png')
plt.savefig(confusion_matrix_image_path, dpi = 300, bbox_inches = 'tight')
plt.close()

correlation_matrix = np.corrcoef(y_test_pred.T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
plt.title("Correlation Matrix of Predicted Labels")
correlation_matrix_image_path = os.path.join(output_path, 'CNN_correlation_matrix.png')
plt.savefig(correlation_matrix_image_path, dpi=300, bbox_inches='tight')
plt.close()

def save_results_to_file(output_path, accuracy, classification_report_df):
    os.makedirs(output_path, exist_ok=True)

    results_file = os.path.join(output_path, "CNN_model_results.xlsx")
    with open(results_file, 'w', encoding="utf-8") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report_df.to_string(index=True))

    print(f"Results saved to {results_file}")

save_results_to_file(output_path, accuracy, classification_df)