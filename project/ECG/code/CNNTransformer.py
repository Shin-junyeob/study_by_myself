import os
import pickle
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_heads, num_layers, dropout):
        super(CNNTransformerModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.linear_input = nn.Linear(input_dim, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))  # Add channel dimension for Conv1D
        x = x.permute(0, 2, 1)  # Reshape for transformer
        x = self.transformer(x, x) 
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_dataset(metadata_path, data_path, Hz):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name, Hz):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in tqdm(subset.iterrows(), total=subset.shape[0], desc=f"Loading {set_name} signals"):
            if Hz == 100:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1]}.dat")
            else:
                file_path = os.path.join(data_path, f'{Hz}Hz', set_name, f"{row['file_name'].split('/')[1].split('_')[0]}_hr.dat")
            try:
                signal = wfdb.rdsamp(file_path.replace('.dat', ''), sampfrom=0, sampto=Hz * 10)[0]
                signals.append(signal.flatten())
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals, dtype=np.float32), labels

    X_train, y_train = load_signals('train', Hz)
    X_val, y_val = load_signals('validation', Hz)
    X_test, y_test = load_signals('test', Hz)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(X, y, label_map, fit_scaler=False, scaler=None, encoder=None):
    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()

    y_mapped = [label_map.get(label, "OTHERS") for label in y]
    y = encoder.fit_transform(y_mapped) if fit_scaler else encoder.transform(y_mapped)

    return X, y, scaler, encoder

def smote_augment(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Original data shape: {X.shape, y.shape}")
    print(f"Resampled data shape: {X_res.shape, y_res.shape}")
    return X_res, y_res

def train_and_evaluate_model(
    model, train_loader, val_loader, test_loader, criterion, optimizer, device, output_path, stage_name, encoder, epochs=50, patience=5
):
    model.to(device)
    best_accuracy = 0.0
    patience_counter = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, axis=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Validation Accuracy: {accuracy:.4f}, Learning Rate: {current_lr}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\n--- Testing the Model ---")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, axis=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    class_labels = ['NORM', 'MI', 'OTHERS']
    report = classification_report(all_labels, all_preds, target_names=class_labels, output_dict=True)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Report:\n{classification_report(all_labels, all_preds, target_names=class_labels)}")

    os.makedirs(output_path, exist_ok=True)
    results = {
        "validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": report
    }
    with open(os.path.join(output_path, f"{stage_name}_results.pkl"), "wb") as f:
        pickle.dump(results, f)

def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data'
    Hz = 100
    output_path = f'../results/CNNTransformer/{Hz}Hz'

    params = {
        100: {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.105, 'lr': 7.54e-5, 'batch_size': 32},
        500: {'hidden_dim': 256, 'num_heads': 8, 'num_layers': 3, 'dropout': 0.102, 'lr': 1.46e-4, 'batch_size': 32}
    }
    hyperparams = params[Hz]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(metadata_path, data_path, Hz)
    label_map = {"NORM": 0, "MI": 1, "OTHERS": 2}
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, label_map, fit_scaler=True)
    X_val, y_val, _, _ = preprocess_data(X_val, y_val, label_map, scaler=scaler, encoder=encoder)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, label_map, scaler=scaler, encoder=encoder)

    X_train, y_train = smote_augment(X_train, y_train)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformerModel(
        input_dim=X_train.shape[1],
        num_classes=len(encoder.classes_),
        hidden_dim=hyperparams['hidden_dim'],
        num_heads=hyperparams['num_heads'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

    train_and_evaluate_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, output_path, "CNNTransformer", encoder)

if __name__ == "__main__":
    main()