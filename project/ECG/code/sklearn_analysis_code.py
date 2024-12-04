import os
import pandas as pd
import numpy as np
import wfdb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def load_dataset(metadata_path, data_path):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in subset.iterrows():
            file_path = os.path.join(data_path, set_name, f"{row['file_name'].split('/')[1]}.dat")
            try:
                signal, fields = wfdb.rdsamp(file_path.replace('.dat', ''))
                # signals.append(signal[:, 0]) # 첫번째 채널만 사용
                signals.append(signal.flatten()) # lead-I만 추출했기 떄문에 전체 신호 로드 (채널 선택 불필요)
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals), labels

    X_train, y_train = load_signals('train')
    X_val, y_val = load_signals('validation')
    X_test, y_test = load_signals('test')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess_data(X, y, fit_scaler=False, scaler=None, encoder=None):
    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()

    if fit_scaler:
        y = encoder.fit_transform(y)
    else:
        y = encoder.transform(y)

    return X, y, scaler, encoder


def train_and_evaluate_sklearn_model(model, X_train, y_train, X_test, y_test, encoder, output_path):
    print(f'Operating about {model.__class__.__name__} ...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)

    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    # Save results
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{model.__class__.__name__}_results.txt"), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(report)

    return accuracy, report


def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data/100Hz'
    output_path = '../results/sklearn'

    # Load and preprocess data
    (X_train, y_train), (_, _), (X_test, y_test) = load_dataset(metadata_path, data_path)
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, fit_scaler=True)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, scaler=scaler, encoder=encoder)

    # Models
    models = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=100),
        LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1),
        SVC(probability=True)
    ]

    # Train and evaluate models
    results = []
    for model in models:
        accuracy, report = train_and_evaluate_sklearn_model(model, X_train, y_train, X_test, y_test, encoder, output_path)
        results.append({"Model": model.__class__.__name__, "Accuracy": accuracy})

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, "results_summary.csv"), index=False)
    print("Summary saved.")

    print("sklearn analysis completed")

if __name__ == "__main__":
    main()