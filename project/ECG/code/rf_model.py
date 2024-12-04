import os
import pandas as pd
import numpy as np
import wfdb
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_dataset(metadata_path, data_path):
    metadata = pd.read_csv(metadata_path)

    def load_signals(set_name):
        subset = metadata[metadata['set'] == set_name]
        signals, labels = [], []
        for _, row in subset.iterrows():
            file_path = os.path.join(data_path, set_name, f"{row['file_name'].split('/')[1]}.dat")
            try:
                signal, fields = wfdb.rdsamp(file_path.replace('.dat', ''))
                signals.append(signal.flatten())  # Lead-I 데이터 로드
                labels.append(row['sub_label'])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(signals), labels

    X_train, y_train = load_signals('train')
    X_val, y_val = load_signals('validation')
    X_test, y_test = load_signals('test')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(X, y, fit_scaler=False, scaler=None, encoder=None):
    if len(X.shape) == 1:  # 1D 데이터를 2D로 변환
        X = X.reshape(-1, 1)

    if scaler is None:
        scaler = StandardScaler()
    X = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if encoder is None:
        encoder = LabelEncoder()
    y = encoder.fit_transform(y) if fit_scaler else encoder.transform(y)

    return X, y, scaler, encoder

def save_results_to_pickle(report, accuracy, best_params, best_estimator, file_path):
    results = {
        "classification_report": report,
        "accuracy_score": accuracy,
        "best_params": best_params,
        "best_estimator": best_estimator
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 디렉터리 생성
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")

def main():
    metadata_path = '../data/metadata.csv'
    data_path = '../data/processed_data/100Hz'
    output_path = '../results/sklearn'

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(metadata_path, data_path)

    # 데이터 전처리
    X_train, y_train, scaler, encoder = preprocess_data(X_train, y_train, fit_scaler=True)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, fit_scaler=False, scaler=scaler, encoder=encoder)

    # GridSearchCV 설정
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    CV_rf.fit(X_train, y_train)

    # 결과 저장
    y_pred = CV_rf.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    best_params = CV_rf.best_params_
    best_estimator = CV_rf.best_estimator_

    output_pickle_path = '../results/RF_model.pkl'
    save_results_to_pickle(report, accuracy, best_params, best_estimator, output_pickle_path)

if __name__ == "__main__":
    main()