import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime

def random_forest_train(data_file, system_name, n_nodes):
    """
    Train Random Forest for harmonic source localization and save metrics.
    
    Parameters:
    - data_file: Path to CSV file (e.g., 'harmonic_data_5th_118bus.csv')
    - system_name: System identifier (e.g., '118bus')
    - n_nodes: Number of nodes (e.g., 9, 39, 118)
    """
    # 1. Read data
    data = pd.read_csv(data_file)
    features = data.iloc[:, :-1].values  # All columns except last
    labels = data.iloc[:, -1].values    # Last column

    # 2. Split train/test
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    # 3. Feature preprocessing: Normalize magnitude features
    n_branches = features.shape[1] // 3
    V_diff_mag_train = X_train[:, :n_branches]
    V_diff_mag_test = X_test[:, :n_branches]
    V_diff_mag_min = V_diff_mag_train.min(axis=0)
    V_diff_mag_max = V_diff_mag_train.max(axis=0)
    V_diff_mag_train = (V_diff_mag_train - V_diff_mag_min) / (V_diff_mag_max - V_diff_mag_min + 1e-10)
    V_diff_mag_test = (V_diff_mag_test - V_diff_mag_min) / (V_diff_mag_max - V_diff_mag_min + 1e-10)
    X_train = np.hstack([V_diff_mag_train, X_train[:, n_branches:2*n_branches], X_train[:, 2*n_branches:]])
    X_test = np.hstack([V_diff_mag_test, X_test[:, n_branches:2*n_branches], X_test[:, 2*n_branches:]])

    # 4. Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # 5. Feature importance
    feature_importance = rf_model.feature_importances_
    feature_names = data.columns[:-1]
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    short_names = []
    for name in sorted_names:
        if 'mag' in name:
            short_name = name.replace('V_diff_mag_', 'Mag ').replace('_', '-')
        elif 'sine' in name:
            short_name = name.replace('V_diff_sine_', 'Sin ').replace('_', '-')
        else:
            short_name = name.replace('V_diff_cosine_', 'Cos ').replace('_', '-')
        short_names.append(short_name)
    
    n_display = min(20, len(sorted_importance))
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_display), sorted_importance[:n_display])
    plt.xticks(range(n_display), short_names[:n_display], rotation=60)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance ({system_name})')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'feature_importance_{system_name}_{timestamp}.png', bbox_inches='tight')
    plt.close()
    print(f'Feature importance saved to feature_importance_{system_name}_{timestamp}.png')

    # 6. Metrics
    y_pred = rf_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    n_classes = conf_matrix.shape[0]
    metrics = {
        'Node': list(range(1, n_classes + 1)),
        'Precision': [report[str(i)]['precision'] * 100 for i in range(1, n_classes + 1)],
        'Recall': [report[str(i)]['recall'] * 100 for i in range(1, n_classes + 1)],
        'F1_Score': [report[str(i)]['f1-score'] * 100 for i in range(1, n_classes + 1)]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'metrics_{system_name}.csv', index=False)
    print(f'Metrics saved to metrics_{system_name}.csv')
    print(f'Average Precision ({system_name}): {metrics_df["Precision"].mean():.2f}%')
    print(f'Average Recall ({system_name}): {metrics_df["Recall"].mean():.2f}%')
    print(f'Average F1-Score ({system_name}): {metrics_df["F1_Score"].mean():.2f}%')

if __name__ == "__main__":
    random_forest_train('harmonic_data_5th_118bus.csv', '118bus', 118)
