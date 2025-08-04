import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def learning_curve_analysis(data_file, system_name, n_nodes):
    """
    Analyze learning curve for different sample sizes.
    
    Parameters:
    - data_file: Path to CSV file (e.g., 'harmonic_data_5th_118bus.csv')
    - system_name: System identifier (e.g., '118bus')
    - n_nodes: Number of nodes (e.g., 9, 39, 118)
    """
    # 1. Read data
    data = pd.read_csv(data_file)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # 2. Split train/test
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    # 3. Feature preprocessing
    n_branches = features.shape[1] // 3
    V_diff_mag_train = X_train[:, :n_branches]
    V_diff_mag_test = X_test[:, :n_branches]
    V_diff_mag_min = V_diff_mag_train.min(axis=0)
    V_diff_mag_max = V_diff_mag_train.max(axis=0)
    V_diff_mag_train = (V_diff_mag_train - V_diff_mag_min) / (V_diff_mag_max - V_diff_mag_min + 1e-10)
    V_diff_mag_test = (V_diff_mag_test - V_diff_mag_min) / (V_diff_mag_max - V_diff_mag_min + 1e-10)
    X_train = np.hstack([V_diff_mag_train, X_train[:, n_branches:2*n_branches], X_train[:, 2*n_branches:]])
    X_test = np.hstack([V_diff_mag_test, X_test[:, n_branches:2*n_branches], X_test[:, 2*n_branches:]])

    # 4. Learning curve
    sample_sizes = [50, 100, 200]
    acc_results = []
    for s in sample_sizes:
        n_samples = s * n_nodes
        indices = np.random.choice(len(X_train), min(n_samples, len(X_train)), replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42)
        rf_model.fit(X_subset, y_subset)
        y_pred = rf_model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        acc_results.append(acc)
        print(f'Sample size {n_samples} (per node {s}): Test accuracy {acc*100:.2f}%')
    
    # Save results
    results_df = pd.DataFrame({
        'Samples_Per_Node': sample_sizes,
        'Test_Accuracy': [acc * 100 for acc in acc_results]
    })
    results_df.to_csv(f'learning_curve_{system_name}.csv', index=False)
    print(f'Learning curve saved to learning_curve_{system_name}.csv')

if __name__ == "__main__":
    learning_curve_analysis('harmonic_data_5th_118bus.csv', '118bus', 118)
