import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

output_dir = "activation_reports"
os.makedirs(output_dir, exist_ok=True)

mapping = {
    '0000': 1.5,
    '1000': 2.73,
    '0001': 39.0843,
    '1001': 43.34,
    '0010': 3.62853,
    '1010': 5.57,
    '0011': 60.793,
    '1011': 103,
    '0100': 3.05127,
    '1100': 4.79,
    '0101': 53.166,
    '1101': 68.97,
    '0110': 6.785,
    '1110': 9.8386,
    '0111': 115,
    '1111': 136
}

file_paths = ['1_trajectory.xlsx', '2_trajectory.xlsx', '3_trajectory.xlsx',
              '4_trajectory.xlsx', '5_trajectory.xlsx', '6_trajectory.xlsx',
              '7_trajectory.xlsx', '8_trajectory.xlsx']
data = []
labels = []

data_dir = "trajectory_data"
for i, file in enumerate(file_paths):
    full_path = os.path.join(data_dir, file)
    df = pd.read_excel(full_path, header=None, dtype=str)
    data.append(df.values)
    labels += [i] * len(df.columns)

data = np.concatenate(data, axis=1).T
labels = np.array(labels)

X_mapped = []
for row in data:
    sample = []
    for val in row.flatten():
        sample.append(mapping[val] if not pd.isna(val) else 0)
    X_mapped.append(sample)
X_mapped = np.array(X_mapped, dtype=float)

num_classes = len(file_paths)
y_encoded = np.eye(num_classes)[labels]

input_size = X_mapped.shape[1]
output_size = num_classes
learning_rate = 0.0003
epochs = 50

W = np.random.randn(input_size, output_size) * 0.01
b = np.zeros((1, output_size))

beta1, beta2, epsilon = 0.9, 0.999, 1e-8
mW, vW = np.zeros_like(W), np.zeros_like(W)
mb, vb = np.zeros_like(b), np.zeros_like(b)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

activation_records = {
    i: {
        'class_name': file_paths[i],
        'all_labels': np.zeros((epochs, num_classes)),
        'samples': []
    } for i in range(num_classes)
}

for class_idx in range(num_classes):
    class_samples = np.where(labels == class_idx)[0]
    for sample_idx in class_samples:
        activation_records[class_idx]['samples'].append({
            'all_labels': np.zeros((epochs, num_classes))
        })

training_history = {
    'epoch': [],
    'loss': [],
    'accuracy': [],
    'timestamp': []
}

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    z = np.dot(X_mapped, W) + b
    a = softmax(z)
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        class_activations = a[class_indices]
        activation_records[class_idx]['all_labels'][epoch] = np.mean(class_activations, axis=0)
        for i, sample_idx in enumerate(class_indices):
            activation_records[class_idx]['samples'][i]['all_labels'][epoch] = a[sample_idx]
    train_loss = cross_entropy(y_encoded, a)
    train_accuracy = np.mean(np.argmax(a, axis=1) == labels)
    training_history['epoch'].append(epoch + 1)
    training_history['loss'].append(train_loss)
    training_history['accuracy'].append(train_accuracy)
    training_history['timestamp'].append(time.strftime("%Y-%m-%d %H:%M:%S"))
    dz = cross_entropy_derivative(y_encoded, a)
    dW = np.dot(X_mapped.T, dz)
    db = np.sum(dz, axis=0, keepdims=True)
    for param, dparam, m, v in zip([W, b], [dW, db], [mW, mb], [vW, vb]):
        m[:] = beta1 * m + (1 - beta1) * dparam
        v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    epoch_time = time.time() - epoch_start
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
        print(f'Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f} | Time: {epoch_time:.2f}s')

def create_activation_reports():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    for class_idx in range(num_classes):
        class_name = activation_records[class_idx]['class_name']
        base_name = class_name.replace('.xlsx', '')
        report_filename = os.path.join(report_dir, f"{base_name}_activation_report.xlsx")
        trend_df = pd.DataFrame(
            activation_records[class_idx]['all_labels'],
            columns=file_paths
        )
        trend_df.insert(0, 'Epoch', range(1, epochs + 1))
        final_activations = activation_records[class_idx]['all_labels'][-1]
        max_activations = np.max(activation_records[class_idx]['all_labels'], axis=0)
        min_activations = np.min(activation_records[class_idx]['all_labels'], axis=0)
        summary_df = pd.DataFrame({
            'Class': file_paths,
            'Final Activation': final_activations,
            'Max Activation': max_activations,
            'Min Activation': min_activations,
            'Activation Range': max_activations - min_activations
        })
        with pd.ExcelWriter(report_filename) as writer:
            trend_df.to_excel(writer, sheet_name='Activation Trend', index=False)
            summary_df.to_excel(writer, sheet_name='Statistical Summary', index=False)
        plt.figure(figsize=(12, 8))
        for col_idx, col_name in enumerate(trend_df.columns[1:]):
            plt.plot(trend_df['Epoch'], trend_df[col_name],
                     label=f"{col_name}", linewidth=2.5, alpha=0.8)
        current_class_idx = class_idx
        plt.plot(trend_df['Epoch'], trend_df[file_paths[current_class_idx]],
                 label=f"{file_paths[current_class_idx]} (Current Class)",
                 linewidth=3.5, color='red', linestyle='-')
        plt.title(f'Activation Trend - {base_name}')
        plt.xlabel('Training Epoch')
        plt.ylabel('Activation Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=9, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f"{base_name}_activation_trend.png"),
                   dpi=200, bbox_inches='tight')
        plt.close()
    return report_dir

final_report_dir = create_activation_reports()
print(f"Activation reports and trend charts saved to folder: {final_report_dir}")
print(f"Output contents:")
print(f"├── 1_trajectory_activation_report.xlsx")
print(f"├── 1_trajectory_activation_trend.png")
print(f"├── 2_trajectory_activation_report.xlsx")
print(f"├── 2_trajectory_activation_trend.png")
print(f"├── ...")
print(f"└── 8_trajectory_activation_trend.png")
