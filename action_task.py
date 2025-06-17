import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


class Config:
    TASK_MODE = 'action'
    DATA_DIR = './action_data/'
    RESULT_DIR = './action_results/'
    ACTION_FILES = ['1_forehand_heavy.xlsx', '2_forehand_light.xlsx', '3_backhand_heavy.xlsx', '4_backhand_light.xlsx']
    ACTION_MAPPING = {
        '0000': 1.5, '1000': 2.73, '0001': 39.0843, '1001': 43.34,
        '0010': 3.62853, '1010': 5.57, '0011': 60.793, '1011': 103,
        '0100': 3.05127, '1100': 4.79, '0101': 53.166, '1101': 68.97,
        '0110': 6.785, '1110': 9.8386, '0111': 115, '1111': 136
    }
    LEARNING_RATE = 0.001
    EPOCHS = 100
    TEST_SIZE = 0.5
    RANDOM_STATE = 42


def load_and_preprocess_data():
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)
        raise FileNotFoundError(f"Data directory {Config.DATA_DIR} not found, created empty directory")

    if not os.path.exists(Config.RESULT_DIR):
        os.makedirs(Config.RESULT_DIR)

    data, labels = [], []
    for i, file in enumerate(Config.ACTION_FILES):
        file_path = os.path.join(Config.DATA_DIR, file)
        try:
            df = pd.read_excel(file_path, header=None, dtype=str)
            data.append(df.values)
            labels += [i] * df.shape[1]
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found")

    if not data:
        raise ValueError("No valid data files found")

    data = np.concatenate(data, axis=1).T
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=labels
    )

    def apply_mapping(X, mapping_dict):
        mapped = []
        for row in X:
            sample = [mapping_dict.get(val, 0) if not pd.isna(val) else 0
                      for val in row.flatten()]
            mapped.append(sample)
        return np.array(mapped, dtype=float)

    class_names = ['forehand_heavy', 'forehand_light', 'backhand_heavy', 'backhand_light']

    return (
        apply_mapping(X_train, Config.ACTION_MAPPING),
        apply_mapping(X_test, Config.ACTION_MAPPING),
        y_train,
        y_test,
        class_names
    )


class NeuralNetwork:
    def __init__(self, input_size, num_classes):
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def train(self, X, y, X_test, y_test, epochs):
        y_encoded = np.eye(self.W.shape[1])[y]
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            z = np.dot(X, self.W) + self.b
            a = self.softmax(z)
            train_loss = self.cross_entropy(y_encoded, a)
            train_acc = np.mean(np.argmax(a, axis=1) == y)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            test_loss, test_acc = self.evaluate(X_test, y_test)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            dz = a - y_encoded
            dW = np.dot(X.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            self._adam_update(dW, db, epoch)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}')
        return history

    def _adam_update(self, dW, db, epoch):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for param, dparam, m, v in zip(
                [self.W, self.b], [dW, db],
                [self.mW, self.mb], [self.vW, self.vb]
        ):
            m[:] = beta1 * m + (1 - beta1) * dparam
            v[:] = beta2 * v + (1 - beta2) * (dparam ** 2)
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))
            param -= Config.LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)

    def evaluate(self, X, y):
        z = np.dot(X, self.W) + self.b
        a = self.softmax(z)
        loss = self.cross_entropy(np.eye(self.W.shape[1])[y], a)
        acc = np.mean(np.argmax(a, axis=1) == y)
        return loss, acc

    def predict(self, X):
        return np.argmax(np.dot(X, self.W) + self.b, axis=1)


def plot_and_save_figures(history, cm, class_names):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULT_DIR, 'training_curves.png'))
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Action Recognition Confusion Matrix')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(Config.RESULT_DIR, 'confusion_matrix.png'))


def main():
    print("=" * 50)
    print("Action Recognition System Started")
    print(f"Data Directory: {Config.DATA_DIR}")
    print(f"Results Directory: {Config.RESULT_DIR}")
    print("=" * 50)

    try:
        X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
        print(f"\nData loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"Recognized classes: {', '.join(class_names)}")
        model = NeuralNetwork(X_train.shape[1], len(class_names))
        print(f"Model initialized: Input dimension={X_train.shape[1]}, Output classes={len(class_names)}")
        print(f"\nTraining started (Epochs={Config.EPOCHS}, Learning Rate={Config.LEARNING_RATE})...")
        history = model.train(X_train, y_train, X_test, y_test, Config.EPOCHS)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"\nTraining completed! Final test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_and_save_figures(history, cm, class_names)
        weight_file = os.path.join(Config.RESULT_DIR, 'model_weights.csv')
        pd.DataFrame(model.W).to_csv(weight_file)
        print(f"\nModel weights saved to: {weight_file}")
        result_df = pd.DataFrame({
            'Metric': ['Test Accuracy', 'Test Loss'],
            'Value': [test_acc, test_loss]
        })
        result_df.to_csv(os.path.join(Config.RESULT_DIR, 'evaluation_results.csv'), index=False)
        print(f"All results saved to {Config.RESULT_DIR}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Program terminated")


if __name__ == "__main__":
    main()
