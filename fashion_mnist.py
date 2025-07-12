# -*- coding: utf-8 -*-
# @Time    : 2024/10/31 20:11
# @Author  : stp
# @File    : main.py

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        expected_size = num_images * num_rows * num_cols

        if image_data.size != expected_size:
            raise ValueError(f"Expected image data size {expected_size}, but got {image_data.size}")

        image_data = image_data.reshape(num_images, num_rows, num_cols)
        return image_data, num_images

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    file=r".\test_images\mapped_images.idx3-ubyte"
    #mapped_images, _ = load_idx3_ubyte(file)
    noisy_images, _ = load_idx3_ubyte("./test_images/mapped_images.idx3-ubyte")

    #new_test_labels_mapped = test_labels[:mapped_images.shape[0]]
    new_test_labels_noisy = test_labels[:noisy_images.shape[0]]

    classes = np.random.choice(np.arange(10), 7, replace=False)
    print(f"Selected classes: {classes}")

    train_mask = np.isin(train_labels, classes)
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]

    #test_mask = np.isin(new_test_labels_mapped, classes)
    #test_images = mapped_images[test_mask]
    #test_labels = new_test_labels_mapped[test_mask]
    test_mask = np.isin(new_test_labels_noisy, classes)
    test_images = noisy_images[test_mask]
    test_labels = new_test_labels_noisy[test_mask]

    # 确保数据长度一致
    print(f"Train data shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

    label_map = {old_label: new_label for new_label, old_label in enumerate(classes)}
    train_labels = np.array([label_map[label] for label in train_labels])
    test_labels = np.array([label_map[label] for label in test_labels])

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    return train_images, train_labels, test_images, test_labels, classes

def flatten_and_split(images):
    split_images = []
    for img in images:
        flattened = img.flatten()  #
        split = np.split(flattened, 196)
        split_images.append(split)
    return np.array(split_images)

# 映射关系
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

def apply_mapping(split_images):
    mapped_images = []
    for image in split_images:
        mapped_image = []
        for vector in image:
            vector_key = ''.join(map(lambda x: '1' if x > 0.166 else '0', vector))
            mapped_image.append(mapping.get(vector_key, 0))
        mapped_images.append(mapped_image)
    return np.array(mapped_images)

def create_model(input_shape, learning_rate, l2_lambda):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(7, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    ])

    def custom_loss(y_true, y_pred):
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_variables if 'kernel' in v.name])
        return base_loss + l2_lambda * reg_loss

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=custom_loss,
                  metrics=['accuracy'])
    return model


def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_model(model, train_images, train_labels, test_images, test_labels, epochs, batch_size):
    best_accuracy = 0
    best_weights = None
    best_epoch = 0

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=1)

    val_accuracy = history.history['val_accuracy']
    epoch_list = list(range(1, len(val_accuracy) + 1))
    df = pd.DataFrame({'Epoch': epoch_list, 'Val_Accuracy': val_accuracy})

    df.to_excel('val_accuracy.xlsx', index=False)
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_accuracy = max(history.history['val_accuracy'])
    best_weights = model.get_weights()
    return best_epoch, best_accuracy, best_weights


def export_to_excel(initial_weights, best_weights, best_accuracy, selected_classes, filename='fiv1.xlsx'):

    max_len = max([len(w.flatten()) for w in initial_weights + best_weights])

    def flatten_and_pad(weights, length):
        return [np.pad(w.flatten(), (0, length - len(w.flatten())), 'constant') for w in weights]

    initial_flattened = flatten_and_pad(initial_weights, max_len)
    best_flattened = flatten_and_pad(best_weights, max_len)

    data = {
        'Initial Weights': initial_flattened,
        'Best Weights': best_flattened,
        'Best Accuracy': [best_accuracy] * len(initial_flattened),
        'Selected Classes': [selected_classes] * len(initial_flattened)
    }

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():

    train_images, train_labels, test_images, test_labels, selected_classes = load_and_preprocess_data()

    split_train_images = flatten_and_split(train_images)
    split_test_images = flatten_and_split(test_images)

    mapped_train_images = apply_mapping(split_train_images)
    mapped_test_images = apply_mapping(split_test_images)

    input_shape = (196,)
    learning_rate = 0.001
    l2_lambda = 0.001
    epochs = 100
    batch_size = 64

    model = create_model(input_shape, learning_rate, l2_lambda)

    initial_weights = model.get_weights()

    best_epoch, best_accuracy, best_weights = train_model(model, mapped_train_images, train_labels, mapped_test_images,
                                                          test_labels, epochs, batch_size)

    print(f"Best Epoch: {best_epoch}, Best Accuracy: {best_accuracy}")

    test_loss, test_acc = model.evaluate(mapped_test_images, test_labels)
    print(f"Final Test Accuracy: {test_acc}")

    export_to_excel(initial_weights, best_weights, best_accuracy, selected_classes)

    y_pred = model.predict(mapped_test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    plot_confusion_matrix(test_labels, y_pred_classes, classes=[0, 1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    main()
