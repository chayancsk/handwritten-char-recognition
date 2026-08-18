import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from model import build_cnn

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_data()

    model = build_cnn(input_shape=(28, 28, 1), num_classes=10)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=[early_stop],
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    model.save("models/digit_model.h5")
    print("Saved -> models/digit_model.h5")


if __name__ == "__main__":
    main()