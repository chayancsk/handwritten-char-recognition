import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from model import build_cnn

NUM_CLASSES = 26


def _to_numpy(ds):
    images, labels = [], []
    for img, label in tfds.as_numpy(ds):
        img = np.transpose(img.squeeze(), (1, 0))
        images.append(img)
        labels.append(label)
    x = np.array(images, dtype="float32") / 255.0
    x = np.expand_dims(x, -1)
    y = np.array(labels) - 1
    y = to_categorical(y, NUM_CLASSES)
    return x, y


def load_data():
    ds_train, ds_test = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
    )
    x_train, y_train = _to_numpy(ds_train)
    x_test, y_test = _to_numpy(ds_test)
    return x_train, y_train, x_test, y_test


def main():
    print("Downloading/loading EMNIST letters (first run can take a while)...")
    x_train, y_train, x_test, y_test = load_data()

    model = build_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)
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

    model.save("models/letter_model.h5")
    print("Saved -> models/letter_model.h5")


if __name__ == "__main__":
    main()