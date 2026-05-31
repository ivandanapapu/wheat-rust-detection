import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join("..", "backend", "model"), exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)

all_labels = []
for _, labels in train_ds:
    all_labels.extend(labels.numpy().ravel().astype(int))

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=np.array(all_labels)
)

class_weights = {
    0: float(class_weights_values[0]),
    1: float(class_weights_values[1])
}

print("Class Weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.15),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "wheat_rust_cnn.keras"),
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

true_labels = []
pred_probs = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0).ravel()
    pred_probs.extend(probs)
    true_labels.extend(labels.numpy().ravel())

true_labels = np.array(true_labels).astype(int)
pred_probs = np.array(pred_probs)
pred_labels = (pred_probs >= 0.5).astype(int)

precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

fn = cm[1][0]
tp = cm[1][1]
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

report = {
    "class_names": class_names,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "false_negative_rate": float(false_negative_rate)
}

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "accuracy_plot.png"))

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))

shutil.copy(
    os.path.join(MODEL_DIR, "wheat_rust_cnn.keras"),
    os.path.join("..", "backend", "model", "wheat_rust_cnn.keras")
)

shutil.copy(
    os.path.join(MODEL_DIR, "class_names.json"),
    os.path.join("..", "backend", "model", "class_names.json")
)

print("Model and class names copied to backend/model successfully.")