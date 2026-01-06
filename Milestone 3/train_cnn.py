import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "wiki")
MODEL_DIR = os.path.join(BASE_DIR, "models")


os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE =128 
EPOCHS = 30
BATCH_SIZE = 1


X = []
y = []

print("Reading images from:", IMAGE_DIR)


if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, file)

        
        label = file.split("_")[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Scanner labels:", set(y))


le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

print("Encoded classes:", le.classes_)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    X,
    y_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


model_path = os.path.join(MODEL_DIR, "cnn_scanner.h5")
model.save(model_path)



acc = history.history["accuracy"]
loss = history.history["loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, marker='o')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()



final_accuracy = history.history["accuracy"][-1] * 100
plt.savefig("training_curves.png")

print("\n✅ Training completed successfully")
print(f"🎯 Final Training Accuracy: {final_accuracy:.2f}%")
print("📦 Model saved at:", model_path)
print("🧾 Scanner Classes:", le.classes_)