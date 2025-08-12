# train.py (Final Version for Compatibility)

import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- 1. SETUP AND HYPERPARAMETERS ---
DATA_DIR = 'C:\\Users\\Samuel I\\Downloads\\malaria\\cell_images\\cell_images'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

print(f"TensorFlow Version: {tf.__version__}")

# --- 2. LOAD AND PREPROCESS DATA ---
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
print("Class Names:", class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD THE MODEL (TRANSFER LEARNING) - UPDATED FOR COMPATIBILITY ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
])

# Load the base model without its top layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Create the new model architecture
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
# THIS IS THE KEY CHANGE: We use a standard Rescaling layer instead of the problematic function.
# This scales pixel values from [0, 255] to [-1, 1], just like the old function.
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)


# --- 4. COMPILE THE MODEL ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
model.summary()

# --- 5. INITIAL TRAINING ---
print("\n--- Starting Initial Model Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS
)
print("--- Initial Model Training Finished ---")

# --- 6. PREPARE FOR FINE-TUNING ---
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
print("\n--- Model Re-compiled for Fine-Tuning ---")
model.summary()

# --- 7. FINE-TUNE THE MODEL ---
print("\n--- Starting Fine-Tuning ---")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1]
)
print("--- Fine-Tuning Finished ---")

# --- 8. VISUALIZE COMBINED TRAINING RESULTS ---
# Visualization code remains the same
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(INITIAL_EPOCHS -1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(INITIAL_EPOCHS -1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.ylim([0, 0.2])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.suptitle('Model Training History (with Fine-Tuning)', fontsize=16)
plt.show()

# --- 9. FINAL EVALUATION ---
final_loss, final_accuracy = model.evaluate(val_ds)
print("\n--- Final Model Evaluation After Fine-Tuning ---")
print(f"Final Validation Loss: {final_loss:.4f}")
print(f"Final Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# --- 10. SAVE THE FINAL MODEL ---
model.save('malaria_model.h5')
print("\n--- Model saved successfully as malaria_model.h5 ---")
