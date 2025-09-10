



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

# ----------------------------- #
# CONFIGURATION
# ----------------------------- #
train_dir = "cropped_split/train"
val_dir = "cropped_split/val"
img_size = (224, 224)
batch_size = 32
epochs_transfer = 100
epochs_finetune = 100
early_stop_patience = 10

# ----------------------------- #
# DATA PREPARATION
# ----------------------------- #
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ----------------------------- #
# MODEL: VGG16
# ----------------------------- #
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base for transfer learning

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------- #
# EARLY STOPPING
# ----------------------------- #
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=early_stop_patience,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

# ----------------------------- #
# TRAINING: TRANSFER LEARNING
# ----------------------------- #
start_time = time.time()

history_transfer = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_transfer,
    callbacks=[early_stop]
)

# ----------------------------- #
# TRAINING: FINE-TUNING
# ----------------------------- #
base_model.trainable = True  # Unfreeze for fine-tuning

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=[early_stop]
)

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")

# ----------------------------- #
# SAVE MODEL
# ----------------------------- #
model.save("models/vgg19_finetuned_fish_species_split.keras")
print("Model saved as 'vgg16_finetuned_fish_species_split.keras'")

# ----------------------------- #
# PLOT & SAVE LOSS AND ACCURACY
# ----------------------------- #
loss = history_transfer.history['loss'] + history_finetune.history['loss']
val_loss = history_transfer.history['val_loss'] + history_finetune.history['val_loss']
accuracy = history_transfer.history['accuracy'] + history_finetune.history['accuracy']
val_accuracy = history_transfer.history['val_accuracy'] + history_finetune.history['val_accuracy']
epochs_range = range(1, len(loss) + 1)
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])

plt.figure(figsize=(10, 4), dpi=150)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Loss per Epoch (VGG16)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.title('Accuracy per Epoch (VGG16)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('graphs/loss_accuracy_vgg16.png')
print("Loss and accuracy graphs saved as 'loss_accuracy_vgg16.png'")
plt.show()

# ----------------------------- #
# CONFUSION MATRIX (Save)
# ----------------------------- #
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8), dpi=150)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (VGG16)')
plt.tight_layout()
plt.savefig('graphs/confusion_matrix_vgg16.png')
print("Confusion matrix saved as 'confusion_matrix_vgg16.png'")
plt.show()


total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

print(f"Total training time: {time_str} (HH:MM:SS)")

# Save training report
with open("graphs/training_report_vgg.txt", "w") as f:
    f.write(f"Total training time: {time_str} (HH:MM:SS)\n")
    best_val_acc = max(val_accuracy)
    f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
    f.write(f"Total epochs run: {len(epochs_range)}\n")
    f.write(f"Total trainable parameters {trainable_params}\n")
print("Training report saved as 'training_report_vgg.txt'")
