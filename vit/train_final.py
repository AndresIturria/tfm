from vision_transformer import create_VisionTransformer
import tensorflow as tf
from pathlib import Path
import tensorflow as tf
from pathlib import Path
import seaborn as sns
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom
)
model = create_VisionTransformer(8)


# ----------------------------- #
# CONFIGURATION
# ----------------------------- #
train_dir = "cropped_split/train"
val_dir = "cropped_split/val"
img_size = (224, 224)
batch_size = 32
NUMBER_OF_EPOCHS = 100
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



# ---- Compile your model ----
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)


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
start_time = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=NUMBER_OF_EPOCHS,
    callbacks=[early_stop]
)





# ---- Save trained model ----
model.save("models/vision_transformer_model.keras")

# Report
loss = history.history['loss']
val_loss = history.history['val_loss'] 
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs_range = range(1, len(loss) + 1)
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

print(f"Total training time: {time_str} (HH:MM:SS)")

# Save training report
with open("graphs/training_report_vit.txt", "w") as f:
    f.write(f"Total training time: {time_str} (HH:MM:SS)\n")
    best_val_acc = max(val_accuracy)
    f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
    f.write(f"Total epochs run: {len(epochs_range)}\n")
    f.write(f"Total trainable parameters {trainable_params}\n")
print("Training report saved as 'training_report_vit.txt'")



# ----------------------------- #
# PLOT & SAVE LOSS AND ACCURACY
# ----------------------------- #



plt.figure(figsize=(10, 4), dpi=150)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Loss per Epoch (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.title('Accuracy per Epoch (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('graphs/loss_accuracy_vit.png')
print("Loss and accuracy graphs saved as 'loss_accuracy_vit.png'")
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
plt.title('Confusion Matrix (VIT)')
plt.tight_layout()
plt.savefig('graphs/confusion_matrix_vit.png')
print("Confusion matrix saved as 'confusion_matrix_vit.png'")
plt.show()

