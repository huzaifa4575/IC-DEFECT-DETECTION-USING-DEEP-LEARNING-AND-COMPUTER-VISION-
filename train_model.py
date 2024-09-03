import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def split_data(input_dir, train_dir, val_dir, split_ratio=0.8, all_classes=None):
    if all_classes is None:
        raise ValueError("All classes must be provided")

    # Get a list of all images with their corresponding subfolder paths
    images = []
    labels = []
    existing_classes = set()

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract label based on subfolder name
                label = os.path.basename(root)
                images.append(os.path.join(root, file))
                labels.append(label)
                existing_classes.add(label)

    print(f"Total images found: {len(images)}")  # Print number of images found

    # Ensure there are images to split
    if len(images) == 0:
        raise ValueError("No images found in the specified directory.")

    # Ensure that all specified classes are included
    all_classes = set(all_classes)  # Ensure all_classes is a set
    missing_classes = all_classes - existing_classes
    if missing_classes:
        print(f"Warning: These classes are missing in the dataset: {missing_classes}")

    # Split the data into training and validation sets
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(images, labels, train_size=split_ratio)

    # Create directories for each label if they don't exist
    for label in all_classes:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)

    # Move images to training and validation directories with subfolder structure
    for img, label in zip(train_imgs, train_labels):
        shutil.copy(img, os.path.join(train_dir, label, os.path.basename(img)))

    for img, label in zip(val_imgs, val_labels):
        shutil.copy(img, os.path.join(val_dir, label, os.path.basename(img)))

    print(f"Images moved to {train_dir} and {val_dir}")


def get_data_generators(train_dir, val_dir, img_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator


def build_model(input_shape=(150, 150, 3), num_classes=11):
    base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze base model layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='relu'),  # Increased number of neurons
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),  # Added additional dense layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Adjusted learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load data generators
train_gen, val_gen = get_data_generators('D:/TRAIN IMAGES/TRAIN', 'D:/TRAIN IMAGES/VALIDATION')

# Build and compile the model
model = build_model()

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=55,  # Set to 40 epochs
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5)
    ]
)

# Save the trained model
model.save('C:/Users/Huzaifa/PycharmProjects/DEFECT DETECTION/defect_model.keras')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()