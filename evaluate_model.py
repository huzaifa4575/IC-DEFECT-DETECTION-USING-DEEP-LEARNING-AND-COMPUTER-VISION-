import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, val_gen):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Reset the generator
    val_gen.reset()

    # Make predictions
    predictions = model.predict(val_gen, verbose=1)
    y_pred = predictions.argmax(axis=1)
    y_true = val_gen.classes

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(val_gen.class_indices.values()))

    # Print class indices and confusion matrix shape for debugging
    print("Class Indices:", val_gen.class_indices)
    print("Confusion Matrix Shape:", cm.shape)
    print("Unique Labels in Predictions:", set(y_pred))
    print("Unique Labels in Ground Truth:", set(y_true))

    # Display confusion matrix
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_gen.class_indices.keys()))
        fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)  # Hide colorbar if desired
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.yticks(rotation=0)  # Rotate y-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        plt.show()
    except ValueError as e:
        print("Error displaying confusion matrix:", e)

# Define the validation data generator
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_gen = val_datagen.flow_from_directory(
    'D:/TRAIN IMAGES/VALIDATION',  # Update this path to your validation data directory
    target_size=(150, 150),  # Update this size based on your model's input size
    batch_size=32,
    class_mode='categorical',  # or 'binary' if it's a binary classification problem
    shuffle=False
)

# Call the evaluation function
evaluate_model('C:/Users/Huzaifa/PycharmProjects/DEFECT DETECTION/defect_model.keras', val_gen)
