import os
import shutil
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


# Example usage
all_classes = [
    'Non-uniform color', 'Tooling marks', 'Exposed copper on the ends of the leads',
    'Bent or nonplanar leads', 'Excessive or uneven plating', 'Missing Pins',
    'Discoloration, dirt, or residues on the leads',
    'Scratches (or insertion marks) on the inside and outside faces of the leads',
    'Gross Oxidation', 'Excessive solder on the leads', 'Non-uniform thickness'
]

split_data('D:/DEFECTIVE IMAGES', 'D:/TRAIN IMAGES/TRAIN', 'D:/TRAIN IMAGES/VALIDATION', all_classes=all_classes)

