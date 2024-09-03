Deep Learning Model for Defect Detection in Integrated Circuit (IC) Leads

Overview:
This project focuses on developing a deep learning model to detect multiple types of defects in Integrated Circuit (IC) leads, which are crucial in ensuring the reliability and performance of electronic components. The model leverages computer vision techniques using TensorFlow, Keras, and OpenCV to perform accurate multi-class classification on high-resolution images of IC leads.

Objectives:
The primary goal of this project is to build a robust and scalable model that can:
- Detect various types of defects in IC leads with high accuracy.
- Generalize well to unseen data in real-world scenarios where multiple defects may be present in a single image.
- Provide an efficient and automated quality control mechanism for the electronics manufacturing industry.

Defects Detected:
The model is designed to detect the following defects in IC leads:
1. Non-uniform color
2. Tooling marks
3. Exposed copper on the ends of the leads
4. Bent or non-planar leads
5. Excessive or uneven plating
6. Missing pins
7. Discoloration, dirt, or residues on the leads
8. Scratches (or insertion marks) on the inside and outside faces of the leads
9. Gross oxidation
10. Excessive solder on the leads
11. Non-uniform thickness

Methodology and Key Steps:

1. Data Preparation and Augmentation:
- Image Data Organization: High-resolution images (1920x1200) were organized into separate folders for each defect type, facilitating clear labeling and batch generation.
- Data Augmentation: To address data imbalance, techniques such as random flipping, rotation, brightness variation, and zooming were applied to augment the dataset, ensuring a more balanced and diverse training set.

 2. Model Architecture and Training:
- Base Model: A pre-trained VGG16 model was used as the backbone for transfer learning. The final layers were modified to include fully connected layers and a softmax activation function for multi-class classification.
- Training Strategy: The model was trained using the Adam optimizer with a carefully tuned learning rate. Advanced techniques like early stopping and learning rate reduction on plateau were employed to prevent overfitting and optimize convergence.
- Model Performance: Achieved an overall accuracy of 85% on the validation set, with a good balance between bias and variance.

 3. Evaluation and Performance Metrics:
- Accuracy and Loss Monitoring: Training and validation accuracy and loss were plotted to monitor performance and diagnose issues like overfitting or underfitting.
- Confusion Matrix Analysis: A confusion matrix was used to visualize model performance across different defect classes, providing insights into strengths and weaknesses.
- Results: High true positive rates for defects like 'Missing Pins' and 'Excessive Solder' were achieved, while defects with similar visual characteristics, such as 'Non-uniform color' and 'Discoloration,' required further fine-tuning.

 4. Testing on New Data and Real-World Scenarios:
- The model was tested on new data directly stored in the folder `F:\Image\temp`. The results were formatted into a CSV file, with each row representing an image and each column indicating the detection status ('Detected' or 'Not Detected') for each defect type.
- This approach demonstrated the model’s robustness in handling complex, multi-defect scenarios.

 5. Visualization and Interpretation:
- Training Visualization: Plotted graphs for training accuracy, loss, and confusion matrices to visualize model performance.
- Confusion Matrix Visualization: Helped in interpreting model performance and identifying areas for improvement in defect detection.

 Future Directions:
- Model Improvement: Future work includes integrating ensemble models, leveraging synthetic data generation techniques such as GANs, and exploring more advanced architectures like EfficientNet to improve model performance further.
- Real-World Deployment: Plans to develop a real-time detection system for deployment in manufacturing environments to automate quality control processes.


 How to Run the Code
1. Clone the Repository:  
   ```bash
   git clone https://github.com/yourusername/defect-detection-IC-leads.git
   cd defect-detection-IC-leads
   ```

2. Install Dependencies:  
   Ensure you have Python 3.7 or above and install all required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare Your Data:  
   Organize your IC lead images into respective folders as per the defect types. Ensure data is formatted correctly for the scripts to work.

4. Run Training:  
   Use the `pretrained_model.py` to start training:
   ```bash
   python pretrained_model.py
   ```

5. Visualize Results:  
   Use `confusion_matrix.py` and `model_accuracy_loss_visualization.py` to visualize the model's performance.

 Dependencies
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Numpy
- Pandas

 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.


 Contact
For any queries or collaboration opportunities, feel free to reach out!



