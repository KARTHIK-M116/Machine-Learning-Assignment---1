import os
import cv2
import numpy as np
from skimage.feature import hog

# Define directories
dataset_path = r"C:\Users\jaswa\OneDrive\Documents\Vehicles"  # Change this to your dataset path
categories = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]

# Parameters
image_size = (128, 128)  # Resize images to 128x128
hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}

# Prepare dataset
data = []
labels = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign numeric labels
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is None:
            continue  # Skip unreadable files
        
        img = cv2.resize(img, image_size)  # Resize
        
        # Normalize image (optional)
        img = img.astype("float32") / 255.0
        
        # Extract HOG features
        hog_features = hog(img, **hog_params)
        
        # Store features and label
        data.append(hog_features)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Assuming 'data' contains HOG features and 'labels' contains corresponding classes
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel="linear", C=1.0)  # Linear kernel is good for HOG features
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

# Print detailed classification report
print(classification_report(y_test, y_pred_svm))
