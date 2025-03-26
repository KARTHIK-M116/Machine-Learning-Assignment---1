import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern  # Import LBP function
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Dataset path (update with your actual path)
dataset_path = r"C:\Users\jaswa\OneDrive\Documents\Vehicles"
categories = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]

# Image parameters
image_size = (128, 128)  # Resize images to 128x128
radius = 1  # LBP radius
n_points = 8 * radius  # LBP sample points

# Function to extract Local Binary Patterns (LBP) features
def extract_lbp(image):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")  # Compute LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)  # Histogram
    return hist  # Return LBP histogram

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
        
        # Extract LBP features
        lbp_features = extract_lbp(img)
        
        # Store features and label
        data.append(lbp_features)
        labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
svm = SVC(kernel="linear", C=1.0)
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)

# Evaluate model performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

# Print precision, recall, F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))
