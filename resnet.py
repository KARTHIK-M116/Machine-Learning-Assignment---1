import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set dataset path (Update with your actual path)
dataset_path = r"C:\Users\jaswa\OneDrive\Documents\Vehicles"  # Update this path
categories = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]

# Image parameters
image_size = (224, 224)  # ResNet requires 224x224 images

# Load Pretrained ResNet Model (Feature Extractor)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.eval()  # Set to evaluation mode

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Prepare dataset
data = []
labels = []

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Assign numeric labels
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        # Read image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable files
        
        # Convert to PIL image & apply transformations
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Extract features using ResNet
        with torch.no_grad():
            features = resnet(img).squeeze().numpy()  # Flatten features
        
        data.append(features)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
svm = SVC(kernel="linear", C=1.0)
svm.fit(X_train, y_train)

# Predictions
y_pred_svm = svm.predict(X_test)

# Print Accuracy & Classification Report
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred_svm))
