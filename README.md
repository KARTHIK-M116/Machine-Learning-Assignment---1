# Machine Learning Assignment - Image Feature Extraction and Classification

## Problem Statement
The goal of this project is to explore various image feature extraction techniques and analyze their impact on classification performance across different machine learning models. By applying both traditional and deep learning-based feature extraction methods, we aim to improve accuracy in image classification tasks.

## Dataset
For this experiment, we use a synthetic vehicle image dataset similar to CIFAR-10, containing 5000 images categorized into 6 classes, including airplanes, cars, and ships. The dataset is split into:
- 80 percent Training Set - Used to train models
- 20 percent Testing Set - Used for evaluation

## Approach
We apply two main feature extraction techniques:

### Traditional Feature Extraction Methods
- Histogram of Oriented Gradients HOG: Detects edge-based features.
- Local Binary Pattern LBP: Captures texture patterns.

### Deep Learning-Based Feature Extraction
- ResNet-50: Captures high-level hierarchical features.
- VGG-16: Extracts spatial hierarchies from images.

## Model Training and Evaluation

### Classifiers Used
- Support Vector Machines SVM: Works well with HOG and LBP.
- Deep Learning Models ResNet and VGG-16: Extract features and classify images using fully connected layers.

### Evaluation Metrics
- Accuracy - Percentage of correct predictions.
- Precision and Recall - Measures reliability of classification.
- F1-Score - Balance between precision and recall.

## Results and Observations

| Feature Extraction Method | Classifier | Accuracy |
|--------------------------|------------|----------|
| HOG | SVM | 78 percent |
| LBP | SVM | 52 percent |
| ResNet-50 | CNN | 98.3 percent |
| VGG-16 | CNN | 98 percent |

## Innovative Approach and Insights
- Combination of Traditional and Deep Learning Methods: Using HOG or LBP for SVM and CNN-based feature extraction significantly improves classification accuracy.
- Fine-tuning CNN models ResNet-50, VGG-16 achieves 98 percent or higher accuracy.
- Feature extraction reduces computational load, making deep models more efficient for image classification.

## How to Run the Code

### Clone the repository
```bash
git clone https://github.com/KARTHIK-M116/Machine-Learning-Assignment-1.git
cd Machine-Learning-Assignment-1
