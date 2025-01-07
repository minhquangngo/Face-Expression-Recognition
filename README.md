# Face Expression Recognition Extracted using HOG
This project revisits classical machine learning models, such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM), to evaluate their performance in Facial Expression Recognition tasks. Using the widely adopted CK+ Dataset and Histogram of Oriented Gradients (HOG) for feature extraction, I compare the effectiveness of these simpler models against state-of-the-art FER methods 

## Histogram of Oriented Gradients (HOG)
HOG is a widely used feature extraction method in computer vision that captures image details by analyzing gradients in pixel intensity. Images are divided into 8x8 pixel cells, where gradient magnitudes and orientations are computed to create histograms. These histograms, normalized over 4x4 blocks to account for lighting variations, form HOG feature vectors that represent the image’s structure for further analysis.

![hogvisual](https://github.com/user-attachments/assets/9b80cb70-bb05-449b-b689-8fbf90cfde12)

The HOG is implemented using scikit-image. KNN (with 1 neighbour) and SVM (with rbf) is implemented using scikit-learn

## Dependencies
random, os, pickle, matplotlib, matplotlib.pyplot, cv2 (OpenCV), numpy (np), pandas (pd), sklearn.model_selection (KFold, cross_val_score, GridSearchCV, train_test_split), sklearn.svm (SVC), sklearn.metrics (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay), skimage.feature (hog), skimage.color, imutils.paths, seaborn (sns).


## The dataset can be obtained from here
[Kaggle](https://www.kaggle.com/datasets/shawon10/ckplus)

## Results
KNN accuracy: 94.6%
SVM accuracy:96.7%

