# import os
# import cv2
# import numpy as np
# from sklearn import svm
# from sklearn.metrics import accuracy_score
#
# # Define paths for dataset
# data_dir = r'C:\Users\91730\PycharmProjects\rs\dataset'
# training_dir = os.path.join(data_dir, 'Training')
# testing_dir = os.path.join(data_dir, 'Testing')
#
# # Prepare data for SVM training
# X_train = []
# y_train = []
#
# # Load and preprocess training images
# training_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
#
# for label in training_labels:
#     train_folder = os.path.join(training_dir, label)
#     for filename in os.listdir(train_folder):
#         img_path = os.path.join(train_folder, filename)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (150, 150))
#         img_array = np.array(img)
#         img_array = img_array.reshape(1, 150, 150, 3)
#
#         # Append image and label
#         X_train.append(img_array.flatten())  # Flatten image array for SVM input
#         y_train.append(training_labels.index(label))  # Label index for SVM
#
# # Convert lists to numpy arrays
# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# # Load trained SVM model
# svm_model = svm.SVC()
# svm_model.fit(X_train, y_train)
#
# # Prepare data for SVM testing
# X_test = []
# y_true = []
#
# # Load and preprocess test images
# for label in training_labels:
#     test_folder = os.path.join(testing_dir, label)
#     for filename in os.listdir(test_folder):
#         img_path = os.path.join(test_folder, filename)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (150, 150))
#         img_array = np.array(img)
#         img_array = img_array.reshape(1, 150, 150, 3)
#
#         # Append image and true label
#         X_test.append(img_array.flatten())  # Flatten image array for SVM input
#         y_true.append(training_labels.index(label))  # True label index for accuracy calculation
#
# # Make predictions using SVM
# y_pred_svm = svm_model.predict(X_test)
#
# # Calculate accuracy
# accuracy_svm = accuracy_score(y_true, y_pred_svm)
#
# print("Accuracy of SVM model:", accuracy_svm)


import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Define paths for dataset
data_dir = r'C:\Users\91730\PycharmProjects\rs\dataset'
training_dir = os.path.join(data_dir, 'Training')

# Prepare data for SVM training
X_train = []
y_train = []

# Load and preprocess training images
training_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

for label in training_labels:
    train_folder = os.path.join(training_dir, label)
    for filename in os.listdir(train_folder):
        img_path = os.path.join(train_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3)

        # Append image and label
        X_train.append(img_array.flatten())  # Flatten image array for SVM input
        y_train.append(training_labels.index(label))  # Label index for SVM

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train SVM model
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

# Save the trained SVM model using pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Calculate and print accuracy
y_pred_svm = svm_model.predict(X_train)
accuracy_svm = accuracy_score(y_train, y_pred_svm)
print("Accuracy of SVM model:", accuracy_svm)

