# import streamlit as st
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the trained model
# model = load_model('braintumor.h5')
#
# st.title("Brain Tumor Classifier")
#
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Read the uploaded image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     image = cv2.resize(image, (150, 150))
#
#     # Display the uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#
#     # Process the image for prediction
#     img_array = np.array(image)
#     img_array = img_array.reshape(1, 150, 150, 3)
#
#     # Make prediction
#     prediction = model.predict(img_array)
#     predicted_class_index = np.argmax(prediction)
#
#     # Display prediction result
#     classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
#     st.write(f"Predicted class: {classes[predicted_class_index]}")




# using  svm
# import streamlit as st
# import cv2
# import numpy as np
# import pickle
# from sklearn.metrics import accuracy_score
#
# # Load the trained SVM model
# with open('svm_model.pkl', 'rb') as f:
#     svm_model = pickle.load(f)
#
# st.title("Brain Tumor Classifier")
#
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Read the uploaded image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, 1)
#     image = cv2.resize(image, (150, 150))
#
#     # Display the uploaded image
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#
#     # Process the image for prediction
#     img_array = np.array(image)
#     img_array = img_array.reshape(1, 150, 150, 3)
#
#     # Make prediction using SVM model
#     prediction_svm = svm_model.predict(img_array.flatten().reshape(1, -1))
#
#     # Define classes
#     classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
#
#     # Display prediction result from SVM model
#     st.write("Prediction by SVM model:", classes[prediction_svm[0]])
#
#     # Calculate accuracy based on the predicted class
#     true_label = 'pituitary_tumor'  # Assuming the true label for testing images
#     true_label_index = classes.index(true_label)
#     y_true = [true_label_index]
#     y_pred = [prediction_svm[0]]
#     accuracy_svm = accuracy_score(y_true, y_pred)
#
#     # Display accuracy of SVM model
#     st.write("Accuracy of SVM model:", accuracy_svm)



# currently working code
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the trained model
model = load_model('braintumor.h5')

st.title("Brain Tumor Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (150, 150))

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image for prediction
    img_array = np.array(image)
    img_array = img_array.reshape(1, 150, 150, 3)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Define classes
    classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    # Display prediction result
    if predicted_class_index == 0:
        st.write("Result: Glioma Tumor detected")
    elif predicted_class_index == 1:
        st.write("Result: Meningioma Tumor detected")
    elif predicted_class_index == 2:
        st.write("Result: Normal")
    else:
        st.write("Result: Pituitary Tumor detected")


#nrs :)
# import streamlit as st
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the trained model
# model = load_model('braintumor.h5')
#
# st.title("Brain Tumor Classifier")
#
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Check if the uploaded file is an image
#     if uploaded_file.type.startswith('image/'):
#         # Check the filename to ensure it is from the trained dataset
#         filename = uploaded_file.name.lower()
#         if any(label in filename for label in ['glioma', 'meningioma', 'no_tumor', 'pituitary']):
#             # Read the uploaded image
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             image = cv2.imdecode(file_bytes, 1)
#             image = cv2.resize(image, (150, 150))
#
#             # Display the uploaded image
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Process the image for prediction
#             img_array = np.array(image)
#             img_array = img_array.reshape(1, 150, 150, 3)
#
#             # Make prediction
#             prediction = model.predict(img_array)
#             predicted_class_index = np.argmax(prediction)
#
#             # Define classes
#             classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
#
#             # Display prediction result
#             if predicted_class_index == 0:
#                 st.write("Result: Glioma Tumor detected")
#             elif predicted_class_index == 1:
#                 st.write("Result: Meningioma Tumor detected")
#             elif predicted_class_index == 2:
#                 st.write("Result: Normal")
#             else:
#                 st.write("Result: Pituitary Tumor detected")
#         else:
#             st.write("Please upload an MRI image from the trained dataset.")
#     else:
#         st.write("Please upload an MRI image (jpg, jpeg, or png format).")
