from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt



# Load the trained model
model = load_model('braintumor.h5')

# Now you can run the prediction cell
img_path = 'C:/Users/91730/PycharmProjects/rs/dataset/Training/pituitary_tumor/p (107).jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))
img_array = np.array(img)
img_array = img_array.reshape(1, 150, 150, 3)


# # Display the image
plt.imshow(img, interpolation='nearest')
plt.show()

a = model.predict(img_array)
indices = a.argmax()
# indices
print("Predicted class index:", indices)
