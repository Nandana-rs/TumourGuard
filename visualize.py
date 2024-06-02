import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Load training history from history.pkl
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)
# Extract accuracy and loss values
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(len(acc))

# Plotting Training/Validation Accuracy
# plt.figure(figsize=(14, 7))
# plt.plot(epochs, acc, 'r', label="Training Accuracy")
# plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
# plt.legend(loc='upper left')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

# Plotting Training/Validation Loss
plt.figure(figsize=(14, 7))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper left')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
