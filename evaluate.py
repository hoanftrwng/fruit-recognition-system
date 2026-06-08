import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình ---
model_path = 'D:/PyNet/fruit_recognition/nn/fruit_model.h5'
test_dir = 'D:/PyNet/fruit_recognition/dataset/images'
labels = ['apple', 'banana', 'grape', 'orange', 'pear', 'watermelon']

img_size = (64, 64)

# --- Load model ---
model = load_model(model_path)

# --- Load dữ liệu test ---
y_true = []
y_pred = []

for idx, label in enumerate(labels):
    folder = os.path.join(test_dir, label)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            pred_label = np.argmax(pred)
            y_true.append(idx)
            y_pred.append(pred_label)
# --- In kết quả ---
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
