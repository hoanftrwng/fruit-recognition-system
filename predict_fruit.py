import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import os

def predict_image(img_path, model_path = 'D:/PyNet/fruit_recognition/nn/fruit_model.h5', class_indices_path='D:/PyNet/fruit_recognition/dataset/images/class_indices.json'):
    # Load model đã huấn luyện
    model = load_model(model_path)

    # Load ánh xạ class (ví dụ {'banana': 0, 'orange': 1, 'apple': 2})
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}  # Đảo ngược: {0: 'banana', ...}

    # Load ảnh test
    img = load_img(img_path, target_size=(64, 64))  # Resize như khi train
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension: (1, 64, 64, 3)

    # Dự đoán
    pred = model.predict(img)[0]  # Kết quả là 1 array các xác suất
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]

    # In kết quả
    print(f"Ảnh: {os.path.basename(img_path)}")
    print(f"==> Dự đoán: {labels[predicted_class]} ({confidence*100:.2f}%)")

if __name__ == '__main__':
    predict_image('D:/PyNet/fruit_recognition/dataset/images/banana/r_7_100.jpg')
