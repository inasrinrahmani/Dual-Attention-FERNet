import os
import numpy as np
import cv2
import struct
from sklearn.model_selection import train_test_split

data_dir = "F:\\__dualattention_fernet\\main"

images = []
labels = []

for filename in os.listdir(data_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(data_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        images.append(image)

        if filename.startswith("angry"):
            labels.append(0)
        elif filename.startswith("disgust"):
            labels.append(1)
        elif filename.startswith("fear"):
            labels.append(2)
        elif filename.startswith("happy"):
            labels.append(3)
        elif filename.startswith("sad"):
            labels.append(4)
        elif filename.startswith("surprise"):
            labels.append(5)
        elif filename.startswith("neutral"):
            labels.append(6)

images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.uint8)

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.10, stratify=labels, random_state=42
)

x_train.shape

np.savez("face.npz", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

import matplotlib.pyplot as plt
plt.imshow(x_train[10],cmap='gray')

