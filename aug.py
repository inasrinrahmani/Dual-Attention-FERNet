import os
import cv2
import numpy as np

data_dir = "F:\\__dualattention_fernet\\main"
class_name = "disgust"  
target_samples_per_class = 8989

angles = [5, 10, 15, 20, 25, 30]

def flip_image(image):
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def invert_contrast(image):
    return cv2.bitwise_not(image)

images = []  
labels = []  
for filename in os.listdir(data_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        img_path = os.path.join(data_dir, filename)
        if filename.startswith(class_name):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(class_name)

num_current_samples = len(images)
augmented_images = []
augmented_labels = []

augmented_images_dir = data_dir  

while len(augmented_images) + num_current_samples < target_samples_per_class:
    for i, image in enumerate(images):
        flipped_image = flip_image(image)
        cv2.imwrite(os.path.join(augmented_images_dir, f"{class_name}_flipped_{len(augmented_images)}.jpg"), flipped_image)
        augmented_images.append(flipped_image)
        
        for angle in angles:
            rotated_image = rotate_image(image, angle)
            cv2.imwrite(os.path.join(augmented_images_dir, f"{class_name}_rotated_{angle}_{len(augmented_images)}.jpg"), rotated_image)
            augmented_images.append(rotated_image)
        
        inverted_image = invert_contrast(image)
        cv2.imwrite(os.path.join(augmented_images_dir, f"{class_name}_inverted_{len(augmented_images)}.jpg"), inverted_image)
        augmented_images.append(inverted_image)
        
        if len(augmented_images) + num_current_samples >= target_samples_per_class:
            break

print(f"Number of samples for {class_name}: {len(augmented_images) + num_current_samples}")
