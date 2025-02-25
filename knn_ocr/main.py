import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
from pathlib import Path

train_data_dir = "task/train/"
test_data_dir = "task/"

def get_data_for_training(train_data_dir, image_size=(32, 32)):
    train_data_dir = Path(train_data_dir)

    features = []
    labels = []
    class_names = []

    for label_idx, label_name in enumerate(sorted(train_data_dir.iterdir())): 
        if not label_name.is_dir():
            continue

        class_names.append(label_name.name)
        
        for image_path in label_name.glob("*.png"):
            try:
                template = plt.imread(image_path)[:, :, :3].mean(axis=2)
                template = (template > 0).astype(np.uint8)

                labeled_image = label(template)
                regions = regionprops(labeled_image)

                if not regions:
                    print(f"No regions found in {image_path}")
                    continue 

                target_region = max(regions, key=lambda r: r.area)
                extracted_features = extract_features(target_region, image_size)
                features.append(extracted_features)
                labels.append(label_idx)
              
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)

    return features, labels
