import cv2
import numpy as np
import os
from skimage.measure import label, regionprops
from pathlib import Path
from skimage.io import imread

TRAIN_DIR = "task/train/"
TEST_IMAGES = "task/" 
MIN_REGION_SIZE = 250
KNN_NEIGHBORS = 2
SPACE_THRESHOLD = 30

def extract_features(region):
    img = region.image
    h, w = img.shape

    area = img.sum() / img.size
    perimeter = region.perimeter / img.size
    wh_ratio = h / w if w > 0 else 0

    cy, cx = region.local_centroid
    cy /= h
    cx /= w

    euler = region.euler_number
    central_region = img[int(0.45 * h):int(0.55 * h), int(0.45 * w):int(0.55 * w)]
    kl = 3 * central_region.sum() / img.size if img.size > 0 else 0
    kls = 2 * central_region.sum() / img.size if img.size > 0 else 0
    eccentricity = region.eccentricity * 8 if hasattr(region, 'eccentricity') else 0

    have_v1 = (np.mean(img, axis=0) > 0.87).sum() > 2
    have_g1 = (np.mean(img, axis=1) > 0.85).sum() > 2
    have_g2 = (np.mean(img, axis=1) > 0.5).sum() > 2

    hole_size = img.sum() / region.filled_area if region.filled_area > 0 else 0
    solidity = region.solidity * 2 if hasattr(region, 'solidity') else 0

    return np.array([
        area, perimeter, cy, cx, euler, eccentricity, have_v1 * 3,
        hole_size, have_g1 * 4, have_g2 * 5, kl, wh_ratio, kls, solidity
    ])


def load_training_data(training_dir):
    training_dir = Path(training_dir)
    features, labels_list = [], []

    for label_idx, label_name in enumerate(training_dir.iterdir()):
        if not label_name.is_dir():
            continue

        for image_path in label_name.glob("*.png"):
            try:
                image = imread(image_path, as_gray=True)

                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue

                image[image > 0] = 1

                labeled_image = label(image)
                regions = regionprops(labeled_image)

                if not regions:
                    print(f"Warning: No regions found in image {image_path}")
                    continue
                    
                target_region = max(regions, key=lambda r: r.area)

                # Extract features
                feature_vector = extract_features(target_region)
                features.append(feature_vector)
                labels_list.append(label_idx)

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    return np.array(features), np.array(labels_list)

def train_knn(features, labels):
    knn = cv2.ml.KNearest_create()
    knn.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32).reshape(-1, 1))
    return knn

def recognize_text(image_path, knn, labels):
    try:
        image = imread(image_path, as_gray=True)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return ""

        image[image > 0] = 1
        labeled_image = label(image)
        regions = regionprops(labeled_image)
        regions = sorted(regions, key=lambda x: x.centroid[1])

        recognized_text = ""
        last_x_right = 0 

        for i, region in enumerate(regions):
            if np.sum(region.image) > MIN_REGION_SIZE:
                feature_vector = extract_features(region)
                reshaped_feature_vector = feature_vector.reshape(1, -1).astype(np.float32)
                ret, results, neighbors, dist = knn.findNearest(reshaped_feature_vector, KNN_NEIGHBORS)
                predicted_label_index = int(ret)

                current_x_left = region.bbox[1]
                if i > 0 and (current_x_left - last_x_right) > SPACE_THRESHOLD:
                    recognized_text += " "

                recognized_text += labels[predicted_label_index][-1]
                last_x_right = region.bbox[3]

        return recognized_text
    except Exception as e:
        print(f"Problem loading the image: {image_path}, exception: {e}")
        return ""

training_features, training_targets = load_training_data(TRAIN_DIR)

if training_features.size > 0 and training_targets.size > 0:
    knn = train_knn(training_features, training_targets)
else:
    print("Error: No training data loaded.  Cannot train KNN.")
    exit()

labels = os.listdir(TRAIN_DIR)

image_files = [f for f in os.listdir(TEST_IMAGES) if f.endswith(('.png', '.jpg', '.jpeg'))]
for i, image_file in enumerate(image_files):
    image_path = os.path.join(TEST_IMAGES, image_file)
    recognized_text = recognize_text(image_path, knn, labels)
    print(f"Image {i+1}: {recognized_text}")
