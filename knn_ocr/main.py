import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
from pathlib import Path

train_data_dir = "task/train/"
test_data_dir = "task/"
