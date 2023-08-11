import os
import cv2 as cv
from tqdm import tqdm
import numpy as np

mask_dir = r"data\crackdataset2\test\masks"
mask_01_dir = r"data\crackdataset2_mask01\masks\test"

if not os.path.exists(mask_01_dir):
    os.mkdir(mask_01_dir)


def preprocess_masks(mask):
    mask[(mask >= 0.0) & (mask <= 10.0)] = 0
    mask[(mask > 10.0) & (mask <= 255.0)] = 1
    mask = mask.astype(np.int32)
    return mask


for mask in tqdm(os.listdir(mask_dir)):
    mask_path = os.path.join(mask_dir, mask)
    mask_img = cv.imread(mask_path, -1)
    # mask_img_pre = np.divide()
    mask_img_pre = preprocess_masks(mask_img)
    mask_01_img_path = os.path.join(mask_01_dir, mask)
    cv.imwrite(mask_01_img_path, mask_img_pre, [cv.IMWRITE_PNG_COMPRESSION, 0])

mask_path = os.path.join(mask_01_dir, "1_Crack_7_73_269.png")
mask_img = cv.imread(mask_path, -1)
unique, count = np.unique(mask_img, return_counts=True)
data_count = dict(zip(unique, count))
print(data_count)
