from typing import List
from tqdm import tqdm
import os
import shutil


def move2results(out_root, to_root: List, end: str, split_tate=None):
    if split_tate is None:
        split_tate = [0.7, 0.3]

    train_len = int(len(os.listdir(out_root)) * split_tate[0])
    val_end_len = int(len(os.listdir(out_root)))
    # test_end_len = int(len(os.listdir(out_root)))
    for i in tqdm(range(1, train_len + 1)):
        out_path = os.path.join(out_root, str(i) + end)
        shutil.copy(out_path, to_root[0])
    for i in tqdm(range(train_len + 1, val_end_len + 1)):
        out_path = os.path.join(out_root, str(i) + end)
        shutil.copy(out_path, to_root[1])
    # for i in tqdm(range(val_end_len + 1, test_end_len + 1)):
    #     out_path = os.path.join(out_root, str(i) + end)
    #     shutil.copy(out_path, to_root[2])


imgs_out_root = 'test_data\\img_results'
imgs_to_root = ['test_data/data/train/images/', 'test_data/data/val/images/', 'test_data/data/test/images/']

labels_out_root = 'test_data\\label_results'
labels_to_root = ['test_data/data/train/masks/', 'test_data/data/val/masks/', 'test_data/data/test/masks/']

move2results(imgs_out_root, imgs_to_root, end=str('.jpg'))
move2results(labels_out_root, labels_to_root, end=str('.png'))
