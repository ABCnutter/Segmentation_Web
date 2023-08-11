import os
import shutil
from tqdm import tqdm


def read_name(filename):
    filename_base = os.path.splitext(filename)[0]
    return filename_base


def move_files(root_path, moved_path):
    root_filenames = os.listdir(root_path)
    moved_filenames = os.listdir(moved_path)
    root_filenames_base = []
    moved_filenames_base = []
    for filename in tqdm(root_filenames):
        filename_base = read_name(filename)
        root_filenames_base.append(filename_base)
    for filename in tqdm(moved_filenames):
        filename_base = read_name(filename)
        moved_filenames_base.append(filename_base)
    for filename in tqdm(root_filenames_base):

        if filename in moved_filenames_base:
            shutil.move(os.path.join(root_path, filename + str('.jpg')), os.path.join(moved_path))


root_path = 'data/crackdataset2/test/images'
moved_path = 'data/crackdataset2/test/image_buyao'
move_files(root_path, moved_path)
