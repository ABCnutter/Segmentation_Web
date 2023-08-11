import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


def main(img, save_img, size, repetition, end: str, channels: int):
    img = cv.imread(img, -1)  # 加载原图
    save_img = save_img  # 存储路径
    if not os.path.exists(save_img):
        os.makedirs(save_img)
    print(f'原图像(h, w, ch):', img.shape)
    size = size  # 裁剪大小: 500x500
    repetition = repetition  # 重复像素
    h, w = img.shape[0], img.shape[1]  # img.shape[0]是高, img.shape[1]是宽, img.shape[2]是通道数
    num_h = (h - repetition) // (size - repetition)  # 裁剪后行数
    num_w = (w - repetition) // (size - repetition)  # 裁剪后列数
    img = np.array(img)  # to array
    print(f'不考虑右、下情况下，每行生成{num_w}幅小图')
    print(f'不考虑右、下情况下，每列生成{num_h}幅小图')

    if end == str('.jpg'):
        flag = [cv.IMWRITE_JPEG_QUALITY, 100]
    elif end == str('.png'):
        flag = [cv.IMWRITE_PNG_COMPRESSION, 0]
    else:
        raise ValueError(" end must be 'jpg' or 'png' ! ")

    # 1.正常范围裁剪:
    if channels == 3:
        shape = (size, size, channels)
    elif channels == 1:
        shape = (size, size)
    else:
        raise ValueError(" channel must be '1' or '3' ! ")

    img_crop = np.zeros(shape)
    image = []
    i, j, k = 0, 0, 0
    for i in range(0, num_h):
        for j in range(0, num_w):
            img_crop = img[
                       i * size - i * repetition:i * size - i * repetition + size,
                       j * size - j * repetition:j * size - j * repetition + size]
            image.append(img_crop)

    for k in range(0, (num_h * num_w)):
        image_k = image[k]
        path_image_k = save_img + str(k + 1) + end
        cv.imwrite(path_image_k, image_k, flag)
    print(f'正常裁剪{num_h * num_w}幅图')

    # 2.最下面一行的裁剪:
    img_crop_down = np.zeros(shape)
    image_down = []
    i, j, k = 0, 0, 0
    for j in range(0, num_w):
        img_crop_down = img[
                        h - size:h,
                        j * size - j * repetition:j * size - j * repetition + size]
        image_down.append(img_crop_down)

    for k in range(0, num_w):
        image_k = image_down[k]
        path_image_k = save_img + str(k + 1 + num_h * num_w) + end
        cv.imwrite(path_image_k, image_k, flag)
    print(f'最下面一行裁剪{num_w}幅图')

    # 2.最右边一列的裁剪:
    img_crop_right = np.zeros(shape)
    image_right = []
    i, j, k = 0, 0, 0
    for i in range(0, num_h):
        img_crop_right = img[
                         i * size - i * repetition:i * size - i * repetition + size,
                         w - size:w]
        image_right.append(img_crop_right)

    for k in range(0, num_h):
        image_k = image_right[k]
        path_image_k = save_img + str(k + 1 + num_h * num_w + num_w) + end
        cv.imwrite(path_image_k, image_k, flag)
    print(f'最右边一列裁剪{num_h}幅图')

    # 3.最右下角的一幅小图:
    image_d_r = []
    i, j, k = 0, 0, 0
    img_crop_d_r = img[
                   h - size:h,
                   w - size:w]
    image_d_r.append(img_crop_right)
    path_image = save_img + str(k + 1 + num_h * num_w + num_w + num_h) + end
    cv.imwrite(path_image, img_crop_d_r, flag)
    print(f'最右下角裁剪1幅图')


path_label = 'test_data\label'
path_label_divie = 'test_data\label_d2'

path_img = 'test_data\img'
path_img_divie = 'test_data\img_d2'

path_test = 'test_data\\test\imgs'
path_test_divie = 'test_data\\test256'

def clip(path_img, path_img_divie, size, repetition, end, channels):
    for dirpath, dirnames, filenames in os.walk(path_img):
        print(f'dirpath: {dirpath}')
        print(f'dirnames: {dirnames}')
        print(f'filenames: {filenames}')

        for filename in tqdm(filenames):
            if filename.endswith("jpg") or filename.endswith("JPG") or filename.endswith("png"):
                path = os.path.join(dirpath, filename)
                # print(path3)
                path2 = os.path.splitext(filename)[0] + '\\'
                path3 = os.path.join(path_img_divie, path2)
                # print(path)
                # print(path3)
                main(path,  # 选择要裁剪的图像
                     path3,  # 裁剪后小图的存储路径
                     size=size,  # 裁剪正方形的大小
                     repetition=repetition,  # 像素重合距离
                     end=end,
                     channels=channels
                     )


clip(path_test, path_test_divie, size=256, repetition=50, end=str('.jpg'), channels=3)
