import os
import numpy as np
import cv2 as cv


def main(ori_img, croped_path, output_path, output_name, size, repetition):
    ori_img = cv.imread(ori_img, -1)
    croped_path = croped_path
    output_path = output_path
    output_name = output_name
    size = size
    repetition = repetition
    h, w = ori_img.shape[0], ori_img.shape[1]  # 获取原始图像的高和宽
    num_h = (h - repetition) // (size - repetition)  # 裁剪后行数
    num_w = (w - repetition) // (size - repetition)  # 裁剪后列数
    img = np.zeros((h, w))  # 创建与原始图像等大的画布

    all_img = os.listdir(croped_path)  # ['1.jpg', '10.jpg', '100.jpg', ...]
    all_img.sort(key=lambda x: int(x[:-4]))  # ['1.jpg', '2.jpg', '3.jpg', ...]

    # 1.正常范围拼接
    i, j = 0, 0
    for i in range(0, num_h):
        for j in range(0, num_w):
            small_img_path = os.path.join(croped_path, all_img[i * num_w + j])
            print(f'正常范围拼接:{all_img[i * num_w + j]}')
            small_img = cv.imread(small_img_path, -1)
            small_img = np.array(small_img)
            img[
            i * (size - repetition):i * (size - repetition) + size,
            j * (size - repetition):j * (size - repetition) + size
            ] = small_img[0:size, 0:size]

    # 2.最下面一行的拼接:
    i, j = 0, 0
    for j in range(0, num_w):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + j])
        print(f'最下面一行的拼接:{all_img[num_h * num_w + j]}')
        small_img = cv.imread(small_img_path, -1)
        small_img = np.array(small_img)
        img[
        h - size:h,
        j * (size - repetition):j * (size - repetition) + size
        ] = small_img[0:size, 0:size]

    # 3.最右边一列的拼接
    i, j = 0, 0
    for i in range(0, num_h):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + num_w + i])
        print(f'最右边一列的拼接:{all_img[num_h * num_w + num_w + i]}')
        small_img = cv.imread(small_img_path, -1)
        small_img = np.array(small_img)
        img[
        i * (size - repetition):i * (size - repetition) + size,
        w - size:w
        ] = small_img[0:size, 0:size]

    # 4.最右下角的一幅小图
    small_img_path = os.path.join(croped_path, all_img[-1])
    print(f'最右下角的一幅小图拼接:{all_img[-1]}')
    small_img = cv.imread(small_img_path, -1)
    small_img = np.array(small_img)
    img[
    h - size:h,
    w - size:w
    ] = small_img[0:size, 0:size]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(os.path.join(output_path, output_name), img, [cv.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main(ori_img=r'E:\vs_code\pytorch\work\crack_seg_station\data\80.JPG',  # 读取原图，用于创建画布
         croped_path=r'E:\vs_code\pytorch\work\crack_seg_station\data\test_masks\89_remove225\\',  # 读取存放小图的路径
         output_path=r'E:\vs_code\pytorch\work\crack_seg_station\data\test_masks\results',  # 设置结果输出路径
         output_name='89_remove225.png',  # 设置拼接后的图像名
         size=256,  # 小图大小
         repetition=50  # 像素重合距离
         )