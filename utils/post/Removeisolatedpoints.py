import cv2 as cv
import numpy as np
from tqdm import tqdm
import os


def Img1(src):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  # 创建个全0的黑背景
    for i in tqdm(range(1, num_labels)):
        mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > 150:  # 300是面积 可以随便调
            img[mask] = 1
            img[mask] = 1
            img[mask] = 1  # 面积大于300的区域涂白留下，小于300的涂0抹去
        else:
            img[mask] = 0
            img[mask] = 0
            img[mask] = 0
    return img


# def Img2(img):
#     contours, hierarch = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     area = []
#     for i in range(len(contours)):
#         area.append(cv.contourArea(contours[i]))  # 计算轮廓所占面积
#         if area[i] < 300:  # 轮廓面积，可以自己随便调
#             cv.drawContours(img, [contours[i]], 0, 0, -1)  # 该轮廓区域填0
#             continue
#     return imgr


src = cv.imread(r'E:\vs_code\pytorch\work\crack_seg_station\data\test_masks\results\89.png', 0)
unique, count = np.unique(src, return_counts=True)
data_count = dict(zip(unique, count))
print(data_count)
# cv.imshow('input', src)
# cv.waitKey(0)
srcr = Img1(src)
unique, count = np.unique(srcr, return_counts=True)
data_count = dict(zip(unique, count))
print(data_count)
src_path = r'E:\vs_code\pytorch\work\crack_seg_station\data\test_masks\results\removeis'
if not os.path.exists(src_path):
    os.makedirs(src_path)
cv.imwrite(os.path.join(src_path, '89remove150.png'), srcr, [cv.IMWRITE_PNG_COMPRESSION, 0])
cv.imshow('output', src)
cv.waitKey()