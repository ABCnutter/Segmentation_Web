import os
import logging
from osgeo import gdal
import numpy as np


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def stitchTiff(ori_img_path, croped_path, output_path, output_name, size, repetition, logger:logging.Logger):
    ori_img = readTif(ori_img_path)

    croped_path = croped_path
    output_path = output_path
    output_name = output_name
    size = size
    repetition = repetition

    w = ori_img.RasterXSize
    h = ori_img.RasterYSize
    proj = ori_img.GetProjection()
    geotrans = ori_img.GetGeoTransform()
    num_h = (h - repetition) // (size - repetition)  # 裁剪后行数
    num_w = (w - repetition) // (size - repetition)  # 裁剪后列数
    img = np.zeros((h, w))  # 创建与原始图像等大的画布

    all_img = os.listdir(croped_path)  # ['1.tif', '10.tif', '100.tif', ...]
    all_img = [img for img in all_img if img.endswith(".tif")]
    all_img.sort(key=lambda x: int(x[:-4]))  # ['1.tif', '2.tif', '3.tif', ...]

    logger.info("----------------==============  Start Stitching ==============----------------")

    # 1.正常范围拼接
    i, j = 0, 0
    for i in range(0, num_h):
        for j in range(0, num_w):
            small_img_path = os.path.join(croped_path, all_img[i * num_w + j])
            # print(f'正常范围拼接:{all_img[i * num_w + j]}')
            small_img = readTif(small_img_path)
            small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
            small_img = np.array(small_img)
            img[
            i * (size - repetition):i * (size - repetition) + size,
            j * (size - repetition):j * (size - repetition) + size
            ] = small_img[0:size, 0:size]
    logger.info(f"---------------- Normal range is complete. A total of {num_w * num_h} small block images！----------------")

    # 2.最右边一列的拼接
    i, j = 0, 0
    for i in range(0, num_h):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + i])
        # print(f'最右边一列的拼接:{all_img[num_h * num_w + i]}')
        small_img = readTif(small_img_path)
        small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
        small_img = np.array(small_img)
        img[
        i * (size - repetition):i * (size - repetition) + size,
        w - size:w
        ] = small_img[0:size, 0:size]
    logger.info(f"---------------- Rightmost column is complete. A total of {num_h} small block images！----------------")

    # 3.最下面一行的拼接:
    i, j = 0, 0
    for j in range(0, num_w):
        small_img_path = os.path.join(croped_path, all_img[num_h * num_w + num_h + j])
        # print(f'最下面一行的拼接:{all_img[num_h * num_w + num_h + j]}')
        small_img = readTif(small_img_path)
        small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
        small_img = np.array(small_img)
        img[
        h - size:h,
        j * (size - repetition):j * (size - repetition) + size
        ] = small_img[0:size, 0:size]
    logger.info(f"---------------- Bottom line is complete. A total of {num_w} small block images！----------------")

    # 4.最右下角的一幅小图
    small_img_path = os.path.join(croped_path, all_img[-1])
    # print(f'最右下角的一幅小图拼接:{all_img[-1]}')
    small_img = readTif(small_img_path)
    small_img = small_img.ReadAsArray(0, 0, size, size)  # 获取数据
    small_img = np.array(small_img)
    img[
    h - size:h,
    w - size:w
    ] = small_img[0:size, 0:size]
    logger.info(f"---------------- Bottom right corner is complete. A total of {1} small block images！----------------")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writeTiff(img, geotrans, proj, os.path.join(output_path, output_name))

    logger.info(f"----------------============== Stitch complete! ==============----------------")

    logger.info(f"============== the output file is at: [{os.path.join(output_path, output_name)}] ==============")

if __name__ == '__main__':
    stitchTiff(ori_img_path=r'E:\CS\work\seg_dev_BEIFEN(1)\seg_dev_BEIFEN\crackdataset\test\rgb_org\test2.tif',  # 读取原图，用于创建画布
         croped_path=r'E:\CS\work\seg_dev_BEIFEN(1)\seg_dev_BEIFEN\crackdataset\test\rgb_result\\',  # 读取存放小图的路径
         output_path=r'E:\CS\work\seg_dev_BEIFEN(1)\seg_dev_BEIFEN\crackdataset\test\rgb_stitch',  # 设置结果输出路径
         output_name='test.tif',  # 设置拼接后的图像名
         size=256,  # 小图大小
         repetition=0  # 像素重合距离
         )