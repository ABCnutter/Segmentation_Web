import os
import logging
from osgeo import gdal
import numpy as np


#  读取tif数据集
def readTif(image_path):
    dataset = gdal.Open(image_path)
    if dataset == None:
        print(image_path + "文件无法打开")
    # if image_path[-4:] != ".tif" or image_path[-4:] != ".TIF":
    #     raise TypeError(f"The type of input image must be in TIF format, but is in {image_path[-4:]} format")
    
    # dataset = gdal.Open(image_path)

    # if dataset is None:
    #     raise  FileNotFoundError("Unable to open the image for the path you entered!")

    # projection = dataset.GetProjectionRef()
    # geotransform = dataset.GetGeoTransform()

    # if projection is None or geotransform is None:
    #     raise AttributeError("The image file does not have a coordinate system or projection!")
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


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate, logger:logging.Logger):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    logger.info(f"width:{width}")
    logger.info(f"height:{height}")
    logger.info(f"proj:{proj}")
    logger.info(f"geotrans:{geotrans}")
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    num_h = (height - CropSize * RepetitionRate) // (CropSize * (1 - RepetitionRate))
    num_w = (width - CropSize * RepetitionRate) // (CropSize * (1 - RepetitionRate))
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath))
    #  裁剪图片,重复率为RepetitionRate
    logger.info("-------------------==================== Start Croping ======================---------------------")

    for i in range(num_h):
        for j in range(num_w):
            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  写图像
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
            #  文件名 + 1
            new_name = new_name + 1
    logger.info(f"---------------- Normal range is complete. A total of {num_h * num_w} small block images！----------------")

    #  向前裁剪最后一列
    for i in range(num_h):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        #  写图像
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1
    logger.info(f"---------------- Rightmost column is complete. A total of {num_h} small block images！----------------")

    #  向前裁剪最后一行
    for j in range(num_w):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        #  文件名 + 1
        new_name = new_name + 1
    logger.info(f"---------------- Bottom line is complete. A total of {num_w} small block images！----------------")

    #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    logger.info(f"---------------- Bottom right corner is complete. A total of {1} small block images！----------------")

    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    new_name = new_name + 1

    logger.info(f"---------------- Crop complete! the output file is at {SavePath} ----------------")

    return width, height, proj, geotrans