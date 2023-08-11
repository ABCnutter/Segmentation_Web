import cv2 as cv
import numpy as np
import os
import shutil


# path_img_data = 'test_data/img_results/'
# path_label_data = 'test_data/label_results/'
# path = 'test_data/label_d'
# path2 = 'test_data/img_d\\'
# num_label_liefeng = 0
# save_label_num = 1
def sreach(
        path_img_data_crack='test_data/img_result2/',
        path_label_data_crack='test_data/label_result2/',
        path_img_data_nocrack='test_data/img_result2/',
        path_label_data_nocrack='test_data/label_result2/',
        path='test_data/label_d2',
        path2='test_data/img_d2\\',
        num_label_liefeng=0,
        save_label_num=1,
):
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            path_num_label = os.path.join(dirpath, dirname)
            # print(dirname)
            for dirpath2, dirnames2, filenames2 in os.walk(path_num_label):
                for filename2 in filenames2:
                    num_label = os.path.splitext(filename2)[0]
                    path_label = os.path.join(dirpath2, filename2)
                    # print(num_label)
                    # print(path_label)

                    label = cv.imread(path_label, -1)
                    # label_g = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
                    if np.any(label):
                        num_label_liefeng += 1
                        print('第{}张图像中的第{}张标签存在裂缝！！！！！！'.format(dirname, num_label))

                        path_img = path2 + str(dirname) + '\\' + str(num_label) + str('.jpg')

                        save_img_path = path_img_data_crack + 'Crack' + str(save_label_num) + "_" + str(dirname) + "_" + str(num_label) + str('.jpg')
                        save_label_path = path_label_data_crack + 'Crack' + str(save_label_num) + "_" + str(dirname) + "_" + str(num_label) + str('.png')

                        # print(path_label)
                        # print(path_img)

                        # 复制label到label_data文件夹
                        shutil.copy(path_label, save_label_path)
                        # 复制img到img_data文件夹
                        shutil.copy(path_img, save_img_path)

                        save_label_num += 1
                    else:
                        print('第{}张图像中的第{}张标签不存在裂缝'.format(dirname, num_label))

                        path_img = path2 + str(dirname) + '\\' + str(num_label) + str('.jpg')

                        save_img_path = path_img_data_crack + 'No_Crack' + str(save_label_num) + "_" + str(dirname) + "_" + str(num_label) + str('.jpg')
                        save_label_path = path_label_data_crack + 'No_Crack' + str(save_label_num) + "_" + str(dirname) + "_" + str(num_label) + str('.png')

                        # 复制label到label_data文件夹
                        shutil.copy(path_label, save_label_path)
                        # 复制img到img_data文件夹
                        shutil.copy(path_img, save_img_path)
    print('裂缝总数为：', num_label_liefeng)


sreach(path_img_data_crack='test_data/img_result2/',
       path_label_data_crack='test_data/label_result2/',
       path='test_data/label_d2',
       path2='test_data/img_d2\\',
       num_label_liefeng=0,
       save_label_num=1)
