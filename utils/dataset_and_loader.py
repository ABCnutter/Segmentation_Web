import os
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def preprocess_mask(mask):
    mask[(mask >= 0.0) & (mask <= 10.0)] = 0
    mask[(mask > 10.0) & (mask <= 255.0)] = 1
    mask = mask.astype(np.int32)
    return mask

def make_full_path(root_list, root_path):
    file_full_path_list = []
    for filename in root_list:
        file_full_path = os.path.join(root_path, filename)
        file_full_path_list.append(file_full_path)

    return file_full_path_list


def read_filepath(root):
    train_image_path = os.path.join(root, 'train','images')
    train_mask_path = os.path.join(root, 'train','masks')
    val_image_path = os.path.join(root, 'val','images')
    val_mask_path = os.path.join(root, 'val','masks')
    test_image_path = os.path.join(root, 'test','images')
    test_mask_path = os.path.join(root, 'test','masks')

    train_image_list = os.listdir(train_image_path)
    train_mask_list = os.listdir(train_mask_path)
    val_image_list = os.listdir(val_image_path)
    val_mask_list = os.listdir(val_mask_path)
    test_image_list = os.listdir(test_image_path)
    test_mask_list = os.listdir(test_mask_path)

    train_image_full_path_list = make_full_path(train_image_list, train_image_path)
    train_mask_full_path_list = make_full_path(train_mask_list, train_mask_path)
    val_image_full_path_list = make_full_path(val_image_list, val_image_path)
    val_mask_full_path_list = make_full_path(val_mask_list, val_mask_path)
    test_image_full_path_list = make_full_path(test_image_list, test_image_path)
    test_mask_full_path_list = make_full_path(test_mask_list, test_mask_path)

    return train_image_full_path_list, train_mask_full_path_list, val_image_full_path_list, \
           val_mask_full_path_list, test_image_full_path_list, test_mask_full_path_list


class CrackDataset(Dataset):

    def __init__(self, images_list: List, masks_list: List, phase: str = 'train', transforms=None):
        self.images_list = images_list
        self.masks_list = masks_list
        self.phase = phase
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images_list[index]
        mask_path = self.masks_list[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, -1)
        mask = preprocess_mask(mask)

        if self.transforms is not None:
            image_augmented = self.transforms[self.phase](image=image, mask=mask)

            image = image_augmented['image']
            mask = image_augmented['mask']

        return image, mask

    def __len__(self):
        return len(self.images_list)


def get_dataset(root, transforms):
    train_image_path_list, train_mask_path_list, val_image_path_list, val_mask_path_list, test_image_path_list, test_mask_path_list = read_filepath(
        root)
    train_dataset = CrackDataset(train_image_path_list,
                                 train_mask_path_list,
                                 phase="train",
                                 transforms=transforms)

    val_dataset = CrackDataset(val_image_path_list,
                               val_mask_path_list,
                               phase="val",
                               transforms=transforms)

    test_dataset = CrackDataset(test_image_path_list,
                                test_mask_path_list,
                                phase="test",
                                transforms=transforms)

    dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    return dataset_dict


def get_dataloader(dataset_dict, batch_size, num_workers, is_distributed, shuffle_val: bool = False):
    train_sampler = DistributedSampler(dataset_dict["train"]) if is_distributed else None
    train_loader_kwargs = {
        "dataset": dataset_dict["train"],
        "batch_size": batch_size,
        "shuffle": (train_sampler is None),
        "sampler": train_sampler,
        "num_workers": num_workers,
    }

    if is_distributed:
        train_loader_kwargs.pop("shuffle")

    train_dataloader = DataLoader(**train_loader_kwargs)

    val_loader_kwargs = {
        "dataset": dataset_dict["val"],
        "batch_size": batch_size,
        "shuffle": shuffle_val,
        "num_workers": num_workers,
    }
    val_dataloader = DataLoader(**val_loader_kwargs)

    test_loader_kwargs = {
        "dataset": dataset_dict["test"],
        "batch_size": batch_size,
        "shuffle": shuffle_val,
        "num_workers": num_workers,
    }
    test_dataloader = DataLoader(**test_loader_kwargs)

    dataloader_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    return dataloader_dict, train_sampler


# def visualize(image, mask, original_image=None, original_mask=None):
#     fontsize = 18

#     if original_image is None and original_mask is None:
#         f, ax = plt.subplots(1, 2, figsize=(8, 8))

#         ax[0].imshow(image)
#         ax[0].set_title('Transformed image', fontsize=fontsize)
#         ax[1].imshow(mask)
#         ax[1].set_title('Transformed image', fontsize=fontsize)
#     else:
#         f, ax = plt.subplots(2, 2, figsize=(8, 8))

#         ax[0, 0].imshow(original_image)
#         ax[0, 0].set_title('Original image', fontsize=fontsize)

#         ax[1, 0].imshow(original_mask)
#         ax[1, 0].set_title('Original mask', fontsize=fontsize)

#         ax[0, 1].imshow(image)
#         ax[0, 1].set_title('Transformed image', fontsize=fontsize)

#         ax[1, 1].imshow(mask)
#         ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

#     plt.show()


# def visualize_train_dataset(i: int, dataset: Dataset):
#     img, mask = dataset[i]
#     img_per = img.permute(1, 2, 0)

#     img_org = cv2.imread(train_image_full_path_list[i])
#     mask_org = cv2.imread(train_mask_full_path_list[i])
#     visualize(img_per, mask, img_org, mask_org)


# def visualize_val_dataset(i: int, dataset: Dataset):
#     img, mask = dataset[i]
#     img_per = img.permute(1, 2, 0)

#     img_org = cv2.imread(val_image_full_path_list[i])
#     mask_org = cv2.imread(val_mask_full_path_list[i])
#     visualize(img_per, mask, img_org, mask_org)


# if __name__ == '__main__':
    # root = r'C:\ChenPeng\SWork\seg_dev\crackdataset'
    # train_image_full_path_list, train_mask_full_path_list, val_image_full_path_list, val_mask_full_path_list, \
    # test_image_full_path_list, test_mask_full_path_list = read_filepath(
    #     root)

    # transforms = {
    #     'train': A.Compose([
    #         # A.Resize(224, 224),
    #         # A.HorizontalFlip(p=0.8),
    #         # A.VerticalFlip(p=0.8),
    #         # A.RandomRotate90(p=0.8),
    #         # # A.OneOf([
    #         # #     A.RandomGamma(p=0.6),
    #         # #     A.CoarseDropout(p=0.6),
    #         # # ], p=0.6),
    #         # A.Normalize(
    #         #     mean=(0.4741, 0.4937, 0.5048),
    #         #     std=(0.1621, 0.1532, 0.1523),
    #         #     max_pixel_value=255.0
    #         # ),
    #         ToTensorV2()
    #     ]),
    #     'val': A.Compose([
    #         # # A.Resize(224, 224),
    #         # A.Normalize(
    #         #     mean=(0.4741, 0.4937, 0.5048),
    #         #     std=(0.1621, 0.1532, 0.1523),
    #         #     max_pixel_value=255.0
    #         # ),
    #         ToTensorV2()
    #     ]),
    #     'test': A.Compose([
    #         # A.Resize(224, 224),
    #         # A.Normalize(
    #         #     mean=(0.4741, 0.4937, 0.5048),
    #         #     std=(0.1621, 0.1532, 0.1523),
    #         #     max_pixel_value=255.0
    #         # ),
    #         ToTensorV2()
    #     ]),
    # }

    ###isprs
    # ----------------
    # tensor([0.5835, 0.5820, 0.5841])
    # ----------------
    # tensor([0.1149, 0.1111, 0.1064])

    # dataset_dict = get_dataset(root=root, transforms=transforms)
    #
    # dataloader_dict, _ = get_dataloader(dataset_dict=dataset_dict, batch_size=32, num_workers=0, is_distributed=False,
    #                                     shuffle_val=False)

    # train_dataset = CrackDataset(train_image_full_path_list, train_mask_full_path_list, phase='train',
    #                              transforms=transformer)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # val_dataset = CrackDataset(val_image_full_path_list, val_mask_full_path_list, phase='val',
    #                            transforms=transformer)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

    # test_dataset = CrackDataset(test_image_full_path_list, test_mask_full_path_list, phase='test',
    #                             transforms=transformer)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # # mean, std = getStat(train_dataset)

    # visualize_train_dataset(2228, dataset_dict['train'])

    # visualize_val_dataset(156, val_dataset)

    # imgs, labels = next(iter(dataloader_dict['train']))
    # print(imgs.shape)
    # print(labels.shape)
    # imgs = imgs[0].permute(1, 2, 0).numpy()
    # labels = labels[0].unsqueeze(0).permute(1, 2, 0).numpy()
    # f, ax = plt.subplots(1, 2, figsize=(8, 8))
    #
    # ax[0].imshow(imgs)
    # ax[0].set_title('Transformed image')
    # ax[1].imshow(labels)
    # ax[1].set_title('Transformed image')
    # plt.show()
    # unique, count = torch.unique(labels, return_counts=True)
    # data_count = dict(zip(unique, count))
    # # print(data_count)
    # import torch
    # from tqdm import tqdm
    #
    # print('开始计算')
    # imgs = torch.stack([img_t for img_t, _ in tqdm(dataset_dict['train'])], dim=3)
    # print('shape of imgs: {}'.format(imgs.shape))
    # mean = imgs.view(3, -1).to(torch.float32).mean(dim=1)
    # mean = mean / 255
    # print("--------" * 2)
    # print(mean)
    # std = imgs.view(3, -1).to(torch.float32).std(dim=1)
    # std = std / 255
    # print("--------" * 2)
    # print(std)

# test_dataset_804
# ----------------
# tensor([0.0379, 0.0931, 0.1393])
# ----------------
# tensor([0.8039, 0.8127, 0.8176])
