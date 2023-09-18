import json
import json
import logging
import os

import albumentations as A
import cv2
import cv2 as cv
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from typing import Tuple

from model.set_model import set_model


class DeployDataset(Dataset):
    def __init__(self, root: str, is_use_transforms: bool, resize: Tuple[int] = (256, 256)):
        self.images_list = self._make_file_path_list(root)
        self.resize = resize
        self.transforms = self._set_transformer(is_use_transforms)

    def __getitem__(self, index):
        image_path = self.images_list[index]

        image = cv.imread(image_path, -1)
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transforms is not None:
            image_augmented = self.transforms(image=image)
            image = image_augmented['image']

        return image, image_name

    def __len__(self):
        return len(self.images_list)

    def _make_full_path(self, root_list, root_path):
        file_full_path_list = []
        for filename in root_list:
            file_full_path = os.path.join(root_path, filename)
            file_full_path_list.append(file_full_path)

        return file_full_path_list

    def _make_file_path_list(self, image_root):
        if not os.path.exists(image_root):
            raise FileNotFoundError(
                f"dataset of cliped image save path:[{image_root}] does not exist!"
            )
        from natsort import natsorted

        image_list = natsorted(os.listdir(image_root))
        image_list = [img for img in image_list if img.endswith(".jpg")]

        image_full_path_list = self._make_full_path(image_list, image_root)

        return image_full_path_list

    def _set_transformer(self, is_use=False):
        if is_use:
            transformer = A.Compose(
                [
                    # A.Resize(self.resize[0], self.resize[1]),
                    A.Normalize(
                        mean=(0.4014, 0.4500, 0.2985),
                        std=(0.2129, 0.2129, 0.1983),
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            transformer = None

        return transformer


def set_dataloader(
        root,
        use_transforms: bool = True,
        resize: int = 608,
        batch_size: int = 32,
        num_workers: int = 0,
):
    dataset = DeployDataset(root=root, is_use_transforms=use_transforms, resize=resize)

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return dataloader


def infer_process(
        model,
        dataloader,
        pred_save_path,
        instance_name,
        local_rank,
        seg_threshold,
        pixel_threshold,
):
    pred_save_path = os.path.join(pred_save_path, instance_name)
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    print(f"model outputs save dir: [{pred_save_path}]!")

    model.to(local_rank)

    model.eval()
    print('------------------' * 3)
    print('(start deploying)')

    for imgs, imgsname in tqdm(dataloader):
        # logger.info(f'Processing item {batch_index}')
        # 执行一些操作
        imgs = imgs.to(local_rank)

        outputs = model(imgs)

        probs = (
            outputs[0].to(torch.float32)
            if model.use_deep_supervision
            else outputs.to(torch.float32)
        )

        outs = torch.sigmoid(probs)
        outs = torch.where(
            outs > seg_threshold, torch.ones_like(outs), torch.zeros_like(outs)
        )

        outs = outs.cpu().detach().squeeze(1).numpy()

        for out_index, out in enumerate(outs):
            _, count = np.unique(out, return_counts=True)
            if count[-1] <= pixel_threshold:
                out = np.zeros((outs.shape[-2], outs.shape[-1]))

            save_path = os.path.join(
                pred_save_path,
                str(imgsname[out_index]) + ".png",
            )
            cv2.imwrite(save_path, np.multiply(out, 255).astype(np.int32))
            # out = Image.fromarray(np.multiply(out, 1).astype(np.int8))
            # out.save(save_path, "PNG", encoding="utf-8")
    return


def set_infermodel(
        test_json_filename,
        checkpoint_path,
):
    # test_json_path = "args/" + test_json_filename

    with open(test_json_filename, "r") as file:
        data = json.load(file)
        model = set_model(task_mode=data["task_mode"],
                          framework=data["model_framework"],
                          backbone=data["backbone_name"],
                          encoder_predicted=False,
                          use_deep_supervision=data["use_deep_supervision"],
                          deep_supervision_use_separable_conv=data["deep_supervision_use_separable_conv"],
                          use_separable_conv=data["use_separable_conv"],
                          attention_name=data["attention"],
                          num_classes=data["num_classes"],
                          activation=data["activation"],
                          )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path: {checkpoint_path} does not exist!")

    model.load_state_dict(state_dict=torch.load(checkpoint_path)['model'])
    logging.warning(f"Model: [{model.name}] weights loaded!")
    print(f"权重训练epoch是：{torch.load(checkpoint_path)['epoch']}")
    print('网络设置完毕 ：成功载入了训练完毕的权重。')

    return model


def infer_fn(
        root_org,
        use_transforms,
        size,
        batch_size,
        num_workers,
        test_json_filename,
        checkpoint_path,
        pred_save_path,
        seg_threshold=0.5,
        pixel_threshold=0
):
    dataloader = set_dataloader(
        root=root_org,
        use_transforms=use_transforms,
        resize=size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = set_infermodel(test_json_filename,
                           checkpoint_path)
    instance_name = os.path.basename(checkpoint_path).split('.')[0] + str(120)
    infer_process(
        model=model,
        pred_save_path=pred_save_path,
        instance_name=instance_name,
        dataloader=dataloader,
        local_rank=device,
        seg_threshold=seg_threshold,
        pixel_threshold=pixel_threshold,
    )


if __name__ == "__main__":
    args = dict(
        root_org=r"test\images",
        use_transforms=True,
        size=(227, 227),
        batch_size=1,
        num_workers=0,
        test_json_filename=r"crackqt_mitb3_upernet_01_2023_7_10.json",
        checkpoint_path=r"crackqt_mitb3_upernet_01_2023_7_10_val_best_iou.pth",
        pred_save_path=r"qiangti",
        seg_threshold=0.5,
        pixel_threshold=0,
    )
    infer_fn(**args)
