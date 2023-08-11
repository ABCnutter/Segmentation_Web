import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from model.set_model import set_model
from utils.dataset_and_loader import get_dataset, get_dataloader
from utils.test_results import test

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print("使用的设备为：{}".format(device))

root = r'crackdataset'

transforms = {
    'test': A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.8),
        A.RandomRotate90(p=0.8),
        A.OneOf([
            A.RandomGamma(p=0.8),
            A.CoarseDropout(p=0.8),
        ], p=0.8),
        A.Normalize(
            mean=(0.586, 0.582, 0.584),
            std=(0.115, 0.111, 0.106),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]),
}

batch_size = 4

dataset_dict = get_dataset(root=root, transforms=transforms)

dataloader_dict, _ = get_dataloader(dataset_dict, batch_size, num_workers=0, is_distributed=False, shuffle_val=False)
model = set_model(task_mode="Object Recognition",
                  framework='upernet',
                  backbone='mitb3',
                  encoder_predicted=None,
                  use_deep_supervision=False,
                  attention_name=None,
                  num_classes=1,
                  activation=None,
                  )

state_dict = torch.load('checkpoint/mitb3_upernet_01_2023_5_16/mitb3_upernet_01_2023_5_16_val_best_iou.pth')
model.load_state_dict(state_dict['model'])
print(f"权重训练epoch是：{state_dict['epoch']}")
print('网络设置完毕 ：成功载入了训练完毕的权重。')
instance_name = "mitb3_upernet_01_2023_5_16_val_best_iou_gcrack512_5_20_3"
tensorboard_test_log_save_dir = "log/test"

test(model=model,
           dataloader=dataloader_dict["test"],
           local_rank=device,
           instance_name=instance_name,
           test_log_save_dir=tensorboard_test_log_save_dir,
           metrics_measures=['IoU', 'F1', 'Pre', 'Rec', 'Acc'],
           mertric_mode="binary",
           mertric_reduction="micro-imagewise",
           seg_threshold=0.5,
           mertric_ignore_index=None,
           mertric_num_classes=None
           )
