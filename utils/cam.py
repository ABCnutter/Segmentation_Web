##################################################################################################################################################################
 
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from PIL import Image
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import numpy as np 
from pprint import pprint
import json
from model.set_model import set_model

def read_img(image_file, transformer):
    img = Image.open(image_file)
    image_np = np.array(img)
    augmented = transformer(image=image_np)
    vir_image = torch.randn(size=(3, augmented['image'].shape[1], augmented['image'].shape[2]), dtype=torch.float32).to(device=device)
    augmented_img = augmented['image'].to(device)  # 小批量化 ：torch.Size([1, 3, 475, 475])
    augmented_img = torch.stack([augmented_img, vir_image], dim=0)
    print(augmented_img.shape)
    return augmented_img


img_mask_size = 448
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# ******************************************************************************** data ********************************************************************************

image_file = r"test\images"
image_name = "CRACK61"

image_file_path = os.path.join(image_file, image_name + (".jpg"))
transformer = A.Compose([
    A.Resize(img_mask_size, img_mask_size),
    A.Normalize(
        mean=(0.4923, 0.4962, 0.4987),
        std=(0.1652, 0.1633, 0.1606),
        # mean=(0.5835, 0.5820, 0.5841),
        # std=(0.1149, 0.1111, 0.1064),
        max_pixel_value=255.0
    ),
    ToTensorV2()
])
augmented_img = read_img(image_file_path, transformer)

# ******************************************************************************** model ********************************************************************************

test_json_filename = "mitb3_upernet_01_2023_6_26"
test_json_path = "args/" + test_json_filename + ".json"

with open(test_json_path, "r") as file:
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

state_dict = torch.load('checkpoint/mitb3_upernet_01_2023_6_26_train_best_f1.pth')
model.load_state_dict(state_dict['model'])
model.to(device)
print(f"权重训练epoch是：{state_dict['epoch']}")
print('网络设置完毕 ：成功载入了训练完毕的权重。')
pprint([key for key, value in list(model.named_modules())])




img = Image.open(image_file_path)
output = model(augmented_img)
pprint(output.shape)

output = torch.nn.functional.sigmoid(output).cpu()
sem_classes = [
    'crack',
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["crack"] 
car_mask = torch.where(output[0] > 0.3, torch.ones_like(output[0]), torch.zeros_like(output[0]))
print(car_mask.shape)
car_mask = car_mask.detach().cpu().squeeze(0).numpy()
print(car_mask.shape)
# car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category + 1)
car_mask_float = np.float32(car_mask == car_category)


both_images = np.hstack((img, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
img_mask = Image.fromarray(both_images)
img_mask.show()

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


target_layers = [model.encoder.norm4]
# target_layers = [model.decoder.conv_seg]



def reshape_transform(in_tensor):
    result = in_tensor.reshape(in_tensor.size(0),
        int(np.sqrt(in_tensor.size(1))), int(np.sqrt(in_tensor.size(1))), in_tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available(),
             reshape_transform=reshape_transform
             ) as cam:
    grayscale_cam = cam(input_tensor=augmented_img[0].unsqueeze(0), targets=targets)[0, :]
    cam_image = show_cam_on_image(np.float32(img) /255, grayscale_cam, use_rgb=True)

vir_image = Image.fromarray(cam_image)
vir_image.show()
vir_image.save(f"cam/{image_name}_cam_norm44.png")