import argparse
import json
import os
import time

import requests
import glo
import shutil
import torch
import logging
import cv2 as cv
import numpy as np
import albumentations as A
import sys

sys.path.append(os.path.join("utils"))

sys.path.append(os.path.join("model", "modules"))
from pprint import pprint
from osgeo import gdal
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from threading import Thread
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from flask import Flask, jsonify, request
from utils.process.clipgdal import TifCrop, writeTiff, readTif
from utils.post.stitchgdal import stitchTiff
from model.set_model import set_model
from utils.set_logger import set_logger

app = Flask(__name__)

glo._init()

glo.set_value(["infer_ids"], set())

logger = None


class InferingStatus(Enum):
    NOT_STARTED = 1
    INFERING = 2
    FINISHED = 3
    FAILED = 4


# 保存状态变量到指定JSON文件
def auto_save_global_variables(workspace_path, key):
    auto_save_glo_interval = 1
    auto_save_glo_file_path = os.path.join(workspace_path, "glo_infer.json")
    while True:
        # 执行保存函数
        glo.save_global_variables(auto_save_glo_file_path, key)

        # 等待一段时间后继续循环
        time.sleep(auto_save_glo_interval)


def download_url_file(url, save_path, logger: logging.Logger):
    if not (url.endswith('.tif') or url.endswith('.TIF')):
        raise ValueError(f"url extension must be .tif or.TIF, but got {url[-4:]}")

    # 创建tqdm进度条
    progress_bar = tqdm(total=100, ncols=80)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_size = int(response.headers.get("Content-Length", 0))
        progress_bar.total = file_size

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)
                progress_bar.update(len(chunk))
                logger.info(
                    'Downloading: {:.2f}%'.format(
                        progress_bar.n / progress_bar.total * 100
                    )
                )

        progress_bar.close()
        logger.info('File downloaded successfully.')
    else:
        logger.error(
            'Failed to download file. Status code: {}'.format(response.status_code)
        )


def check_img(image_path):
    if not (image_path.endswith(".tif", -4) or image_path.endswith(".TIF", -4)):
        raise TypeError(f"The type of input image must be in TIF format")

    dataset = gdal.Open(image_path)

    if dataset is None:
        raise FileNotFoundError("Unable to open the image for the path you entered!")

    projection = dataset.GetProjectionRef()
    geotransform = dataset.GetGeoTransform()

    if projection is None or geotransform is None:
        raise AttributeError(
            "The image file does not have a coordinate system or projection!"
        )

    dataset = None


def delete_dir(dir):
    try:
        shutil.rmtree(dir)
        print(f"path:[{dir}] had been deleted")
    except FileNotFoundError:
        print(f"path: [{dir}] is not exist")
    except Exception as e:
        print(f"delete path: [{dir}] happen error: [{str(e)}]")


def croptif(imgpath, save_path, cropsize, logger: logging.Logger):
    check_img(imgpath)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.info(f"clip results save path: [{save_path}]!")

    assert isinstance(cropsize, int)

    width, height, proj, geotrans = TifCrop(imgpath, save_path, cropsize, 0, logger)

    return save_path, width, height, proj, geotrans


class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.pbar = None

    def write(self, msg):
        if self.pbar is None:
            self.logger.log(self.level, msg.rstrip())
        else:
            self.pbar.write(msg)

    def flush(self):
        pass


class DeployDataset(Dataset):
    def __init__(self, root: str, is_use_transforms: bool, resize: int = 256):
        self.images_list = self._make_file_path_list(root)
        self.resize = resize
        self.transforms = self._set_transformer(is_use_transforms)

    def __getitem__(self, index):
        image_path = self.images_list[index]

        image = cv.imread(image_path, -1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transforms is not None:
            image_augmented = self.transforms(image=image)
            image = image_augmented['image']

        return image

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
        image_list = [img for img in image_list if img.endswith(".tif")]

        image_full_path_list = self._make_full_path(image_list, image_root)

        return image_full_path_list

    def _set_transformer(self, is_use=False):
        if is_use:
            transformer = A.Compose(
                [
                    A.Resize(self.resize, self.resize),
                    A.Normalize(
                        mean=(0.474, 0.494, 0.505),
                        std=(0.162, 0.153, 0.152),
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
    resize: int = 256,
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
    local_rank,
    im_geotrans,
    im_proj,
    seg_threshold,
    pixel_threshold,
    logger: logging.Logger,
):
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    logger.info(f"model outputs save dir: [{pred_save_path}]!")

    batch_size = dataloader.batch_size

    model.to(local_rank)

    model.eval()
    logger.info('------------------' * 3)
    logger.info('(start deploying)')
    with tqdm(
        total=len(dataloader), ncols=100, colour='#C0FF20', file=TqdmToLogger(logger)
    ) as pbar:
        for batch_index, imgs in enumerate(dataloader):
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
                    str(batch_index * batch_size + out_index + 1) + ".tif",
                )
                writeTiff(out, im_geotrans=im_geotrans, im_proj=im_proj, path=save_path)
            pbar.update(1)

        return


def stitchtif(
    ori_img_path, croped_path, output_path, output_name, size, logger: logging.Logger
):
    if not os.path.exists(ori_img_path):
        raise FileNotFoundError(f"ori_img_path: {croped_path} does not exist!")

    if not os.path.exists(croped_path):
        raise FileNotFoundError(f"croped_path: {croped_path} does not exist!")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Infer results save dir: [{output_path}]!")

    output_name = output_name + ".tif"

    stitchTiff(
        ori_img_path,
        croped_path,
        output_path,
        output_name,
        size,
        repetition=0,
        logger=logger,
    )


def set_infermodel(
    task_mode="Object Recognition",
    framework='upernet',
    backbone='mitb3',
    encoder_predicted=None,
    use_deep_supervision=False,
    attention_name=None,
    num_classes=1,
    activation=None,
    checkpoint_path=r"checkpoint\mitb3_upernet_01_2023_5_16_val_best_iou.pth",
):
    model = set_model(
        task_mode=task_mode,
        framework=framework,
        backbone=backbone,
        encoder_predicted=encoder_predicted,
        use_deep_supervision=use_deep_supervision,
        attention_name=attention_name,
        num_classes=num_classes,
        activation=activation,
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path: {checkpoint_path} does not exist!")

    model.load_state_dict(state_dict=torch.load(checkpoint_path)['model'])
    logging.warning(f"Model: [{model.name}] weights loaded!")

    return model


def infer_fn(
    root_org,
    root_crop,
    root_pred,
    root_result,
    output_name,
    model,
    batch_size,
    num_workers,
    logger: logging.Logger,
    size=256,
):
    clip_save_path, _, _, proj, geotrans = croptif(
        root_org, root_crop, cropsize=size, logger=logger
    )

    dataloader = set_dataloader(
        root=clip_save_path,
        use_transforms=True,
        resize=size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infer_process(
        model=model,
        pred_save_path=root_pred,
        dataloader=dataloader,
        local_rank=device,
        im_geotrans=geotrans,
        im_proj=proj,
        seg_threshold=0.5,
        pixel_threshold=0,
        logger=logger,
    )

    stitchtif(
        ori_img_path=root_org,
        croped_path=root_pred,
        output_path=root_result,
        output_name=output_name,
        size=size,
        logger=logger,
    )


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infer_primal_image_path',
        type=str,
        default=r"crackdataset\deploy\rgb_org\test2.tif",
    )
    parser.add_argument('--output_name', type=str, default="test")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=r"checkpoint\mitb3_upernet_01_2023_5_16\mitb3_upernet_01_2023_5_16_val_best_iou.pth",
    )
    parser.add_argument("--task_mode", type=str, default="Object Recognition")
    parser.add_argument(
        "--model_framework",
        type=str,
        default="upernet",
        choices=["upernet", "pspnet", "deeplabv3plus", "fcn"],
        help="framework for segmentation recognition.",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="mitb3",
        choices=[
            "resnet50",
            "resnet101",
            "resnet152",
            "convnext_small",
            "convnext_base",
            "convnext_tiny",
            "mitb0",
            "mitb1",
            "mitb2",
            "mitb3",
            "hrnet_w18",
            "hrnet_w32",
            "hrnet_w48",
        ],
        help="backbone network type.",
    )
    parser.add_argument(
        '-d',
        '--workspacebasedir',
        type=str,
        default="workspace",
        help="base dir of workspace",
    )
    # parser.add_argument(
    #     "--predicted_checkpoint_path",
    #     type=str,
    #     default=None,
    #     help="path of complete weights for the model",
    # )
    # parser.add_argument(
    #     "--use_deep_supervision",
    #     type=bool,
    #     default=False,
    #     help="whether to use deep_supervision",
    # )
    # parser.add_argument(
    #     "--deep_supervision_weights",
    #     type=int,
    #     nargs="+",
    #     default=[1.0, 0.6, 0.4, 0.2],
    #     help="weights of deep supervision",
    # )
    # parser.add_argument(
    #     "--encoder_predicted",
    #     type=bool,
    #     default=False,
    #     help="whether to use pretraining weights.",
    # )
    # parser.add_argument(
    #     "--attention",
    #     type=str,
    #     default=None,
    #     choices=[None, "scse", "cbam", "ecanet", "sknet", "senet"],
    #     help="use attention mechanism",
    # )
    # parser.add_argument(
    #     "--num_classes",
    #     type=int,
    #     default=1,
    #     help="number of result categories for segmentation recognition",
    # )
    # parser.add_argument(
    #     "--activation",
    #     type=str,
    #     default=None,
    #     help="type of activation function used for output",
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="batch size for training",
    # )
    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=0,
    #     help="number of workers for dataloader",
    # )

    return parser


def load_infer_json(workspace_path, argparser, json_path, logger: logging.Logger):
    if not os.path.exists(json_path):
        logging.error(f"{json_path} not exists!")
    with open(json_path, "r") as f:
        json_dict = json.load(f)
        print("load args from json!")
        pprint("user paresers:\n")
        pprint(json_dict)
        # model setting

        if json_dict["infer_primal_image_path"].startswith("http://") or json_dict[
            "infer_primal_image_path"
        ].startswith("https://"):
            url2filepath = os.path.join(workspace_path, "infer", "org")
            if not os.path.exists(url2filepath):
                os.makedirs(url2filepath)
            urlimgsavepath = os.path.join(url2filepath, 'org.tif')

            download_url_file(
                json_dict["infer_primal_image_path"], urlimgsavepath, logger=logger
            )
            argparser.infer_primal_image_path = urlimgsavepath
        else:
            infer_primal_image_relat_path = json_dict["infer_primal_image_path"]
            argparser.infer_primal_image_path = os.path.join(argparser.workspacebasedir, infer_primal_image_relat_path)

            if not os.path.exists(argparser.infer_primal_image_path):
                raise FileNotFoundError(f"{argparser.infer_primal_image_path} not exists!")

        argparser.output_name = json_dict["output_name"]
        argparser.backbone_name = json_dict["backbone_name"]
        argparser.model_framework = json_dict["model_framework"]

        checkpoint_relat_path = json_dict["checkpoint"]
        argparser.checkpoint = os.path.join(argparser.workspacebasedir, checkpoint_relat_path)

        if not os.path.exists(argparser.checkpoint):
            raise FileNotFoundError(f"{argparser.checkpoint} not exists!")

        # argparser.infer_primal_image_path = json_dict["infer_primal_image_path"]
        # argparser.output_name

        # argparser.task_mode = json_dict["task_mode"]
        # argparser.num_classes = json_dict["num_classes"]
        # argparser.root_clip = json_dict["root_clip"]
        # argparser.root_pred = json_dict["root_pred"]
        # argparser.root_stitch = json_dict["root_stitch"]
        # argparser.encoder_predicted = json_dict["encoder_predicted"]
        # argparser.use_deep_supervision = json_dict["use_deep_supervision"]
        # argparser.deep_supervision_weights = json_dict["deep_supervision_weights"]
        # argparser.attention = json_dict["attention"]
        # argparser.activation = json_dict["activation"]
        # argparser.batch_size = json_dict["batch_size"]
        # argparser.num_workers = json_dict["num_workers"]


@app.route("/infer", methods=["GET", "POST"])
def start_infering():
    data = request.get_json()

    # 从 JSON 数据中提取工作空间路径
    workspace_rela_path = data.get('path')
    if workspace_rela_path is None:
        return jsonify("Workspace path is not provided.")
    # 从 JSON 数据中提取工作空间编号
    infer_id = data.get('id')

    if infer_id is None:
        return jsonify("infer_id is not provided")

    if infer_id in glo.get_value(["infer_ids"]):
        return jsonify("Workspace is already in infering.")

    glo.set_value(["infer_ids"], infer_id)
    print('-----------------------------------------')
    print("infer space ids include: {}".format(glo.get_value(['infer_ids'])))
    print('-----------------------------------------')

    glo.set_value([infer_id, 'infer_id'], infer_id)

    args = get_argparser().parse_args()

    workspace_path = os.path.join(args.workspacebasedir, workspace_rela_path)

    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
        jsonify({'message': 'Workspace not exists, but is created automatically'})
    logging_save_dir = os.path.join(workspace_path, "log", "logging")
    logger = set_logger(logging.DEBUG, logging_save_dir, infer_id)

    # raise FileNotFoundError(f"{workspace_path} not exists!")
    glo.set_value([infer_id, 'infer_state'], InferingStatus.NOT_STARTED)

    save_thread = Thread(
        target=auto_save_global_variables,
        args=(
            workspace_path,
            infer_id,
        ),
    )
    save_thread.daemon = True  # 设置为守护线程
    save_thread.start()

    # 启动异步训练任务
    infering_thread = Thread(
        target=infer,
        args=(
            workspace_path,
            infer_id,
            logger,
        ),
    )
    infering_thread.start()

    glo.set_value([infer_id, 'infer_state'], InferingStatus.INFERING)

    return jsonify({'message': 'infering started'})


def infer(workspace_path: str, infer_id: str, logger: logging.Logger):
    try:
        json_path = os.path.join(workspace_path, "config", "infer", "demo.json")

        if not os.path.exists(json_path):
            raise FileExistsError(f"{json_path} not exists!")
        args = get_argparser().parse_args()
        load_infer_json(workspace_path, args, json_path, logger=logger)

        model = set_infermodel(
            framework=args.model_framework,
            backbone=args.backbone_name,
            checkpoint_path=args.checkpoint,
            task_mode="Object Recognition",
            encoder_predicted=None,
            use_deep_supervision=False,
            attention_name=None,
            num_classes=1,
            activation=None,
        )

        INFER_CROP_SAVE_PATH = os.path.join(workspace_path, "infer", "crop")
        INFER_PRED_SAVE_PATH = os.path.join(workspace_path, "infer", "pred")
        INFER_RESULT_SAVE_PATH = os.path.join(workspace_path, "infer", "result")

        infer_fn(
            root_org=args.infer_primal_image_path,
            root_crop=INFER_CROP_SAVE_PATH,
            root_pred=INFER_PRED_SAVE_PATH,
            root_result=INFER_RESULT_SAVE_PATH,
            output_name=args.output_name,
            model=model,
            batch_size=4,
            num_workers=0,
            size=256,
            logger=logger,
        )

        delete_dir(INFER_CROP_SAVE_PATH)
        logger.warning(f"The cropped image is clear!")

        delete_dir(INFER_PRED_SAVE_PATH)
        logger.warning(f"The predicted small image has been clear!")

        INFER_URL_SAVE_PATH = os.path.join(workspace_path, "infer", "org")
        if os.path.exists(INFER_URL_SAVE_PATH):
            delete_dir(INFER_URL_SAVE_PATH)
            logger.warning(f"The url image has been clear!")

        glo.set_value([infer_id, 'infer_state'], InferingStatus.FINISHED)
        logging.info(f"infer work: {infer_id} has been finished!")
        
        return

    except Exception as e:
        # 捕获推理异常，记录错误信息
        error_message = str(e)
        logger.error(f"Infer Error:{error_message}")
        # 设置模型推理状态为异常失败
        glo.set_value([infer_id, "infer_state"], InferingStatus.FAILED)
        # 返回错误信息给前端
        return 500


@app.route("/infer/state", methods=["GET", "POST"])
def get_infer_state():
    data = request.get_json()
    # 从 JSON 数据中提取工作空间路径
    workspace_relapath = data.get('path')

    args = get_argparser().parse_args()

    workspace_path = os.path.join(args.workspacebasedir, workspace_relapath)
    if not os.path.exists(workspace_path):
        return jsonify({'message':f"state inquire error: {workspace_path} not found"})

    glo_json_path = os.path.join(workspace_path, "glo_infer.json")

    if not os.path.exists(glo_json_path):
        return jsonify({'message':f"state inquire error: {glo_json_path} not found. Infering maybe not started, please check!"})

    with open(glo_json_path, "r") as f:
        glo_json_dict = json.load(f)

    if glo_json_dict['infer_state'] == InferingStatus.NOT_STARTED.value:
        return jsonify({'message': "Infering has not started!", 'code': 1})

    elif glo_json_dict['infer_state'] == InferingStatus.FINISHED.value:
        return jsonify({'message': "Infering has been finished!", 'code': 3})

    elif glo_json_dict['infer_state'] == InferingStatus.FAILED.value:
        return jsonify(
            {
                'message': "Infering has been failed, please check the relevant error reports and reinfer!",
                'code': 4,
            }
        )

    elif glo_json_dict[
        'infer_state'
    ] == InferingStatus.INFERING.value and glo_json_dict['infer_id'] in glo.get_value(
        ['infer_ids']
    ):
        return jsonify({'message': "Infering in progress!", 'code': 2})

    elif glo_json_dict[
        'infer_state'
    ] == InferingStatus.INFERING.value and glo_json_dict[
        'infer_id'
    ] not in glo.get_value(
        ['infer_ids']
    ):
        INFER_URL_SAVE_PATH = os.path.join(workspace_path, "infer", "org")
        INFER_CROP_SAVE_PATH = os.path.join(workspace_path, "infer", "crop")
        INFER_PRED_SAVE_PATH = os.path.join(workspace_path, "infer", "pred")

        if os.path.exists(INFER_URL_SAVE_PATH):
            delete_dir(INFER_URL_SAVE_PATH)
        if os.path.exists(INFER_CROP_SAVE_PATH):
            delete_dir(INFER_CROP_SAVE_PATH)
        if os.path.exists(INFER_PRED_SAVE_PATH):
            delete_dir(INFER_PRED_SAVE_PATH)

        return jsonify(
            {
                'message': "Infering has been terminated abnormally, please reindef!",
                'code': 1,
            }
        )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
