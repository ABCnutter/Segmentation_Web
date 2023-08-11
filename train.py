"""

Returns:
    _type_: _description_
"""
import os
import signal
import sys
import time
import atexit

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
# sys.path.append(os.path.join("model/modules"))
sys.path.append(os.path.join("utils"))

sys.path.append(os.path.join("model", "modules"))
import logging
import argparse
import random
import json
import torch
import logging  
import glo
import numpy as np
import albumentations as A
import torch.nn as nn
import torch.distributed as dist
from pprint import pprint
from threading import Thread, Timer
from albumentations.pytorch import ToTensorV2
from torch.nn.parallel import DistributedDataParallel as DDP
from model.set_model import set_model
from losses.set_loss_fn import set_loss_fn
from optimization.set_optimizer import set_optimizer
from schedulers.set_lr_scheduler import set_lr_scheduler
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from utils.dataset_and_loader import get_dataset, get_dataloader
from utils.train_eval import train_model
from utils.distributeddataparallel import init_process, destroy_process
from utils.train_resume_stop import TrainingStatus
from utils.set_logger import set_logger

app = Flask(__name__)
socketio = SocketIO(app)

glo._init()
glo.set_value(["work_ids"], set())

logger = None
 
def get_argparser():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root", type=str, default="crackdataset", help="path to Dataset"
    )
    parser.add_argument(
        "--dataset_mean",
        type=lambda value: tuple(map(float, value.split(","))),
        default=(0.474, 0.494, 0.505),
        help="the mean of the dataset by channel dimension",
    )
    parser.add_argument(
        "--dataset_var",
        type=lambda value: tuple(map(float, value.split(","))),
        default=(0.162, 0.153, 0.152),
        help="The var of the dataset by channel dimension",
    )

    # # image enhancement
    parser.add_argument(
        "--scaled_size",
        type=lambda value: tuple(map(int, value.split(","))),
        default=(256, 256),
        help="scaled size",
    )
    parser.add_argument("--random_horizontal_flip_probability", type=float, default=0.8)
    parser.add_argument("--random_vertical_flip_probability", type=float, default=0.8)
    parser.add_argument("--random_rotation_90_probability", type=float, default=0.8)
    parser.add_argument("--image_distortion_probability", type=float, default=0.8)

    # Model Options
    parser.add_argument(
        "--instance_name",
        type=str,
        default="resnet101_upernet_01_2023_5_22",
        help="the name of the running instance",
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
        default="mitb0",
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
        "--predicted_checkpoint_path",
        type=str,
        default=None,
        help="path of complete predicted weights for the model",
    )
    parser.add_argument(
        "--use_deep_supervision",
        type=bool,
        default=False,
        help="whether to use deep_supervision",
    )
    parser.add_argument(
        "--deep_supervision_weights",
        type=int,
        nargs="+",
        default=[1.0, 0.6, 0.4, 0.2],
        help="weights of deep supervision",
    )
    parser.add_argument(
        "--encoder_predicted",
        type=bool,
        default=False,
        help="whether to use pretraining weights.",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        choices=[None, "scse", "cbam", "ecanet", "sknet", "senet"],
        help="use attention mechanism",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="number of result categories for segmentation recognition",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="type of activation function used for output",
    )
    # Train Options

    # # #
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epoch")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size (default: 16)"
    )
    parser.add_argument(
        "--val_interval", type=int, default=1, help="verification interval"
    )
    parser.add_argument(
        "--use_benchmark", type=bool, default=False, help="Whether to use benchmark."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Specifies the number of worker processes for the data loader. It determines the number "
        "of CPU cores used when loading data.",
    )
    parser.add_argument(
        "--fix_random_seed", type=bool, default=False, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--random_seed", type=int, default=1024, help="random seed (default: 1024)"
    )

    # # optimizer
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="AdamW",
        choices=["SGD", "AdamW"],
        help="optimizer",
    )
    parser.add_argument(
        "--SGD_init_lr", type=float, default=1e-2, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--SGD_init_lr_scale_factor",
        type=float,
        default=2,
        help="learning rate scale factor (default: 0.01)",
    )
    parser.add_argument(
        "--SGD_weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--SGD_momentum", type=float, default=0.9, help="momentum (default: 1e-4)"
    )
    parser.add_argument(
        "--AdamW_init_lr",
        type=float,
        default=1e-3,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--AdamW_init_lr_scale_factor",
        type=float,
        default=2,
        help="learning rate scale factor (default: 0.01)",
    )
    parser.add_argument(
        "--AdamW_weight_decay",
        type=float,
        default=1e-2,
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--AdamW_betas",
        type=lambda value: tuple(map(float, value.split(","))),
        default=(0.9, 0.999),
        help="AdamW_betas",
    )
    parser.add_argument(
        "--AdamW_amsgrad", type=bool, default=False, help="AdamW_amsgrad"
    )
    parser.add_argument(
        "--AdamW_eps", type=float, default=1e-8, help="AdamW_eps (default: 1e-8)"
    )

    # # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["poly", "cosine"],
        help="learning rate scheduler policy",
    )
    parser.add_argument(
        "--t_initial",
        type=int,
        default=5,
        help="The initial number of epochs. Example, 50, 100 etc.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=2e-5,
        help="Defaults to 1e-5. The minimum learning rate to use during the scheduling. The learning rate does not ever go below this value.",
    )
    parser.add_argument(
        "--cycle_mul",
        type=float,
        default=1.2,
        help="Defaults to 1.0. Updates the SGDR schedule annealing.",
    )
    parser.add_argument(
        "--cycle_decay",
        type=float,
        default=0.8,
        help="When decay_rate > 0 and <1., at every restart the learning rate is decayed by new learning rate which equals lr * decay_rate.",
    )
    parser.add_argument(
        "--cycle_limit",
        type=int,
        default=50,
        help="The number of maximum restarts in SGDR.",
    )
    parser.add_argument(
        "--warmup_t", type=int, default=3, help="Defines the number of warmup epochs."
    )
    parser.add_argument(
        "--warmup_lr_init",
        type=float,
        default=1e-5,
        help="The initial learning rate during warmup.",
    )
    parser.add_argument(
        "--warmup_prefix",
        type=bool,
        default=True,
        help="Defaults to False. If set to True, then every new epoch number equals epoch = epoch - warmup_t.",
    )

    # # loss
    parser.add_argument(
        "--loss_function",
        type=str,
        default="ensembleloss",
        choices=[
            "ensembleloss",
            "diceloss",
            "focalloss",
            "bcewithlogitsloss",
            "crossentropyloss",
        ],
        help="loss function type",
    )
    parser.add_argument(
        "--loss_fn_mode",
        type=str,
        default="binary",
        choices=["binary", "multilabel", "multiclass"],
        help="Loss mode 'binary', 'multiclass' or 'multilabel'",
    )
    parser.add_argument(
        "--ensembleloss_weight",
        type=float,
        nargs="+",
        default=[0.6, 0.4, 0.2],
        help="weight of ensemble loss",
    )

    # # # dice
    parser.add_argument(
        "--loss_classes",
        type=int,
        nargs="+",
        default=None,
        help="List of classes that contribute in loss computation. By default, all channels are "
        "included.",
    )
    parser.add_argument(
        "--loss_log_loss",
        type=bool,
        default=False,
        help=" If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`",
    )
    parser.add_argument(
        "--loss_from_logits",
        type=bool,
        default=True,
        help="learning rate scheduler policy",
    )
    parser.add_argument(
        "--loss_dice_smooth",
        type=float,
        default=0.0,
        help="Smoothness constant for dice coefficient (a)",
    )
    parser.add_argument(
        "--loss_dice_ignore_index",
        type=int,
        default=None,
        help="Label that indicates ignored pixels (does not contribute to loss)",
    )
    parser.add_argument(
        "--loss_eps",
        type=float,
        default=1e-7,
        help="A small epsilon for numerical stability to avoid zero division error",
    )

    # # # focal
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.25,
        help="Prior probability of having positive value in target.",
    )
    parser.add_argument(
        "--loss_gamma",
        type=float,
        default=2.0,
        help="Power factor for dampening weight (focal strength).",
    )
    parser.add_argument(
        "--loss_focal_ignore_index",
        type=int,
        default=None,
        help="Label that indicates ignored pixels (does not contribute to loss)",
    )
    parser.add_argument(
        "--loss_normalized",
        type=bool,
        default=False,
        help="Compute normalized focal loss",
    )
    parser.add_argument(
        "--loss_reduced_threshold",
        type=float,
        default=None,
        help=" Switch to reduced focal loss.",
    )

    # # # bcewithlogits
    parser.add_argument(
        "--loss_weight",
        type=float,
        default=None,
        help=" a manual rescaling weight if provided it's repeated to match input tensor shape",
    )
    parser.add_argument(
        "--loss_pos_weight",
        type=float,
        default=None,
        help="a weight of positive examples.",
    )
    parser.add_argument(
        "--loss_bce_smooth_factor",
        type=float,
        default=None,
        help="Factor to smooth target",
    )
    parser.add_argument(
        "--loss_bce_ignore_index",
        type=int,
        default=None,
        help="Label that indicates ignored pixels (does not contribute to loss)",
    )
    parser.add_argument(
        "--loss_reduction",
        type=str,
        default="mean",
        help="Specifies the reduction to apply to the output:``'none'`` | ``'mean'`` | ``'sum'``.",
    )

    # # # crossentropyLoss
    parser.add_argument("--loss_dim", type=int, default=1)
    parser.add_argument(
        "--loss_ce_smooth_factor",
        type=float,
        default=0.0,
        help="Factor to smooth target",
    )
    parser.add_argument(
        "--loss_ce_ignore_index",
        type=int,
        default=None,
        help="Label that indicates ignored pixels (does not contribute to loss)",
    )

    # # # distributed
    """
        A Distributed Data Parallel (DDP) application can be executed on multiple nodes where each node can consist of multiple GPU devices. 
        Each node in turn can run multiple copies of the DDP application, each of which processes its models on multiple GPUs.

        Let N be the number of nodes on which the application is running and G be the number of GPUs per node. 
        The total number of application processes running across all the nodes at one time is called the World Size - W,  
        and the number of processes running on each node is referred to as the Local World Size - L.

        Each application process is assigned two IDs: a local rank in [0, L-1] and a global rank in [0, W-1].
        """
    parser.add_argument(
        "--use_distributed",
        type=bool,
        default=False,
        help="Whether to perform distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="The rank in the local application processes",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, metavar="N", help="number of node"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="number of gpus per node"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="The rank in the global application processes",
    )

    # # # amp
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=False,
        help="whether to use amp(automatic mixing accuracy training)",
    )
    # # # mertric
    parser.add_argument(
        "--metrics_measures",
        type=str,
        nargs="+",
        default=["IoU", "F1", "Pre", "Rec", "Acc"],
        help="Result accuracy evaluation index",
    )
    parser.add_argument(
        "--seg_threshold",
        type=float,
        default=0.5,
        help="segmentation threshold when dichotomizing outcomes by single channel output",
    )
    parser.add_argument(
        "--mertric_reduction",
        type=str,
        default="micro-imagewise",
        help="reduction (Optional[str]): Define how to aggregate metric between classes and images."
        "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
        "macro-imagesize, weighted-imagewise, none, None]",
    )
    parser.add_argument(
        "--metrics_mode",
        type=str,
        default="binary",
        choices=["binary", "multilabel", "multiclass"],
        help="Model output with following shapes and types depending on the specified ``mode``",
    )
    parser.add_argument(
        "--mertric_ignore_index",
        type=int,
        default=None,
        help="If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or "
        "`255``.`",
    )
    parser.add_argument(
        "--mertric_num_classes",
        type=int,
        default=None,
        help="Number of classes, necessary attribute only for ``'multiclass'`` mode. Class values "
        "should be in range 0..(num_classes - 1)",
    )

    # # # checkpoint save

    parser.add_argument(
        "--checkpoint_save_dir",
        type=str,
        default="checkpoint",
        help="Path to save model checkpoint dir.",
    )

    # # # log save
    parser.add_argument(
        "--tensorboard_log_save_dir",
        type=str,
        default="log/train",
        help="Save dir of tensorboard log file",
    )
    parser.add_argument(
        "--terminal_log_print_interval",
        type=int,
        default=10,
        help="print interval of terminal log (loss, acc, iou, f1, pre, recall)",
    )
    parser.add_argument(
        "--argpareser_save_dir", type=str, default="args", help="Save dir of argpareser"
    )

    parser.add_argument(
        '-d',
        '--workspacebasedir',
        type=str,
        default="workspace",
        help="base dir of workspace",
    )
    return parser


def load_json(argparser, json_path):
    if not os.path.exists(json_path):
        logging.error(f"{json_path} not exists!")
    with open(json_path, "r") as f:
        json_dict = json.load(f)

        argparser.instance_name = json_dict["instance_name"]

        # dataset setting
        data_relat_dir = json_dict["data_root"]
        argparser.data_root = os.path.join(argparser.workspacebasedir, data_relat_dir)
        argparser.dataset_mean = json_dict["dataset_mean"]
        argparser.dataset_var = json_dict["dataset_var"]
        argparser.scaled_size = json_dict["scaled_size"]
        argparser.random_horizontal_flip_probability = json_dict[
            "random_horizontal_flip_probability"
        ]
        argparser.random_vertical_flip_probability = json_dict[
            "random_vertical_flip_probability"
        ]
        argparser.random_rotation_90_probability = json_dict[
            "random_rotation_90_probability"
        ]
        argparser.image_distortion_probability = json_dict[
            "image_distortion_probability"
        ]

        # training setting
        argparser.num_epochs = json_dict["num_epochs"]  # epoch of train
        argparser.val_interval = json_dict["val_interval"]  # val interval
        argparser.num_gpus = json_dict["num_gpus"]  # num of gpus

        # model setting
        argparser.task_mode = json_dict["task_mode"]
        argparser.backbone_name = json_dict["backbone_name"]
        argparser.model_framework = json_dict["model_framework"]
        argparser.batch_size = json_dict["batch_size"]

        # optimizer setting
        argparser.optimizer = json_dict["optimizer"]
        argparser.SGD_init_lr = json_dict["init_lr"]
        argparser.AdamW_init_lr = json_dict["init_lr"]
        argparser.lr_schedulr = json_dict["lr_schedulr"]

        # loss setting
        argparser.loss_function = json_dict["loss_function"]

        # log setting
        argparser.terminal_log_print_interval = json_dict["log_print_iters_interval"]

        # measures of merics, fix ["IoU", "F1", "Pre", "Rec", "Acc"]!
        argparser.metrics_measures = json_dict["metrics_measures"]

        # use automatic mixing accuracy training
        argparser.use_amp = json_dict["use_amp"]




# def print_save_argpareser(argpareser, argpareser_save_dir):
#     if not os.path.exists(argpareser_save_dir):
#         os.mkdir(argpareser_save_dir)
#     argpareser_save_path = os.path.join(argpareser_save_dir, "train_args.json")
#     pprint("total paresers: \n")
#     pprint(vars(argpareser))

#     with open(argpareser_save_path, "w") as f:
#         f.write(json.dumps(vars(argpareser), indent=2))

# 保存状态变量到指定JSON文件
def auto_save_global_variables(workspace_path, key):
    auto_save_glo_interval = 1
    auto_save_glo_file_path = os.path.join(workspace_path, "glo_train.json")
    while True:
        # 执行保存函数
        glo.save_global_variables(auto_save_glo_file_path, key)

        # 等待一段时间后继续循环
        time.sleep(auto_save_glo_interval)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train(
    workspace_path,
    work_id,
    model,
    dataset_dict,
    loss_fn,
    optimizer,
    scheduler,
    local_rank,
    args,
    logger: logging.Logger,
):
    tensorboard_save_dir = os.path.join(workspace_path, "log", "tensorboard")
    # logging_save_dir = os.path.join(workspace_path, "log", "logging")
    checkpoint_save_dir = os.path.join(workspace_path, "checkpoint")

    return train_model(
        model=model,
        dataset_dict=dataset_dict,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        local_rank=local_rank,
        work_id=work_id,
        # instance_name=args.instance_name,
        use_deep_supervision=args.use_deep_supervision,
        deep_supervision_weights=args.deep_supervision_weights,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        num_workers=args.num_workers,
        seg_threshold=args.seg_threshold,
        use_distributed=args.use_distributed,
        use_amp=args.use_amp,
        metrics_measures=args.metrics_measures,
        mertric_mode=args.metrics_mode,
        mertric_reduction=args.mertric_reduction,
        mertric_ignore_index=args.mertric_ignore_index,
        mertric_num_classes=args.mertric_num_classes,
        predicted_checkpoint_path=args.predicted_checkpoint_path,
        checkpoint_save_dir=checkpoint_save_dir,
        tensorboard_save_dir=tensorboard_save_dir,
        # logging_save_dir=logging_save_dir,
        logger=logger,
        terminal_log_print_interval=args.terminal_log_print_interval,
    )


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


@app.route("/train", methods=["GET", "POST"])
def start_training():
    data = request.get_json()

    # 从 JSON 数据中提取工作空间路径
    workspace_rela_path = data.get('path')
    if workspace_rela_path is None:
        return jsonify("Workspace path is not provided.")
    # 从 JSON 数据中提取工作空间编号
    work_id = data.get('id')

    if work_id is None:
        return jsonify("work_id is not provided")

    if work_id in glo.get_value(["work_ids"]):
        return jsonify("Workspace is already in training.")

    glo.set_value(["work_ids"], work_id)
    print('-----------------------------------------')
    print("work space ids include: {}".format(glo.get_value(['work_ids'])))
    print('-----------------------------------------')

    glo.set_value([work_id, 'work_id'], work_id)

    args = get_argparser().parse_args()

    workspace_path = os.path.join(args.workspacebasedir, workspace_rela_path)

    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
        jsonify({'message': 'Workspace not exists, but is created automatically'})

    logging_save_dir = os.path.join(workspace_path, "log", "logging")

    global logger

    logger = set_logger(logging.DEBUG, logging_save_dir, work_id)

    glo.set_value([work_id, 'workspace_path'], workspace_path)
    glo.set_value([work_id, 'train_state'], TrainingStatus.NOT_STARTED)

    save_thread = Thread(
        target=auto_save_global_variables, args=(workspace_path, work_id,)
    )
    save_thread.daemon = True  # 设置为守护线程
    save_thread.start()
    
    # 启动异步训练任务
    training_thread = Thread(target=train, args=(workspace_path, work_id, logger,), name=work_id)
    training_thread.start()
    # training_thread.join()

    glo.set_value([work_id, 'train_state'], TrainingStatus.TRINGING)
    glo.set_value([work_id, 'stop_flag'], False)
    glo.set_value([work_id, 'resume_flag'], False)
    glo.set_value([work_id, 'thread'], training_thread.getName())

    # 启动异步训练任务
    # training_thread)
    # socketio.emit('message', 'Training started')

    return jsonify({'message': 'Training started'})


def train(workspace_path: str, work_id: str, logger: logging.Logger):
    try:
        args = get_argparser().parse_args()

        json_path = os.path.join(workspace_path, "config", "train", "demo.json")

        load_json(args, json_path)

        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True

        if args.fix_random_seed:
            set_seed(args.random_seed)

        if args.use_distributed:
            # 按照pytorch官网推荐，我们使单进程单GPU的分布式训练策略
            init_process(
                rank=args.local_rank,
                world_size=args.num_nodes * args.num_gpus,
                backend=args.backend,
            )

            # 获取可用的GPU数量
            num_gpus = torch.cuda.device_count()
            print(f"num_gpus:{num_gpus}")

            # 获取当前进程的编号rank ,获取的rank值的最大值是 world_size-1
            global_rank = dist.get_rank()
            print(f"global_rank:{global_rank}")

            # 根据进程编号选择对应的GPU设备
            local_rank = torch.device(f"cuda:{global_rank % num_gpus}")

            torch.cuda.set_device(local_rank)
        else:
            local_rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('\n')
        logger.info(
            f"============ The current device is {local_rank} ============="
        )

        if glo.get_value([work_id, 'train_state']) == TrainingStatus.TRINGING:
            logger.info(f"total paresers: {vars(args)}")

        # argpareser_save_dir = os.path.join(workspace_path, "log", "args")
        # if not os.path.exists(argpareser_save_dir):
        #     os.makedirs(argpareser_save_dir)
        # if glo.get_value([work_id, 'train_state']) == TrainingStatus.TRINGING:
        #     print_save_argpareser(args, argpareser_save_dir)

        transforms = {
            "train": A.Compose(
                [
                    A.Resize(args.scaled_size[0], args.scaled_size[1]),
                    A.HorizontalFlip(p=args.random_horizontal_flip_probability),
                    A.VerticalFlip(p=args.random_vertical_flip_probability),
                    A.RandomRotate90(p=args.random_rotation_90_probability),
                    A.OneOf(
                        [
                            A.RandomGamma(p=0.8),
                            A.CoarseDropout(p=0.8),
                        ],
                        p=args.image_distortion_probability,
                    ),
                    A.Normalize(
                        mean=args.dataset_mean, std=args.dataset_var, max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Resize(args.scaled_size[0], args.scaled_size[1]),
                    A.HorizontalFlip(p=args.random_horizontal_flip_probability),
                    A.VerticalFlip(p=args.random_vertical_flip_probability),
                    A.RandomRotate90(p=args.random_rotation_90_probability),
                    A.OneOf(
                        [
                            A.RandomGamma(p=0.9),
                            A.CoarseDropout(p=0.9),
                        ],
                        p=args.image_distortion_probability,
                    ),
                    A.Normalize(
                        mean=args.dataset_mean, std=args.dataset_var, max_pixel_value=255.0
                    ),
                    ToTensorV2(),
                ]
            ),
        }

        dataset_dict = get_dataset(root=args.data_root, transforms=transforms)

        model = set_model(
            task_mode=args.task_mode,
            framework=args.model_framework,
            backbone=args.backbone_name,
            encoder_predicted=args.encoder_predicted,
            use_deep_supervision=args.use_deep_supervision,
            attention_name=args.attention,
            num_classes=args.num_classes,
            activation=args.activation,
        )
        loss_fn = set_loss_fn(
            loss_fn_name=args.loss_function,
            mode=args.loss_fn_mode,
            #################################
            loss_classes=args.loss_classes,
            loss_log_loss=args.loss_log_loss,
            loss_from_logits=args.loss_from_logits,
            loss_dice_smooth=args.loss_dice_smooth,
            loss_dice_ignore_index=args.loss_dice_ignore_index,
            loss_eps=args.loss_eps,
            #################################
            loss_alpha=args.loss_alpha,
            loss_gamma=args.loss_gamma,
            loss_focal_ignore_index=args.loss_focal_ignore_index,
            loss_normalized=args.loss_normalized,
            loss_reduced_threshold=args.loss_reduced_threshold,
            #################################
            loss_weight=torch.tensor(args.loss_weight)
            if args.loss_weight is not None
            else None,
            loss_pos_weight=torch.tensor(args.loss_pos_weight)
            if args.loss_pos_weight is not None
            else None,
            loss_bce_smooth_factor=args.loss_bce_smooth_factor,
            loss_bce_ignore_index=args.loss_bce_ignore_index,
            loss_reduction=args.loss_reduction,
            #################################
            loss_ce_ignore_index=args.loss_ce_ignore_index,
            loss_ce_smooth_factor=args.loss_ce_smooth_factor,
            loss_dim=args.loss_dim,
            #################################
            ensembleloss_weight=args.ensembleloss_weight,
        )

        optimizer = set_optimizer(
            optimizer_name=args.optimizer_name,
            model=model,
            SGD_init_lr=args.SGD_init_lr,
            SGD_init_lr_scale_factor=args.SGD_init_lr_scale_factor,
            SGD_weight_decay=args.SGD_weight_decay,
            SGD_momentum=args.SGD_momentum,
            AdamW_init_lr=args.AdamW_init_lr,
            AdamW_init_lr_scale_factor=args.AdamW_init_lr_scale_factor,
            AdamW_betas=args.AdamW_betas,
            AdamW_eps=args.AdamW_eps,
            AdamW_weight_decay=args.AdamW_weight_decay,
            AdamW_amsgrad=args.AdamW_amsgrad,
        )

        lr_schedulr = set_lr_scheduler(
            scheduler_name=args.lr_scheduler,
            optimizer=optimizer,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_mul=args.cycle_mul,
            cycle_decay=args.cycle_decay,
            cycle_limit=args.cycle_limit,
            warmup_t=args.warmup_t,
            warmup_lr_init=args.warmup_lr_init,
            warmup_prefix=args.warmup_prefix,
        )

        build_train(
            workspace_path=workspace_path,
            model=model,
            dataset_dict=dataset_dict,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=lr_schedulr,
            local_rank=local_rank,
            args=args,
            work_id=work_id,
            logger=logger,
        )

        if args.use_distributed:
            destroy_process()

        if glo.get_value([work_id, 'train_state']) == TrainingStatus.STOPPED:
            # socketio.emit('message', 'Training stoped')
            return 
        
        elif glo.get_value([work_id, 'train_state']) == TrainingStatus.FINISHED:
            glo.remove_setvalue('work_ids', work_id)
            glo.delete_key(work_id)
            logging.info(f"train process :{work_id} finished, relevant global information cleared, but there is a backup [glo_train.json] in the worksapce")
            # socketio.emit('message', 'Training finished')
            return 
    
    except Exception as e:
        # 捕获训练异常，记录错误信息
        error_message = str(e)
        logger.error("Training Error:", error_message)
        # 设置模型训练状态为异常失败
        glo.set_value([work_id, "train_state"], TrainingStatus.FAILED)
        # 返回错误信息给前端
        return 500
    

    


@app.route("/train/stop", methods=["POST"])
def stop_training():
    data = request.get_json()

    # 从 JSON 数据中提取工作空间路径
    work_id = data.get('id')

    if work_id not in glo.get_value(['work_ids']):
        return jsonify({'message': 'Invalid work id'})
    
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.NOT_STARTED:
        return jsonify({'message': 'Training is not started'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.STOPPED:
        return jsonify({'message': 'Training already stoped'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.FINISHED:
        return jsonify({'message': 'Training already finished'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.FAILED:
        return jsonify({'message': 'Training already failed'})
    
    glo.set_value([work_id, "stop_flag"], True)

    return jsonify({'message': 'Training stoped'})


@app.route("/train/resume", methods=["POST"])
def resume_training():
    data = request.get_json()

    # 从 JSON 数据中提取工作空间路径
    work_id = data.get('id')

    if work_id not in glo.get_value(['work_ids']):
        return jsonify({'message': 'Invalid work id'})
    
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.NOT_STARTED:
        return jsonify({'message': 'Training is not started'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.TRINGING:
        return jsonify({'message': 'Training in progress'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.FINISHED:
        return jsonify({'message': 'Training already finished'})
    if glo.get_value([work_id, 'train_state']) == TrainingStatus.FAILED:
        return jsonify({'message': 'Training already failed'})
    
    glo.set_value([work_id, "resume_flag"], True)

    workspace_path = glo.get_value([work_id, "workspace_path"])

    training_thread = Thread(target=train, args=(workspace_path, work_id, logger,), name=work_id + 'resume')
    glo.set_value([work_id, 'thread'], training_thread.getName())

    training_thread.start()

    # training_thread.join()
    return jsonify({'message': 'Training resumed'})


@app.route("/train/state", methods=["GET", "POST"])
def get_train_state():
    data = request.get_json()
    
    # 从 JSON 数据中提取工作空间路径
    workspace_rela_path = data.get('path')
    args = get_argparser().parse_args()
    workspace_path = os.path.join(args.workspacebasedir, workspace_rela_path)
    glo_json_path = os.path.join(workspace_path, "glo_train.json")

    if not os.path.exists(glo_json_path):
        jsonify(f"state file {glo_json_path} not found")
        raise FileNotFoundError(f"{glo_json_path} not found")

    with open(glo_json_path, "r") as f:
        glo_json_dict = json.load(f)
        
    if glo_json_dict['train_state'] == TrainingStatus.NOT_STARTED.value:
        return jsonify({'message': "Training has not started!", 'code': 1})
    
    elif glo_json_dict['train_state'] == TrainingStatus.STOPPED.value:
        return jsonify({'message': "Training has been terminated normally, please resume training!", 'code': 3})
    
    elif glo_json_dict['train_state'] == TrainingStatus.FINISHED.value:
        return jsonify({'message': "Training has been finished!", 'code': 4})
    
    elif glo_json_dict['train_state'] == TrainingStatus.FAILED.value:
        return jsonify({'message': "Training has been failed, please check the relevant error reports and retrain!", 'code': 5})
    
    elif glo_json_dict['train_state'] == TrainingStatus.TRINGING.value and glo_json_dict['work_id'] in glo.get_value(['work_ids']):
        return jsonify({'message': "Training in progress!", 'code': 2})
    
    elif glo_json_dict['train_state'] == TrainingStatus.TRINGING.value and glo_json_dict['work_id'] not in glo.get_value(['work_ids']):

        glo.set_value(["work_ids"], glo_json_dict['work_id'])
        glo.set_value([glo_json_dict['work_id'], "work_id"], glo_json_dict['work_id'])
        glo.set_value([glo_json_dict['work_id'], "workspace_path"], glo_json_dict['workspace_path'])
        glo.set_value([glo_json_dict['work_id'], "train_state"], TrainingStatus.STOPPED.value)
        glo.set_value([glo_json_dict['work_id'], 'stop_flag'], False)
        glo.set_value([glo_json_dict['work_id'], 'resume_flag'], False)

        logging_save_dir = os.path.join(glo_json_dict['workspace_path'], "log", "logging")
        global logger
        logger = set_logger(logging.DEBUG, logging_save_dir, glo_json_dict['work_id'])

        save_thread = Thread(
            target=auto_save_global_variables, args=(workspace_path, glo_json_dict['work_id'])
        )
        save_thread.daemon = True  # 设置为守护线程
        save_thread.start()

        print(f'state:{glo.get_value([glo_json_dict["work_id"], "train_state"])}')
        return jsonify({'message': "Training has been terminated abnormally, please resume training!", 'code': 3})
    else:
        return jsonify({'message': "train state exist error!", 'code': 500})


# 启动Flask应用程序
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)