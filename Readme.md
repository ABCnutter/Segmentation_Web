# Segmentation_Web：通用二维语义分割算法及Web接口端署


![屏幕截图 2023-08-11 220133](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/9533ab09-89ee-4c29-b0e6-8ab3824607e4)


本项目为二维图像语义分割通用项目后端算法框架，可用于自然图像、医疗影像、遥感影像的分割训练以及推理。本项目已基于Flask框架实现了Web端部署，已提供前端可调用的API接口，包括模型训练、推理、暂停恢复、日志打印保存、状态查询等功能，支持分布式训练、自动混合精度等。

![image](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/e8819445-8627-4b5b-a6a1-17ae263d95cb)


![image](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/e8f1bf39-ec09-4f55-bc74-25bc415162cb)





## 项目结构

### 模型

项目的模型主要为编码解码架构，自由度较高，支持多种主干网络和分割框架的交互搭配。目前主要支持的网络结构有：

- 主干网络

```yaml
'resnet18',
'resnet50',
'resnet101',
'resnet152',
'convnext_small',
'convnext_base',
'convnext_tiny',
'mitb0',
'mitb1',
'mitb2',
'mitb3',
'mitb4',
'mitb5',
'dmitplusb0',
'dmitplusb1',
'dmitplusb2',
'dmitplusb3',
'dmitplusb4',
'dmitplusb5',
'hrnet_w18',
'hrnet_w30',
'hrnet_w32',
'hrnet_w40',
'hrnet_w44',
'hrnet_w48',
'hrnet_w64',
'efficientnetv2_l',
'efficientnetv2_m',
'efficientnetv2_rw_m',
'efficientnetv2_rw_s',
'efficientnetv2_rw_t',
'efficientnetv2_s',
'efficientnetv2_xl',
'vgg11',
'vgg11_bn',
'vgg13',
'vgg13_bn',
'vgg16',
'vgg16_bn',
'vgg19',
'vgg19_bn',
'xception',
'xception41',
'xception41p',
'xception65',
'xception65p',
'xception71',
'mobilenetv3_large_075',
'mobilenetv3_large_100',
'mobilenetv3_large_100_miil',
'mobilenetv3_small_100',
'repvgg_a2',
'repvgg_b0',
'repvgg_b1',
'repvgg_b1g4',
'repvgg_b2',
'repvgg_b2g4',
'repvgg_b3',
'repvgg_b3g4',
```

- 分割框架

```yaml
'upernet',
'deeplapv3',
'pspnet',
'fcn',
```

- 注意力机制

```YAML
'SENet',
'SKNet',
'SCSE',
'ECANet',
'CBAM',
```

项目模型同时还支持**深监督**。

### 损失函数

```yaml
'bcewithlogitsloss'\'crossentropyloss',
'focalloss',
'diceloss',
'bounary',
'ensembleloss'——前三个loss的集成loss,
```

注：项目中还支持**tversky  、mcc、 jaccard、lovase** 损失函数，但没有集成到流程中，大家可以自行修改源码或者自行调用。

### lr调整策略

```yaml
'cosine',
'poly',
```

注：项目中还支持**multistep、plateau、 step、tanh** lr调整策略，但没有集成到流程中，大家可以自行修改源码或者自行调用。

### 评价指标

```yaml
'iou_score',
'f1_score',
'precision',
'recall',
'accuracy',
```

注：评价指标不支持选择，在训练时会一起给出。

### 优化器

```yaml
'SGD',
'AdamW',
```



## 使用说明

### 依赖安装

切换到项目根目录下（存在requirements.txt文件的路径下），在脚本环境中输入以下命令安装项目所需要的依赖。**注意：**项目依赖pytorch，请先根据pytorch官方指南进行安装，再运行以下命令。

```shell
pip install -r requirements.txt
```



### 参数设置

项目的数据存放、结果缓存等存放在工作空间内，工作空间的路径自己确定，需要在命令行启动项目时给出，若不给出，则默认为warkspace路径。工作空间的文件夹结构如下：

![image-20230811202146348](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/18a66ff7-e86a-4b69-8845-1d08a34ec260)


如上图所示，P001\T002为工作空间下的相对路径，它是单一任务给出的**分区工作空间**（分路径，此次任务的生成的文件信息都会保存在该分路径下）。项目的参数配置文件（json格式）也在该分路径下，**路径格式必须为：分路径/config/train/demo.json、分路径/config/infer/demo.json**。**大家可以在配置文件中修改项目训练及推理参数，对于一些在配置文件中找不到的参数，可以在train.py或infer.py中的命令行设置函数中寻找并修改**。

配置文件相关内容如下：

**train.json:**

```json
{   
    "instance_name": "resnet50_upernet_01_2023_6_1",

    "data_root": "dataset",

    "dataset_mean": [0.474, 0.493, 0.505],

    "dataset_var": [0.162, 0.153, 0.152],

    "scaled_size": [256, 256],
    
    "random_horizontal_flip_probability": 0.8, 
    
    "random_vertical_flip_probability": 0.8,
    
    "random_rotation_90_probability": 0.8,
    
    "image_distortion_probability": 0.8,

    "num_epochs": 20,

    "val_interval": 10,
    
    "num_classes": 1,
    
    "num_gpus": 1,

    "task_mode": "Object Recognition", 

    "backbone_name": "resnet50",

    "model_framework": "upernet", 

    "batch_size": 16,

    "optimizer": "AdamW",

    "init_lr": 1e-3,

    "lr_schedulr": "cosine",

    "loss_function": "ensembleloss",

    "log_print_iters_interval": 10,

    "metrics_measures": ["IoU", "F1", "Pre", "Rec", "Acc"],

    "use_amp": true
}
```

infer.json:

```json
{   
    "infer_primal_image_path": "crackdataset/deploy/rgb_org/test2.tif",

    "output_name": "test",

    "backbone_name": "mitb3",

    "model_framework": "upernet",

    "checkpoint": "checkpoint/best.pth"
}
```

因为项目已部署为Web端，所以需要使用postman等类似的API调试器进行启动。模型训练的具体操作流程如下：

- 命令行启动项目程序train.py

  ```shell
  python train.py -workspacebasedir # workspacebasedir未给出，则默认创建workspace路径
  ```

- 启动程序后出现，flask持续监听。

```shell
D:\anaconda3\envs\smp\lib\site-packages\scipy\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
WebSocket transport not available. Install simple-websocket for improved performance.
 * Serving Flask app "train" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.31.133:5000/ (Press CTRL+C to quit)
```


- 启动postman，将上一步程序启动生成的网址链接复制粘贴到postman的命令中，如http://192.168.31.133:5000/train。依次选择**POST， Body， raw， JSON** ，然后在下方空格输入如图内容。其中**path为上述的分路径，id为本次任务的唯一id**（可给出随意数字）。

![image-20230811203834557](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/49c4fdd9-abc6-4081-9d2e-2bd2922b615a)


推理流程与训练类似：

- 命令行启动项目程序infer.py

  ```shell
  python infer.py -workspacebasedir # workspacebasedir未给出，则默认创建workspace路径
  ```

- 启动程序后出现，flask持续监听。

```shell
D:\anaconda3\envs\smp\lib\site-packages\scipy\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
 * Serving Flask app "infer" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.31.133:5050/ (Press CTRL+C to quit)
```

- 启动postman，将上一步程序启动生成的网址链接复制粘贴到postman的命令中，如http://192.168.31.133:5000/infer。依次选择**POST， Body， raw， JSON** ，然后在下方空格输入如图内容。其中path为上述的分路径，id为本次任务的唯一id（可给出随意数字）。

![image-20230811204857986](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/34bbb3b2-817c-4f6e-b238-39cf7f65c2a1)


### 状态查询

程序启动之后，在训练和推理的过程中可进行状态查询。语句如下图所示。

训练支持以下几种状态：

```yaml
'not_started'，
'training'，
'stopped'，
'falied'，
'resumed'，
'finished'，
```

![image-20230811205455883](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/34e644ba-05f2-4c64-98b2-354843621e1d)


推理支持以下几种状态：

```yaml
'not_started'，
'infering'，
'falied'，
'finished'，
```

![image-20230811205507400](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/4ab27d23-2511-44ee-bd00-051c258593ab)


### 暂停恢复

训练的暂停与恢复,也是通过postman启动。**需给出任务id**。

![image-20230811205605976](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/29477412-57e1-4f28-8324-4382a02e95e6)


![image-20230811205653506](https://github.com/ABCnutter/Segmentation_Web/assets/91233657/b7044789-ddc0-4a68-b46c-f550091ddd85)


**注意**：

1. **对于异常退出，如关机、卡退等，重新启动程序，可首先查询训练状态，会提醒训练异常退出，直接通过resume API恢复模型的训练。**
2. **对于推理部分，因之前项目需求，目前仅支持带有空间位置信息的Tif图像格式，后续会对其进行更改，以便支持更多数据格式。**

