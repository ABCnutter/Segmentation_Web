import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from model.seg_upernet import Seg_UperNet
from model.seg_pspnet import Seg_PSPNet
from model.seg_deeplabv3plus import Seg_DeepLabV3Plus
from model.seg_fcn import Seg_FCN


def set_model(task_mode, framework, backbone, encoder_predicted, use_deep_supervision, attention_name, num_classes,
              activation):
    if task_mode != "Object Recognition":
        raise ValueError("Currently, only 'Object Recognition' is supported (single class)")

    if framework == "fcn":
        return Seg_FCN(encoder_name=backbone,
                       encoder_predicted=encoder_predicted,
                       attention=attention_name,
                       num_classes=num_classes,
                       activation=activation,
                       use_deep_supervision=use_deep_supervision,
                       )
    elif framework == "upernet":
        return Seg_UperNet(encoder_name=backbone,
                           encoder_predicted=encoder_predicted,
                           decoder_out_channels=128,
                           attention=attention_name,
                           num_classes=num_classes,
                           activation=activation,
                           use_deep_supervision=use_deep_supervision,
                           )
    elif framework == "pspnet":
        return Seg_PSPNet(encoder_name=backbone,
                          encoder_predicted=encoder_predicted,
                          decoder_out_channels=128,
                          attention=attention_name,
                          num_classes=num_classes,
                          activation=activation,
                          use_deep_supervision=use_deep_supervision,
                          )
    elif framework == "deeplabv3plus":
        return Seg_DeepLabV3Plus(encoder_name=backbone,
                                 encoder_predicted=encoder_predicted,
                                 encoder_output_stride=16,
                                 decoder_channels=256,
                                 decoder_atrous_rates=(12, 24, 36),
                                 attention=attention_name,
                                 num_classes=num_classes,
                                 activation=activation,
                                 use_deep_supervision=use_deep_supervision,
                                 )
    else:
        raise ValueError("Please select the model framework in [upernet, pspnet, deeplabv3plus, fcn]!")
