import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import torch

from model.encoder.function import TimmEncoder
from model.encoder.mit import _mit_extractor

encoders_name = [
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
    'hrnet_w18',
    'hrnet_w30',
    'hrnet_w32',
    'hrnet_w40',
    'hrnet_w44',
    'hrnet_w48',
    'hrnet_w64',
    'efficientnetv2_l',
    'efficientnetv2_m',
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
    'xception65',
    'xception71',
    'mobilenetv3_large_075',
    'mobilenetv3_large_100',
    'mobilenetv3_small_100',
]


def get_encoder(
    name: str = "resnet50",
    predicted: bool = False,
    multiple_features_return: bool = True,
    default_in_channels=3,
    default_depth=4,
    default_output_stride=32,
    **kwargs,
):
    """_summary_

    Args:
        name (str, optional): _description_. Defaults to "resnet50".
        predicted (bool, optional): _description_. Defaults to False.
        multiple_features_return (bool, optional): _description_. Defaults to True.
        default_in_channels (int, optional): _description_. Defaults to 3.
        default_depth (int, optional): _description_. Defaults to 4.
        default_output_stride (int, optional): _description_. Defaults to 32.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if name not in encoders_name:
        raise ValueError(f"Please select name in {encoders_name}!")

    if name.startswith("mit"):
        return _mit_extractor(
            name=name,
            predicted=predicted,
            multiple_features_return=multiple_features_return,
        )
    else:
        encoder = TimmEncoder(
            name=name,
            in_channels=default_in_channels,
            features_only=multiple_features_return,
            pretrained=predicted,
            depth=default_depth,
            output_stride=default_output_stride,
            **kwargs,
        )
        return encoder


if __name__ == "__main__":
    model = get_encoder(
        name="hrnet_w32", predicted=False, multiple_features_return=True
    )
    # print(model.get_outs_channels_nums())
    inputs = torch.randn(2, 3, 256, 256)
    outs = model(inputs)
    output_stride = model.output_stride
    print(output_stride)
    # print(model)
    print([o.shape for o in outs])
    # print(model.extract_model)
