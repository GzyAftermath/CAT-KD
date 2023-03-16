import os
from .resnet import (
    resnet32_test,
    resnet8,
    resnet14,
    resnet20,
    resnet20_test,
    resnet32,
    resnet44,
    resnet56,
    resnet56_test,
    resnet110,
    resnet110_test,
    resnet8x4,
    resnet8x4_test,
    resnet32x4,
    resnet32x4_test,
)
from .resnetv2 import ResNet50, ResNet18, ResNet50_test
from .wrn import wrn_16_1, wrn_16_2, wrn_16_2_test, wrn_40_1, wrn_40_1_test, wrn_40_2, wrn_40_2_test
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg13_bn_test, vgg11_bn, vgg8_bn, vgg8_bn_test
from .mobilenetv2 import mobile_half, mobile_half_test
from .ShuffleNetv1 import ShuffleV1, ShuffleV1_test
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_test
cifar100_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/cifar_teachers/"
)
cifar_model_dict = {
    
    # teachers
    "resnet32x4_test": (
        resnet32x4_test,
        None
    ),
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet56_test": (
        resnet56_test,
        None
    ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
    "resnet110_test": (
        resnet110_test,
        None
    ),
    "resnet32x4": (
        resnet32x4,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50": (
        ResNet50,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50_test": (
        ResNet50_test,
        None
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    ),
    "wrn_40_2_test": (
        wrn_40_2_test,
        None
    ),
    "vgg13": (vgg13_bn, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.pth"),
    "vgg13_test": (vgg13_bn_test,None),

    # students
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet20_test": (resnet20_test, None),
    "resnet32": (resnet32, None),
    "resnet32_test": (resnet32_test, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "resnet8x4_test": (resnet8x4_test, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_16_2_test": (wrn_16_2_test, None),
    "wrn_40_1": (wrn_40_1, None),
    "wrn_40_1_test": (wrn_40_1_test, None),
    "vgg8": (vgg8_bn, None),
    "vgg8_test": (vgg8_bn_test, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "MobileNetV2": (mobile_half, None),
    "MobileNetV2_test": (mobile_half_test, None),
    "ShuffleV1": (ShuffleV1, None),
    "ShuffleV1_test": (ShuffleV1_test, None),
    "ShuffleV2": (ShuffleV2, None),
    "ShuffleV2_test": (ShuffleV2_test, None),
}
