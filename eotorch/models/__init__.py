from .deepresunet import DeepResUNet
from .dinov3_upernet import DINOv3UPerNet
from .unet import UNet

CLF_MODEL_MAPPING = {}

REG_MODEL_MAPPING = {
    "deepresunet": DeepResUNet,
    "unet_lite": UNet,
}

SEG_MODEL_MAPPING = {
    "deepresunet": DeepResUNet,
    "dinov3_upernet": DINOv3UPerNet,
    "unet_lite": UNet,
}
