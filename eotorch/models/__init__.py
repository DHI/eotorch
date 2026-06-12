from .deepresunet import ClfDeepResUNet, RegDeepResUNet
from .dinov3_upernet import DINOv3UPerNet

CLF_MODEL_MAPPING = {}

REG_MODEL_MAPPING = {
    "deepresunet": RegDeepResUNet,
}

SEG_MODEL_MAPPING = {
    "deepresunet": ClfDeepResUNet,
    "dinov3_upernet": DINOv3UPerNet,
}
