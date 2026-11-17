from .deepresunet import DeepResUNet
from .dinov3_upernet import DINOv3UPerNet
from .unet import UNet

# Models implemented in eotorch. Any other model name is created with
# segmentation_models_pytorch.create_model (e.g. "unet", "unetplusplus", "segformer").
MODEL_MAPPING = {
    "deepresunet": DeepResUNet,
    "unet_lite": UNet,
    "dinov3_upernet": DINOv3UPerNet,
}
