from .deepresunet import ClfDeepResUNet, RegDeepResUNet

CLF_MODEL_MAPPING = {
    "deepresunet": ClfDeepResUNet,
}

REG_MODEL_MAPPING = {
    "deepresunet": RegDeepResUNet,
}
