from transformers import FlaxCLIPModel

from .utils import PretrainedFromWandbMixin


class FlaxCLIPModel(PretrainedFromWandbMixin, FlaxCLIPModel):
    pass
