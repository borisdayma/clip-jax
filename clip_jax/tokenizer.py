from transformers import AutoTokenizer

from .wandb_utils import WandbMixin


class AutoTokenizer(WandbMixin, AutoTokenizer):
    pass
