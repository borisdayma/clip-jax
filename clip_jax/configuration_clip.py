# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" CLIP model configuration"""

import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .utils import PretrainedFromWandbMixin

logger = logging.get_logger(__name__)


class CLIPTextConfig(PretrainedConfig):
    model_type = "clip_jax_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        use_glu=False,
        ln_type="preln",  # one of "normformer", "preln"
        use_bias=True,
        force_scale=True,
        use_rmsnorm=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        use_scan=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.use_glu = use_glu
        assert ln_type in ["normformer", "preln"], f"ln_type must be one of 'normformer', 'preln', but is {ln_type}"
        self.ln_type = ln_type
        self.use_bias = use_bias
        self.force_scale = force_scale
        self.use_rmsnorm = use_rmsnorm
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.use_scan = use_scan
        self.gradient_checkpointing = gradient_checkpointing

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to"
                f" instantiate a model of type {cls.model_type}. This is not supported"
                " for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLIPVisionConfig(PretrainedFromWandbMixin, PretrainedConfig):
    model_type = "clip_jax_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        use_glu=False,
        ln_type="preln",  # one of "normformer", "preln"
        use_bias=True,
        force_scale=True,
        use_rmsnorm=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        use_scan=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.use_glu = use_glu
        assert ln_type in ["normformer", "preln"], f"ln_type must be one of 'normformer', 'preln', but is {ln_type}"
        self.ln_type = ln_type
        self.use_bias = use_bias
        self.force_scale = force_scale
        self.use_rmsnorm = use_rmsnorm
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_scan = use_scan
        self.gradient_checkpointing = gradient_checkpointing

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to"
                f" instantiate a model of type {cls.model_type}. This is not supported"
                " for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLIPConfig(PretrainedConfig):
    model_type = "clip_jax"
    is_composition = True

    def __init__(
        self,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        use_scan=False,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        if "text_config" not in kwargs:
            logger.info("text_config is undefined. Initializing the CLIPTextConfig with default" " values.")

        if "vision_config" not in kwargs:
            logger.info("vision_config is undefined. initializing the CLIPVisionConfig with" " default values.")

        text_config = kwargs.pop("text_config", {})
        vision_config = kwargs.pop("vision_config", {})

        self.text_config = CLIPTextConfig(
            **{**text_config, "use_scan": use_scan, "gradient_checkpointing": gradient_checkpointing}
        )
        self.vision_config = CLIPVisionConfig(
            **{**vision_config, "use_scan": use_scan, "gradient_checkpointing": gradient_checkpointing}
        )

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        use_scan=None,
        gradient_checkpointing=None,
        **kwargs,
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to"
                f" instantiate a model of type {cls.model_type}. This is not supported"
                " for all configurations of models and can yield errors."
            )

        if use_scan is not None:
            config_dict["use_scan"] = use_scan
        if gradient_checkpointing is not None:
            config_dict["gradient_checkpointing"] = gradient_checkpointing

        return cls.from_dict(config_dict, **kwargs)
