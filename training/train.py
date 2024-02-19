import itertools
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from platform import python_version
from pprint import pformat
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import flax
import flax.linen as nn
import fsspec
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tensorflow_io as tfio
import transformers
import wandb
from flax.core import FrozenDict
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from PIL import Image
from precondition_local.distributed_shampoo import GraftingType, distributed_shampoo
from tqdm import tqdm
from transformers import HfArgumentParser

from clip_jax import CLIPModel
from clip_jax.data import Dataset, logits_to_image, shift_tokens_left
from clip_jax.partitions import logical_axis_rules
from clip_jax.tokenizer import AutoTokenizer
from clip_jax.utils import count_params, load_config

try:
    from google.cloud import storage
except:
    storage = None


logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": ("The output directory where the model predictions and checkpoints will be written.")},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. Use this to continue"
                " training if output_dir points to a checkpoint directory."
            )
        },
    )
    no_cache: bool = field(default=False, metadata={"help": "Uses jax cache."})
    cache_dir: str = field(
        default="jax_cache",
        metadata={"help": ("Location for jax cache.")},
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    n_predict: Optional[int] = field(default=0, metadata={"help": "Number of predictions."})
    n_predict_batch: Optional[int] = field(default=None, metadata={"help": "Batch size for training."})
    predict_num_beams: Optional[int] = field(default=1, metadata={"help": "Num beams used during prediction."})

    batch_size_per_node: Optional[int] = field(default=64, metadata={"help": "Batch size for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": ("Number of updates steps to accumulate before performing an update" " pass.")},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={"help": ("Freezes vision tower.")},
    )
    loss_type: str = field(
        default="cross_entropy",
        metadata={"help": ("The type of loss to use. Can be 'cross_entropy' (default) or 'sigmoid'.")},
    )
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Use gradient checkpointing."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    optim: str = field(
        default="distributed_shampoo",
        metadata={"help": ('The optimizer to use. Can be "distributed_shampoo" (default), "adam" or "adafactor"')},
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay applied to parameters."})
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.99,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for Adafactor."})
    block_size_text: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    block_size_vision: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    set_opt_spec_vision_to_none: bool = field(
        default=False,
        metadata={"help": "Set the optimizer vision spec to None."},
    )
    preconditioning_compute_steps: int = field(
        default=20, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": (
                "The type of grafting to use. Can be 'rmsprop_normalized' (default),"
                " 'rmsprop', 'adagrad', 'adagrad_normalized', 'sgd' or 'sqrt_n'"
            )
        },
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Use Nesterov momentum for Distributed Shampoo."},
    )
    clip_by_scaled_gradient_norm: float = field(
        default=None,
        metadata={"help": "Clip by scaled gradient norm (only useful when using RMSProp Grafting)."},
    )
    optim_quantized: bool = field(
        default=False,
        metadata={"help": ("Whether to quantize optimizer (only supported with Distributed" " Shampoo).")},
    )
    shard_shampoo_across: str = field(
        default="data",
        metadata={
            "help": ("Whether to shard the optimizer across data devices (data), model devices (model) or both (2d).")
        },
    )
    activation_partitioning_dims: int = field(
        default=1, metadata={"help": "Number of dimensions to partition activations, 1 or 2."}
    )
    parameter_partitioning_dims: int = field(
        default=1, metadata={"help": "Number of dimensions to partition parameters, 1 or 2."}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=500, metadata={"help": "Linear warmup over warmup_steps."})
    lr_decay: str = field(
        default=None,
        metadata={
            "help": ("Decay to be used in the learning rate scheduler. Can be None, linear, cosine or exponential.")
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={"help": ("Number of transition steps associated with learning rate decay when applicable.")},
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={"help": ("Decay rate associated with learning rate when using exponential decay.")},
    )
    lr_staircase: bool = field(
        default=False,
        metadata={"help": ("Whether to use staircase or continuous learning rate when using exponential decay.")},
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
    )
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=500, metadata={"help": "Run an evaluation every X steps."})
    save_steps: int = field(default=None, metadata={"help": "Save checkpoint every X updates steps."})
    log_norm: bool = field(
        default=True,
        metadata={"help": "Log parameters and gradients norm at this frequency."},
    )
    log_histogram_steps: int = field(
        default=False,
        metadata={"help": ("Log parameters and gradients histograms at this frequency. Slows down" " training.")},
    )

    seed_model: int = field(
        default=42,
        metadata={"help": ("Random seed for the model that will be set at the beginning of" " training.")},
    )

    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity to use (for teams)."},
    )
    wandb_project: str = field(
        default="clip-jax",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="train",
        metadata={"help": "The name of the wandb job type."},
    )

    assert_TPU_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    mp_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of devices required for model parallelism. The other dimension"
                " of available devices is used for data parallelism."
            )
        },
    )

    do_profile: bool = field(
        default=False,
        metadata={"help": "Profile performance of training loop."},
    )
    do_test_steps: int = field(
        default=False,
        metadata={"help": "Run script for only a few steps."},
    )

    dp_devices: int = field(init=False)
    batch_size_per_step: int = field(init=False)
    train_batch_size: int = field(init=False)
    valid_batch_size: int = field(init=False)
    log_norm_steps: int = field(init=False)

    def __post_init__(self):
        if self.assert_TPU_available:
            assert jax.local_device_count() > 1, "TPUs in use, please check running processes"
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        if self.output_dir.startswith("gs://"):
            assert (
                storage is not None
            ), 'Could not find google.storage. Install with "pip install google-cloud-storage"'
        if self.do_profile:
            self.start_profile = 3
            self.end_profile = 6
            self.do_test_steps = self.end_profile + 2
        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
            "cosine",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"
        if self.log_norm is True:
            self.log_norm_steps = self.logging_steps
        elif self.log_norm:
            self.log_norm_steps = self.log_norm
        else:
            self.log_norm_steps = False
        if not self.do_train:
            # eval only
            self.num_train_epochs = 1
        assert self.loss_type in [
            "cross_entropy",
            "sigmoid",
        ], f"Selected loss type not supported: {self.loss_type}"
        assert self.optim in [
            "distributed_shampoo",
            "adam",
            "adafactor",
        ], f"Unknown optimizer {self.optim}"
        if self.optim == "adafactor" and self.weight_decay == 0:
            self.weight_decay = None
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        assert self.shard_shampoo_across in [
            "data",
            "model",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."
        assert self.activation_partitioning_dims in [1, 2], f"Only 1D and 2D activation partitioning supported."
        assert self.parameter_partitioning_dims in [1, 2], f"Only 1D and 2D parameter partitioning supported."
        assert self.mp_devices > 0, f"Number of devices for model parallelism must be > 0"
        assert jax.device_count() % self.mp_devices == 0, (
            f"Number of available devices ({jax.device_count()} must be divisible by"
            f" number of devices used for model parallelism ({self.mp_devices})."
        )
        self.dp_devices = jax.device_count() // self.mp_devices
        # batch sizes
        batch_size_per_node_per_step = self.batch_size_per_node * self.gradient_accumulation_steps
        self.batch_size_per_step = batch_size_per_node_per_step * jax.process_count()
        # define batch size for data loader
        self.train_batch_size = batch_size_per_node_per_step
        self.valid_batch_size = self.batch_size_per_node


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want"
                " to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and trained. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    float32_logits: Optional[bool] = field(
        default=False,
        metadata={"help": ("Cast attention logits to float32.")},
    )
    unroll: int = field(
        default=1,
        metadata={"help": ("Number of steps to unroll scanned layers.")},
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={"help": ("Restore optimizer.")},
    )
    config_metadata: Dict = field(init=False)

    def __post_init__(self):
        assert self.config_name is not None, "config_name is required and is not inferred from model_name_or_path"
        assert self.tokenizer_name is not None, "Tokenizer name is required."
        # get config metadata
        if ":" in self.config_name and not os.path.isdir(self.config_name):
            # wandb artifact
            if wandb.run is not None:
                artifact = wandb.run.use_artifact(self.config_name)
            else:
                artifact = wandb.Api().artifact(self.config_name)
            self.config_metadata = artifact.metadata
        else:
            self.config_metadata = None
        # get checkpoint path
        if self.model_name_or_path is None:
            if self.config_metadata is not None:
                self.model_name_or_path = self.config_metadata["output_dir"]
        if self.restore_state is True:
            assert self.config_metadata is not None, "Cannot restore state without config restored from W&B."


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the root training directory which contains tfrecords."}
    )
    valid_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the root validation directory which contains tfrecords."}
    )
    image_crop_size: Optional[int] = field(
        default=None,
        metadata={"help": "The dimension images need to be cropped to, if needed."},
    )
    image_crop_resize: Optional[int] = field(
        default=None,
        metadata={"help": "The dimension cropped images need to be resized to, if needed."},
    )
    min_original_image_size: Optional[int] = field(
        default=None,
        metadata={"help": ("The minimum size (resolution) of each original image from training" " set.")},
    )
    max_original_aspect_ratio: Optional[float] = field(
        default=None,
        metadata={"help": ("The maximum aspect ratio of each original image from training set.")},
    )
    seed_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "The seed used to augment the dataset."},
    )
    format: Optional[str] = field(default="rgb", metadata={"help": "The format of the images (rgb or lab)."})
    key_image: Optional[str] = field(default="webp", metadata={"help": "Name of the key containing the webp images."})
    key_caption: Optional[str] = field(
        default="caption", metadata={"help": "Name of the key containing the captions."}
    )
    mean: Optional[List[float]] = field(default=(0.5, 0.5, 0.5), metadata={"help": "The mean of the dataset."})
    std: Optional[List[float]] = field(default=(0.5, 0.5, 0.5), metadata={"help": "The std of the dataset."})

    def __post_init__(self):
        # correctly use defaults
        if self.mean is None:
            self.mean = (0.5, 0.5, 0.5)
        if self.std is None:
            self.std = (0.5, 0.5, 0.5)


def flat_args(model_args, data_args, training_args):
    args = asdict(model_args)
    args.update(asdict(data_args))
    args.update(asdict(training_args))
    return args


def split_scanned_params(data):
    """Split params between scanned and non-scanned"""
    # NOTE: technically this is not needed with Adam (and could be slower) but is required with shampoo
    flat = flatten_dict(data)
    split = {"text": {}, "vision": {}, "scanned_text": {}, "scanned_vision": {}}
    for k, v in flat.items():
        if "layers" in k:
            if "text" in k:
                split["scanned_text"][k] = v
            else:
                split["scanned_vision"][k] = v
        else:
            if "text" in k:
                split["text"][k] = v
            else:
                split["vision"][k] = v
    # remove empty keys
    split = {k: v for k, v in split.items() if v}
    for k, v in split.items():
        split[k] = unflatten_dict(v)
    return split


def unsplit_scanned_params(data):
    flat = {}
    for k in data.keys():
        flat.update(flatten_dict(data[k]))
    return unflatten_dict(flat)


def trainable_params(data, training_args):
    """Return trainable params"""
    frozen_keys = None
    if training_args.freeze_vision:
        frozen_keys = ["vision:embeddings", "vision:encoder"]
    if frozen_keys is not None:
        data_train = {}
        for k, v in flatten_dict(data, sep=":").items():
            if not any([e in k for e in frozen_keys]):
                data_train[k] = v
        return unflatten_dict(data_train, sep=":")
    else:
        return data


@dataclass
class State:
    step: int = 0
    opt_state_step: int = 0
    epoch: int = 0
    samples: int = 0
    time_total: float = 0.0
    time_train: float = 0.0
    time_per_train_step: float = 0.0
    time_per_eval: float = 0.0
    time_per_save: float = 0.0
    time_per_predict: float = 0.0
    timestamp: float = field(init=False)
    offset_time: float = field(init=False)  # used to substract eval and save times

    def __post_init__(self):
        self.timestamp = time.perf_counter()
        self.offset_time = 0.0

    @classmethod
    def from_config_metadata(cls, config_metadata, restore_state):
        if config_metadata is not None:
            init_state = {
                k: config_metadata[k] for k, v in cls.__dataclass_fields__.items() if k in config_metadata and v.init
            }
        else:
            init_state = {}
        if not restore_state:
            init_state["opt_state_step"] = 0
        return cls(**init_state)

    def update(self, **kwargs):
        # update timing info
        if kwargs.get("step", 0) > self.step:
            now = time.perf_counter()
            self.time_total += now - self.timestamp
            delta_time = now - self.timestamp - self.offset_time
            self.time_train += delta_time
            self.offset_time = 0.0
            self.timestamp = now
            self.time_per_train_step = delta_time / (kwargs["step"] - self.step)
        # update state
        for k, v in kwargs.items():
            if isinstance(v, jnp.ndarray):
                v = v.item()
            setattr(self, k, v)

    def add_time(self, key, duration):
        assert key in ["eval", "save", "predict"]
        if key == "eval":
            self.time_per_eval = duration
        elif key == "save":
            self.time_per_save = duration
        else:
            self.time_per_predict = duration
        self.offset_time += duration

    def to_dict(self):
        # return only items that are not init=False
        return {
            k: v
            for k, v in asdict(self).items()
            if k in self.__dataclass_fields__ and self.__dataclass_fields__[k].init
        }

    def log_predictions(self, predictions):
        if jax.process_index() == 0:
            data = [(wandb.Image(img), cap, pred) for img, cap, pred in predictions]
            columns = ["image", "caption", "prediction"]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"predictions": table})

    def log(self, metrics={}):
        if jax.process_index() == 0:
            metrics = jax.device_get(metrics)
            # unbox LogicallyPartitioned - https://github.com/google/maxtext/blob/ba01756ff0d96819fb2739bd6bc3648e0eeb8d2b/MaxText/max_utils.py#L77
            metrics = jax.tree_util.tree_map(
                lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
                metrics,
                is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
            )

            log_metrics = flatten_dict({"state": self.to_dict()}, sep="/")
            for k, v in metrics.items():
                if "_norm" in k:
                    log_metrics[f"{k}/"] = v
                elif "_hist" in k:
                    v = jax.tree_util.tree_map(
                        lambda x: wandb.Histogram(np_histogram=x),
                        v,
                        is_leaf=lambda x: isinstance(x, tuple),
                    )
                    log_metrics[f"{k}/"] = v
                else:
                    log_metrics[k] = v
            wandb.log(log_metrics)


def should_stop_training(metrics):
    lr = metrics.get("train/learning_rate", None)
    if lr is not None and jax.device_get(lr) == 0:
        return True
    return False


def main():
    # cluster initialization
    jax.distributed.initialize()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Use jax cache
    if not training_args.no_cache:
        cc.initialize_cache(training_args.cache_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Show arguments
    logger.info(f"Training/evaluation parameters:\n{pformat(asdict(training_args))}")
    logger.info(f"Model parameters:\n{pformat(asdict(model_args))}")
    logger.info(f"Data parameters:\n{pformat(asdict(data_args))}")

    # Info on local devices
    logger.info(f"Local TPUs/GPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs/GPUs: {jax.device_count()}")

    # Initialize datasets and pre-processing transforms
    dataset = Dataset(
        train_batch_size=training_args.train_batch_size,
        valid_batch_size=training_args.valid_batch_size,
        **asdict(data_args),
    )

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=flat_args(model_args, data_args, training_args),
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Load config
    clipConfig = load_config(model_args.config_name)

    # Update config
    clipConfig["text_config"]["unroll"] = model_args.unroll
    clipConfig["vision_config"]["unroll"] = model_args.unroll
    clipConfig["text_config"]["gradient_checkpointing"] = training_args.gradient_checkpointing
    clipConfig["vision_config"]["gradient_checkpointing"] = training_args.gradient_checkpointing
    clipConfig["text_config"]["float32_logits"] = model_args.float32_logits
    clipConfig["vision_config"]["float32_logits"] = model_args.float32_logits
    clipConfig["dtype"] = model_args.dtype

    # Load model
    model = CLIPModel(**clipConfig)
    model_eval = model if model_args.dtype == "float32" else CLIPModel(**{**clipConfig, "dtype": "float32"})

    # Update config with default fields
    clipConfig = {k: v for k, v in asdict(model).items() if k not in ["parent", "name"]}

    # Load state
    state = State.from_config_metadata(model_args.config_metadata, model_args.restore_state)

    # set rng
    rng = jax.random.PRNGKey(training_args.seed_model)

    # get PartitionSpec and shape for model params
    logical_params = jax.eval_shape(lambda rng: model.init_weights(rng), rng)["params"]

    # Parameter count
    num_params = {
        "Total": count_params(logical_params),
        "Text": count_params(logical_params["text"]),
        "Vision": count_params(logical_params["vision"]),
    }

    # Log some info
    logger.info(f"Num epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size per node = {training_args.batch_size_per_node}")
    logger.info(f"Number of devices = {jax.device_count()}")
    logger.info(f"Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"Batch size per update = {training_args.batch_size_per_step}")
    logger.info(
        f"Model parameters: Text {num_params['Text']:,} + Vision {num_params['Vision']:,} + Projection = {num_params['Total']:,}"
    )

    # update wandb run
    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        wandb.define_metric("*", step_metric="state/step")

        # add interesting config parameters
        wandb.config.update(
            {
                "batch_size_per_step": training_args.batch_size_per_step,
                "num_params": num_params,
                "model_config": clipConfig,
                "num_devices": jax.device_count(),
                "versions": {
                    "python": python_version(),
                    "jax": jax.__version__,
                    "jaxlib": jaxlib.__version__,
                    "flax": flax.__version__,
                    "optax": optax.__version__,
                    "orbax": orbax.checkpoint.__version__,
                    "numpy": np.__version__,
                    "tensorflow": tf.__version__,
                    "tensorflow-io": tfio.__version__,
                    "transformers": transformers.__version__,
                    "wandb": wandb.__version__,
                },
            }
        )

    # Partition Spec
    logical_spec = nn.get_partition_spec(logical_params)
    rules = logical_axis_rules(
        activation_partitioning_dims=training_args.activation_partitioning_dims,
        parameter_partitioning_dims=training_args.parameter_partitioning_dims,
    )
    params_spec = nn.logical_to_mesh(logical_spec, rules)
    data_spec = nn.logical_to_mesh(PartitionSpec("batch"), rules)
    scan_spec = PartitionSpec(None)

    # Orbax checkpointer
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # Create mesh
    logger.info(f"Creating a mesh of ({training_args.dp_devices}, {training_args.mp_devices})")
    dev_mesh = create_device_mesh((training_args.dp_devices, training_args.mp_devices))
    mesh = Mesh(dev_mesh, ("data", "model"))
    logger.info(f"Mesh: {mesh.shape}")

    # Data sharding parameters
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()
    data_global_spec = PartitionSpec(("data", "model"))
    data_global_sharding = NamedSharding(mesh, data_global_spec)
    data_mesh_spec = PartitionSpec("data")
    data_mesh_sharding = NamedSharding(mesh, data_mesh_spec)
    local_addressable_devices = data_global_sharding.addressable_devices
    data_global_shape = None  # will be set at first batch
    reshard_data = pjit(lambda x: x, in_shardings=(data_global_sharding,), out_shardings=data_mesh_sharding)

    def _get_global_shape(x):
        shape = x.shape
        # multiply first dimension by number of hosts
        shape = (shape[0] * num_hosts,) + shape[1:]
        return shape

    # Initialize or restore model
    logger.info("Initializing model parameters")

    @partial(pjit, in_shardings=None, out_shardings=params_spec)
    def init_params():
        if model_args.model_name_or_path is None:
            params = model.init_weights(rng)["params"]
        else:
            # init to 0 (faster)
            params = jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), logical_params)
        return params

    # Set params
    with mesh:
        params = init_params()

    # Restore checkpoint
    def _restore_checkpoint(ckpt, dir, step):
        logger.info(f"Restoring checkpoint from {dir} at step {step}")
        restore_args = orbax_utils.restore_args_from_target(ckpt, mesh)
        orbax_options = orbax.checkpoint.CheckpointManagerOptions()
        checkpoint_manager = orbax.checkpoint.CheckpointManager(dir, orbax_checkpointer, orbax_options)
        return checkpoint_manager.restore(step, ckpt, restore_kwargs={"restore_args": restore_args, "transforms": {}})

    ckpt = {}
    if model_args.model_name_or_path is not None:
        logger.info("Restoring model parameters")
        ckpt["params"] = params

    # Create learning rate schedule
    def create_learning_rate_fn() -> Callable[[int], jnp.array]:
        """Create the learning rate function."""

        def _add_schedule(schedule, new_schedule, boundary):
            if schedule is None:
                return new_schedule
            else:
                return optax.join_schedules(
                    schedules=[schedule, new_schedule],
                    boundaries=[boundary],
                )

        # build schedule
        schedule_fn = None
        last_boundary = 0

        # offset
        lr_offset = training_args.lr_offset + state.opt_state_step
        if lr_offset:
            schedule_fn = _add_schedule(schedule_fn, optax.constant_schedule(0.0), last_boundary)
            last_boundary += lr_offset

        # warmup
        if training_args.warmup_steps > 0:
            new_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=training_args.learning_rate,
                transition_steps=training_args.warmup_steps,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
            last_boundary += training_args.warmup_steps

        # decay
        if training_args.lr_decay == "linear":
            new_schedule = optax.linear_schedule(
                init_value=training_args.learning_rate,
                end_value=0,
                transition_steps=training_args.lr_transition_steps,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        elif training_args.lr_decay == "exponential":
            new_schedule = optax.exponential_decay(
                init_value=training_args.learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        elif training_args.lr_decay == "cosine":
            new_schedule = optax.cosine_decay_schedule(
                init_value=training_args.learning_rate, decay_steps=training_args.lr_transition_steps
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        else:
            # constant
            new_schedule = optax.constant_schedule(training_args.learning_rate)
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)

        return schedule_fn

    learning_rate_fn = create_learning_rate_fn()

    # create optimizer
    if training_args.optim == "distributed_shampoo":
        graft_type = {
            "sgd": GraftingType.SGD,
            "adagrad": GraftingType.ADAGRAD,
            "rmsprop": GraftingType.RMSPROP,
            "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
            "sqrt_n": GraftingType.SQRT_N,
            "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
        }[training_args.graft_type]
        statistics_partition_spec = (
            PartitionSpec(None, training_args.shard_shampoo_across, None)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(None, "data", "model")
        )
        preconditioner_axis = (
            training_args.shard_shampoo_across
            if training_args.shard_shampoo_across != "2d"
            else "model" if training_args.mp_devices > training_args.dp_devices else "data"
        )
        preconditioner_num_devices = (
            training_args.mp_devices if preconditioner_axis == "model" else training_args.dp_devices
        )
        _opt = partial(
            distributed_shampoo,
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-6,
            weight_decay=training_args.weight_decay,
            start_preconditioning_step=max(training_args.preconditioning_compute_steps + 1, 101),
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=graft_type,
            nesterov=training_args.nesterov,
            exponent_override=0,
            statistics_partition_spec=statistics_partition_spec,
            preconditioner_partition_spec=PartitionSpec(preconditioner_axis, None, None),
            num_devices_for_pjit=preconditioner_num_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=training_args.clip_by_scaled_gradient_norm,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
            generate_training_metrics=False,
        )
        # get the real optimizer and helper functions
        opt_text = _opt(learning_rate_fn, block_size=training_args.block_size_text)
        opt_vision = _opt(learning_rate_fn, block_size=training_args.block_size_text)
        update_fn_text = opt_text.update
        update_fn_vision = opt_vision.update

        # for main optimizer, we need to allow scanned layers
        optimizer = {}
        opt_fn = {}
        for k, p in split_scanned_params(trainable_params(logical_params, training_args)).items():
            if ("scanned_text" in k) or ("scanned_vision" in k):
                # extract 1 layer
                p = jax.eval_shape(lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p)
            optimizer[k] = (
                opt_vision.init(p) if any(name in k for name in ["vision", "scanned_vision"]) else opt_text.init(p)
            )
            opt_fn[k] = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
                optimizer[k].pspec_fn, optimizer[k].shape_and_dtype_fn
            )
            optimizer[k] = optax.GradientTransformation(
                optimizer[k].init_fn,
                update_fn_vision if any(name in k for name in ["vision", "scanned_vision"]) else update_fn_text,
            )

    elif training_args.optim == "adam":
        _opt = partial(
            optax.adamw,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        optimizer = {
            k: _opt(learning_rate=learning_rate_fn)
            for k in split_scanned_params(trainable_params(logical_params, training_args))
        }

    elif training_args.optim == "adafactor":
        _opt = partial(
            optax.adafactor,
            clipping_threshold=training_args.max_grad_norm,
            weight_decay_rate=training_args.weight_decay,
        )
        optimizer = {
            k: _opt(learning_rate=learning_rate_fn)
            for k in split_scanned_params(trainable_params(logical_params, training_args))
        }

    # get PartitionSpecof optimizer state
    def get_opt_state_spec():
        # get opt_state shape without actual init
        opt_state_shape = {}
        for k, p in split_scanned_params(trainable_params(logical_params, training_args)).items():
            if ("scanned_text" in k) or ("scanned_vision" in k):
                opt_state_shape[k] = jax.eval_shape(jax.vmap(optimizer[k].init), p)
            else:
                opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)

        # utility functions for Adam
        def _adam_opt_state_spec_per_leaf(x, spec):
            raise NotImplementedError
            # TODO: no use of FrozenDict anymore so this needs to be updated
            if isinstance(x, FrozenDict):
                # variables with same structure as params
                return spec
            else:
                # other variables such as count
                return None

        def _adam_pspec_fn(spec, shape):
            return (
                None
                if spec is None
                else jax.tree_util.tree_map(
                    partial(_adam_opt_state_spec_per_leaf, spec=spec),
                    shape,
                    # return None spec for empty elements
                    is_leaf=lambda x: isinstance(x, (FrozenDict, optax.EmptyState)),
                )
            )

        # get PartitionSpec
        split_spec = split_scanned_params(trainable_params(params_spec, training_args))
        opt_state_spec = {}

        def _get_spec(**kwargs):
            """Get optimizer spec for a certain model portion"""
            if training_args.optim == "adafactor":
                # we just replicate to each device
                return None
            elif training_args.optim == "adam":
                return _adam_pspec_fn(kwargs["params_spec"], kwargs["opt_state_shape"])
            elif training_args.optim == "distributed_shampoo":
                return kwargs["opt_fn"].pspec_fn(
                    kwargs["params_shape"],
                    kwargs["params_spec"],
                    statistics_partition_spec,
                )
            else:
                raise NotImplementedError

        for k, p in split_scanned_params(trainable_params(logical_params, training_args)).items():
            p_spec = split_spec[k]
            if ("scanned_text" in k) or ("scanned_vision" in k):
                # extract 1 layer
                p = jax.eval_shape(lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p)
                p_spec = jax.tree_util.tree_map(lambda y: PartitionSpec(*y[1:]), p_spec)
            _opt_fn = opt_fn[k] if training_args.optim == "distributed_shampoo" else None
            opt_state_spec[k] = _get_spec(
                params_spec=p_spec,
                opt_state_shape=opt_state_shape[k],
                opt_fn=_opt_fn,
                params_shape=p,
            )
            if "scanned" in k and training_args.optim != "adafactor":
                # add scan dimension (not for adafactor which is replicated)
                opt_state_spec[k] = jax.tree_util.tree_map(
                    lambda x: PartitionSpec(*scan_spec + x),
                    opt_state_spec[k],
                    is_leaf=lambda x: isinstance(x, PartitionSpec),
                )
        return opt_state_spec

    opt_state_spec = get_opt_state_spec()

    # Initialize or restore optimizer state
    logger.info("Initializing optimizer state")

    # when training only vision head, we may have too few parameters for sharding optimizer state
    if training_args.set_opt_spec_vision_to_none:
        if not training_args.freeze_vision:
            logger.warn("Setting optimizer state spec for vision to None while it does not seem required")
        opt_state_spec["vision"] = None    

    @partial(pjit, in_shardings=(params_spec,), out_shardings=opt_state_spec)
    def init_opt_state(params):
        opt_state = {}
        for k, p in split_scanned_params(trainable_params(params, training_args)).items():
            init_fn = optimizer[k].init
            if "scanned" in k:
                init_fn = jax.vmap(init_fn)
            opt_state[k] = init_fn(p)
        return opt_state

    # Set opt_state
    with mesh:
        opt_state = init_opt_state(params)

    if model_args.restore_state:
        # Restore checkpoint
        logger.info("Restoring optimizer checkpoint")
        ckpt["opt_state"] = opt_state

    # restore
    if ckpt:
        ckpt = _restore_checkpoint(ckpt, model_args.model_name_or_path, state.step)
        if "params" in ckpt:
            params = ckpt["params"]
        if "opt_state" in ckpt:
            opt_state = ckpt["opt_state"]

    # Define update function
    def update_params(params, opt_state, grads):
        grads = split_scanned_params(trainable_params(grads, training_args))
        split_params = split_scanned_params(trainable_params(params, training_args))
        new_opt_state = {}
        new_params = {}
        for k, param in split_params.items():
            update_fn = optimizer[k].update
            if ("scanned_text" in k) or ("scanned_vision" in k):
                update_fn = jax.vmap(update_fn, in_axes=(0, 0, 0), out_axes=(0, 0))
            updates, new_opt_state[k] = update_fn(grads[k], opt_state[k], param)
            new_params[k] = optax.apply_updates(param, updates)
        new_params = unsplit_scanned_params(new_params)
        # merge with non-trainable params
        # NOTE: this is only for future compatibility if we want to train only certain parameters
        params, new_params = flatten_dict(params), flatten_dict(new_params)
        params.update(new_params)
        new_params = unflatten_dict(params)

        return new_params, new_opt_state

    # Define loss
    def encoder_decoder_loss(logits, labels, label_mask):
        """Cross entropy for language models"""
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = loss * label_mask
        # normalize
        loss /= jnp.sum(label_mask, axis=-1, keepdims=True)
        loss = jnp.mean(loss)
        return loss

    def cross_entropy(logits, axis):
        logits_max = jnp.max(logits, axis=axis, keepdims=True)
        logits -= jax.lax.stop_gradient(logits_max)
        label_logits = jnp.diag(logits)
        log_normalizers = jax.nn.logsumexp(logits, axis=axis)
        return jnp.mean(log_normalizers - label_logits)

    def cross_entropy_loss(similarity):
        # increase precision for large batches
        similarity = similarity.astype(jnp.float64)
        loss = (cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2
        return loss

    def mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples):
        """Positive samples are on the diagonal"""
        bs = text_embeds.shape[0]
        if negative_samples:
            labels = -np.ones((bs, bs))
        else:
            labels = 2 * np.eye(bs) - np.ones((bs, bs))
        logits = jnp.matmul(text_embeds, image_embeds.T) * logit_scale + logit_bias
        # increase precision for large batches
        logits = logits.astype(jnp.float64)
        return -jnp.mean(jax.nn.log_sigmoid(labels * logits))

    def sigmoid_loss(outputs):
        text_embeds = outputs["text_embeds"]
        image_embeds = outputs["image_embeds"]
        logit_scale = outputs["logit_scale"]
        logit_bias = outputs["logit_bias"]

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(data_spec, data_spec, PartitionSpec(None), PartitionSpec(None)),
            out_specs=data_spec,
        )
        def chunked_loss(text_embeds, image_embeds, logit_scale, logit_bias):
            axis_size = jax.lax.psum(1, axis_name="data")

            # calculate local device loss
            loss = mini_batch_sigmoid_loss(text_embeds, image_embeds, logit_scale, logit_bias, negative_samples=False)

            # add negative losses
            def add_negative_loss(i, carrys):
                cumul_loss, image_embeds = carrys
                # shift image_embeds
                image_embeds = jax.lax.ppermute(
                    image_embeds, axis_name="data", perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
                )
                # add loss (all negative samples)
                cumul_loss += mini_batch_sigmoid_loss(
                    text_embeds, image_embeds, logit_scale, logit_bias, negative_samples=True
                )
                return cumul_loss, image_embeds

            loss, _ = jax.lax.fori_loop(0, axis_size - 1, add_negative_loss, (loss, image_embeds))
            loss = loss / axis_size
            loss = loss.reshape((-1,))
            return loss

        # average loss across devices
        loss = chunked_loss(text_embeds, image_embeds, logit_scale, logit_bias)
        loss = jnp.mean(loss)
        return loss

    def compute_loss(params, minibatch, dropout_rng, model_fn, train):
        rngs = {"dropout": dropout_rng} if train else None
        labels = minibatch.pop("labels", None)
        label_mask = minibatch.pop("label_mask", None)
        outputs = model_fn.apply({"params": params}, rngs=rngs, deterministic=not train, **minibatch)
        with jax.profiler.TraceAnnotation("Compute_Loss"):
            if model.text_config["is_decoder"]:
                logits = outputs["text_model_output"]["last_hidden_state"]
                loss = encoder_decoder_loss(logits, labels, label_mask)
            elif training_args.loss_type == "cross_entropy":
                logits = outputs["logits_per_text"]
                loss = cross_entropy_loss(logits)
            elif training_args.loss_type == "sigmoid":
                loss = sigmoid_loss(outputs)
            else:
                raise NotImplementedError
            return loss

    # Define gradient update step fn
    def train_step(rng, params, opt_state, batch, step):
        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            def _check_shape(x):
                assert (
                    x.shape[0] == training_args.batch_size_per_step
                ), f"{x.shape[0]} != {training_args.batch_size_per_step}"

            jax.tree_map(_check_shape, batch)
            offset = grad_idx * training_args.batch_size_per_node
            length = training_args.batch_size_per_node
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, offset, length), batch)

        train_compute_loss = partial(compute_loss, train=True)
        grad_fn = jax.value_and_grad(train_compute_loss)

        def loss_and_grad(grad_idx, dropout_rng):
            # minibatch at grad_idx for gradient accumulation (None otherwise)
            minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            # ensure it is sharded properly
            minibatch = with_sharding_constraint(minibatch, data_spec)
            # only 1 single rng per grad step, let us handle larger batch size (not sure why)
            dropout_rng, _ = jax.random.split(dropout_rng)
            # get loss and grads
            loss, grads = grad_fn(params, minibatch, dropout_rng, model)
            # ensure grads are sharded
            grads = with_sharding_constraint(grads, params_spec)
            # return loss and grads
            return loss, grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, rng = loss_and_grad(None, rng)
        else:
            # create initial state for cumul_minibatch_step loop
            init_state = (
                0.0,
                with_sharding_constraint(jax.tree_util.tree_map(jnp.zeros_like, params), params_spec),
                rng,
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_state):
                (
                    cumul_loss,
                    cumul_grads,
                    rng,
                ) = cumul_state
                loss, grads, rng = loss_and_grad(grad_idx, rng)
                (
                    cumul_loss,
                    cumul_grads,
                ) = jax.tree_util.tree_map(
                    jnp.add,
                    (cumul_loss, cumul_grads),
                    (loss, grads),
                )
                cumul_grads = with_sharding_constraint(cumul_grads, params_spec)
                return (
                    cumul_loss,
                    cumul_grads,
                    rng,
                )

            # loop over gradients
            loss, grads, rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_state,
            )
            grads = with_sharding_constraint(grads, params_spec)
            # sum -> mean
            loss, grads = jax.tree_util.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps,
                (loss, grads),
            )

        # enforce sharding
        grads = with_sharding_constraint(grads, params_spec)

        # update params
        new_params, new_opt_state = update_params(params, opt_state, grads)

        # get opt_state_step - TODO: only shampoo is supported at the moment
        if training_args.optim == "distributed_shampoo":
            opt_state_step = new_opt_state["text"][0]
        elif training_args.optim == "adafactor":
            opt_state_step = new_opt_state["text"][2].count
        else:
            raise NotImplementedError

        # get metrics
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate_fn(opt_state_step),
        }

        # increment step
        step += 1

        # extract norms and histograms

        def maybe_fn(fn, val, freq):
            """Call fn only if it is a logging step"""
            trainable_val = trainable_params(val, training_args)
            return jax.lax.cond(
                state.step % freq == 0,
                fn,
                lambda p: jax.tree_map(lambda v: jnp.zeros_like(v), fn(p)),
                trainable_val,
            )

        if training_args.log_norm_steps:

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val)

            gradients_norm = maybe_fn(norm, grads, training_args.log_norm_steps)
            params_norm = maybe_fn(norm, params, training_args.log_norm_steps)

            metrics.update(
                {
                    "gradients_norm": gradients_norm,
                    "params_norm": params_norm,
                }
            )

        if training_args.log_histogram_steps:

            def histogram(val):
                return jax.tree_util.tree_map(lambda x: jnp.histogram(x, density=True), val)

            gradients_hist = maybe_fn(histogram, grads, training_args.log_histogram_steps)
            params_hist = maybe_fn(histogram, params, training_args.log_histogram_steps)

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return metrics, new_params, new_opt_state, opt_state_step

    # Evaluation step
    def eval_step(params, batch):
        def compute_eval_loss(batch):
            loss = compute_loss(
                params,
                batch,
                dropout_rng=None,
                model_fn=model_eval,
                train=False,
            )
            return {
                "eval/loss": loss,
            }

        metrics = compute_eval_loss(batch)
        return metrics

    # Predict step
    def predict_step(params, batch):
        outputs = model.generate(
            batch["pixel_values"], params=params, num_beams=training_args.predict_num_beams
        ).sequences
        return outputs

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_shardings=(None, params_spec, opt_state_spec, data_spec, None),
        out_shardings=(None, params_spec, opt_state_spec, None),
        donate_argnums=(1, 2),
    )
    p_eval_step = pjit(
        eval_step,
        in_shardings=(params_spec, data_spec),
        out_shardings=None,
    )
    p_predict_step = pjit(
        predict_step,
        in_shardings=(params_spec, data_spec),
        out_shardings=None,
    )

    def run_evaluation(params, mesh):
        start_eval_time = time.perf_counter()
        metrics = []
        for batch in tqdm(
            dataset.valid,
            desc="Evaluating...",
            position=2,
            leave=False,
            disable=jax.process_index() > 0,
        ):
            # preprocess batch
            captions = [caption.decode("utf-8") for caption in batch[1]]
            txt_inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=model.text_config["max_length"],
                return_tensors="np",
            )
            # keep only input_ids and attention_mask
            txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
            # add labels for decoder
            if model.text_config["is_decoder"]:
                txt_inputs["labels"] = shift_tokens_left(txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id)
                txt_inputs["label_mask"] = shift_tokens_left(txt_inputs["attention_mask"], pad_token_id=0)
            batch = {"pixel_values": batch[0], **txt_inputs}

            # convert to jax arrays
            data_global_shape_eval = jax.tree_map(_get_global_shape, batch)
            # split per device
            batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)
            # put data on device
            batch = jax.tree_map(
                lambda x: [jax.device_put(arr, d) for arr, d in zip(x, local_addressable_devices)],
                batch,
                is_leaf=lambda x: isinstance(x, list),
            )
            # create global array
            batch = jax.tree_map(
                lambda shape, data: jax.make_array_from_single_device_arrays(shape, data_global_sharding, data),
                data_global_shape_eval,
                batch,
                is_leaf=lambda x: isinstance(x, (list, tuple)),
            )
            # reshard per mesh
            with mesh:
                batch = reshard_data(batch)

            # accumulate losses async
            with mesh:
                metrics_batch = p_eval_step(params, batch)
            metrics_batch = jax.device_get(metrics_batch)
            metrics.append(metrics_batch)

        # get the mean of the metrics
        metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *metrics)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        # update timing
        state.add_time("eval", time.perf_counter() - start_eval_time)

        # log metrics
        state.log(metrics)

        # Print metrics and update progress bar
        print(f"Eval metrics: {metrics}")

        # predict if needed
        if training_args.n_predict:
            run_predict(params, mesh)

    def run_predict(params, mesh):
        start_eval_time = time.perf_counter()
        predictions = []

        # shorten batches if possible
        n_predict_batch = training_args.n_predict if training_args.n_predict_batch is None else training_args.n_predict_batch
        max_batch = n_predict_batch // jax.process_count()
        max_batch = max(max_batch, num_local_devices)

        for batch in tqdm(
            dataset.valid,
            desc="Predicting...",
            position=2,
            leave=False,
            disable=jax.process_index() > 0,
        ):
            # shorten batch if possible
            batch = jax.tree_map(lambda x: x[:max_batch], batch)

            # preprocess batch
            captions = [caption.decode("utf-8") for caption in batch[1]]
            txt_inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=model.text_config["max_length"],
                return_tensors="np",
            )
            # keep only input_ids and attention_mask
            txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
            batch = {"pixel_values": batch[0], **txt_inputs}

            # convert to jax arrays
            data_global_shape_eval = jax.tree_map(_get_global_shape, batch)
            # split per device
            batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)
            # put data on device
            batch = jax.tree_map(
                lambda x: [jax.device_put(arr, d) for arr, d in zip(x, local_addressable_devices)],
                batch,
                is_leaf=lambda x: isinstance(x, list),
            )
            # create global array
            batch = jax.tree_map(
                lambda shape, data: jax.make_array_from_single_device_arrays(shape, data_global_sharding, data),
                data_global_shape_eval,
                batch,
                is_leaf=lambda x: isinstance(x, (list, tuple)),
            )
            # reshard per mesh
            with mesh:
                batch = reshard_data(batch)

            # predict
            if len(predictions) < training_args.n_predict:
                with mesh:
                    predictions_batch = p_predict_step(params, batch)
                n_needed = training_args.n_predict - len(predictions)
                predictions_batch = predictions_batch[:n_needed]
                predictions_batch = jax.device_get(predictions_batch)
                preds = tokenizer.batch_decode(predictions_batch, skip_special_tokens=True)
                preds = [c.strip() for c in preds]
                pixel_values = batch["pixel_values"][:n_needed]
                images = jax.device_get(pixel_values)
                images = images[:n_needed]
                images = logits_to_image(images)
                images = [Image.fromarray(img) for img in images]
                input_ids = batch["input_ids"][:n_needed]
                input_ids = jax.device_get(input_ids)
                captions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                captions = [c.strip() for c in captions]
                predictions.extend(list(zip(images, captions, preds)))
            else:
                break

        # log predictions
        if len(predictions) > 0:
            state.log_predictions(predictions)

        # update timing
        state.add_time("predict", time.perf_counter() - start_eval_time)
        state.log({})

    def run_save_model(params, opt_state):
        def _save_checkpoint(ckpt, dir, step):
            orbax_options = orbax.checkpoint.CheckpointManagerOptions(create=True)
            save_checkpoint_manager = orbax.checkpoint.CheckpointManager(dir, orbax_checkpointer, orbax_options)
            save_args = orbax_utils.save_args_from_target(ckpt)
            save_checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})

        start_save_time = time.perf_counter()

        # save config
        if jax.process_index() == 0:
            config_path = f"{training_args.output_dir}/config.json"
            with fsspec.open(config_path, "w") as f:
                json.dump(clipConfig, f, indent=2)
        multihost_utils.sync_global_devices("save_config")

        # save
        ckpt = {}
        if params is not None:
            ckpt["params"] = params
        if opt_state is not None:
            ckpt["opt_state"] = opt_state
        _save_checkpoint(ckpt, training_args.output_dir, state.step)

        # save config
        if jax.process_index() == 0:
            # config
            artifact = wandb.Artifact(
                name=f"config-{wandb.run.id}",
                type="config",
                metadata={"output_dir": training_args.output_dir, **state.to_dict()},
            )
            with artifact.new_file("config.json", mode="w", encoding="utf-8") as f:
                json.dump(clipConfig, f, indent=2)
            with artifact.new_file("state.json", mode="w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f)
            wandb.run.log_artifact(artifact)

        # update timing
        state.add_time("save", time.perf_counter() - start_save_time)
        state.log({"saved_checkpoint": state.step})

    # Init training variables
    evaluation_ran, save_model_ran, metrics_logged, stop_training = False, False, False, False
    step, samples = state.step, state.samples  # separate copies for timing metrics
    opt_state_step = 0  # ensure it is defined in evaluation mode
    step_start = step  # define it for test mode
    metrics = {}  # ensure it is defined in evaluation mode
    epochs = tqdm(
        range(training_args.num_train_epochs),
        desc=f"Epoch ... (1/{training_args.num_train_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    # Training loop
    logger.info("***** Running training *****")
    for epoch in epochs:
        if stop_training:
            break

        state.update(epoch=epoch)
        state.log({})

        # train
        if training_args.do_train:
            if not training_args.do_profile:
                batch_iterator = dataset.train
            else:
                # we don't want tensorflow loading in the profile
                sample = next(iter(dataset.train))
                batch_iterator = itertools.repeat(sample)
            for batch in tqdm(
                batch_iterator,
                desc="Training...",
                position=1,
                leave=False,
                disable=jax.process_index() > 0,
            ):
                # reset control variables
                evaluation_ran, save_model_ran, metrics_logged = False, False, False

                # preprocess batch
                captions = [caption.decode("utf-8") for caption in batch[1]]
                txt_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=model.text_config["max_length"],
                    return_tensors="np",
                )
                # keep only input_ids and attention_mask
                txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
                # add labels for decoder
                if model.text_config["is_decoder"]:
                    txt_inputs["labels"] = shift_tokens_left(
                        txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id
                    )
                    txt_inputs["label_mask"] = shift_tokens_left(txt_inputs["attention_mask"], pad_token_id=0)
                batch = {"pixel_values": batch[0], **txt_inputs}

                # reshape batch
                if data_global_shape is None:
                    data_global_shape = jax.tree_map(_get_global_shape, batch)

                # split per device
                batch = jax.tree_map(lambda x: np.split(x, num_local_devices, axis=0), batch)

                # put data on device
                batch = jax.tree_map(
                    lambda x: [jax.device_put(arr, d) for arr, d in zip(x, local_addressable_devices)],
                    batch,
                    is_leaf=lambda x: isinstance(x, list),
                )

                # create global array
                batch = jax.tree_map(
                    lambda shape, data: jax.make_array_from_single_device_arrays(shape, data_global_sharding, data),
                    data_global_shape,
                    batch,
                    is_leaf=lambda x: isinstance(x, (list, tuple)),
                )

                # reshard per mesh
                with mesh:
                    batch = reshard_data(batch)

                # optional profile
                if training_args.do_profile:
                    if step == training_args.start_profile:
                        jax.block_until_ready(params)
                        jax.profiler.start_trace("./profiles")
                    elif step == training_args.end_profile:
                        jax.block_until_ready(params)
                        jax.profiler.stop_trace()

                # train step
                step_rng, rng = jax.random.split(rng)
                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    with mesh:
                        metrics, params, opt_state, opt_state_step = p_train_step(
                            step_rng, params, opt_state, batch, step
                        )
                    step += 1
                    samples += training_args.batch_size_per_step

                # log metrics
                if step % training_args.logging_steps == 0:
                    state.update(step=step, samples=samples, opt_state_step=opt_state_step)
                    if jax.process_index() == 0:
                        state.log(metrics)
                    metrics_logged = True
                    stop_training = should_stop_training(metrics)

                # evaluation
                if training_args.do_eval and step % training_args.eval_steps == 0:
                    state.update(step=step, samples=samples, opt_state_step=opt_state_step)
                    run_evaluation(params, mesh)
                    evaluation_ran = True

                # save model
                if step % training_args.save_steps == 0:
                    state.update(step=step, samples=samples, opt_state_step=opt_state_step)
                    run_save_model(params, opt_state)
                    save_model_ran = True

                # end test
                if training_args.do_test_steps and (step - step_start >= training_args.do_test_steps):
                    # terminate script
                    print("Test successful")
                    return

                # end training
                if stop_training:
                    break

        # log final metrics
        if not metrics_logged:
            state.update(step=step, samples=samples, opt_state_step=opt_state_step)
            if jax.process_index() == 0:
                state.log(metrics)

        # run final evaluation
        if training_args.do_eval and not evaluation_ran:
            state.update(step=step, samples=samples, opt_state_step=opt_state_step)
            run_evaluation(params, mesh)

        # save final model
        if not save_model_ran:
            state.update(step=step, samples=samples, opt_state_step=opt_state_step)
            run_save_model(params, opt_state)


if __name__ == "__main__":
    main()
