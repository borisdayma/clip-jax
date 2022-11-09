import io
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, NamedTuple, Optional

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import transformers
import wandb
from flax import core, struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import Repository
from jax.experimental import PartitionSpec, maps
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit, with_sharding_constraint
from scalable_shampoo.distributed_shampoo import GraftingType, distributed_shampoo
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.utils import get_full_repo_name

try:
    from google.cloud import storage
except:
    storage = None

try:
    from dalle_mini.model.text import TextNormalizer
except ImportError:
    print("Text normalization not available")

from clip_jax import CLIPConfig, FlaxCLIPModel
from clip_jax.data import Dataset
from clip_jax.partitions import set_partitions

logger = logging.getLogger(__name__)

jax.distributed.initialize()


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": ("The output directory where the model predictions and checkpoints will" " be written.")},
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
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    batch_size_per_node: Optional[int] = field(default=64, metadata={"help": "Batch size for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": ("Number of updates steps to accumulate before performing an update" " pass.")},
    )
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Use gradient checkpointing."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    optim: str = field(
        default="distributed_shampoo",
        metadata={"help": ('The optimizer to use. Can be "distributed_shampoo" (default) or "adam"')},
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay applied to parameters."})
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    block_size: int = field(
        default=1024,
        metadata={"help": "Chunked size for large layers with Distributed Shampoo."},
    )
    preconditioning_compute_steps: int = field(
        default=10, metadata={"help": "Number of steps to update preconditioner."}
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
    optim_quantized: bool = field(
        default=False,
        metadata={"help": ("Whether to quantize optimizer (only supported with Distributed" " Shampoo).")},
    )
    shard_shampoo_across: str = field(
        default="dp",
        metadata={
            "help": ("Whether to shard the optimizer across data devices (dp), model devices" " (mp) or both (2d).")
        },
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=500, metadata={"help": "Linear warmup over warmup_steps."})
    lr_decay: str = field(
        default=None,
        metadata={
            "help": (
                "Decay to be used in the learning rate scheduler. Can be None" " (default), linear or exponential."
            )
        },
    )
    num_train_steps: int = field(
        default=None,
        metadata={
            "help": (
                "Total number of training steps to perform. Required only when defining"
                " using linear learning rate decay."
            )
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={
            "help": ("Number of transition steps associated with learning rate decay when" " using exponential decay.")
        },
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={"help": ("Decay rate associated with learning rate when using exponential decay.")},
    )
    lr_staircase: bool = field(
        default=False,
        metadata={"help": ("Whether to use staircase or continuous learning rate when using" " exponential decay.")},
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
    )
    logging_steps: int = field(default=40, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=400, metadata={"help": "Run an evaluation every X steps."})
    save_steps: int = field(default=4000, metadata={"help": "Save checkpoint every X updates steps."})
    log_model: bool = field(
        default=False,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )
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
        default="clip",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="train",
        metadata={"help": "The name of the wandb job type."},
    )

    push_to_hub: bool = field(
        default=False,
        metadata={"help": ("Whether or not to upload the trained model to the model hub after" " training.")},
    )
    hub_model_id: str = field(
        default=None,
        metadata={"help": ("The name of the repository to keep in sync with the local" " `output_dir`.")},
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    assert_TPU_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    use_vmap_trick: bool = field(
        default=False,
        metadata={"help": "Optimization trick that should lead to faster training."},
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

    dp_devices: int = field(init=False)
    local_dp_devices: int = field(init=False)
    node_groups: int = field(init=False)
    batch_size_per_step: int = field(init=False)
    train_batch_size: int = field(init=False)
    valid_batch_size: int = field(init=False)
    valid_batch_size_per_step: int = field(init=False)
    log_norm_steps: int = field(init=False)

    def __post_init__(self):
        if self.assert_TPU_available:
            assert jax.local_device_count() == 8, "TPUs in use, please check running processes"
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
        assert self.lr_decay in [
            None,
            "linear",
            "exponential",
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
        assert self.optim in [
            "distributed_shampoo",
            "adam",
        ], f"Unknown optimizer {self.optim}"
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        assert self.shard_shampoo_across in [
            "dp",
            "mp",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."
        assert self.mp_devices > 0, f"Number of devices for model parallelism must be > 0"
        assert jax.device_count() % self.mp_devices == 0, (
            f"Number of available devices ({jax.device_count()} must be divisible by"
            f" number of devices used for model parallelism ({self.mp_devices})."
        )
        self.dp_devices = jax.device_count() // self.mp_devices
        # consider batch distributed across nodes (mp > local devices)
        self.node_groups = max(1, self.mp_devices // jax.local_device_count())
        # local dp devices (1 when mp > local devices)
        self.local_dp_devices = jax.local_device_count() * self.node_groups // self.mp_devices
        # batch sizes
        assert self.batch_size_per_node % self.local_dp_devices == 0, (
            f"Batch size per node ({self.batch_size_per_node}) must be divisible by"
            f" number of local devices ({jax.local_device_count()})."
        )
        batch_size_per_node_per_step = self.batch_size_per_node * self.gradient_accumulation_steps
        self.batch_size_per_step = batch_size_per_node_per_step * jax.process_count()
        self.batch_size_per_local_dp_device = self.batch_size_per_node // self.local_dp_devices
        # define batch size for data loader
        self.train_batch_size = batch_size_per_node_per_step * self.node_groups
        self.valid_batch_size = self.batch_size_per_node * self.node_groups
        self.valid_batch_size_per_step = self.batch_size_per_node * jax.process_count()

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


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
    normalize_text: bool = field(
        default=False,
        metadata={"help": "Normalize text before tokenization"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Where do you want to store the pretrained models downloaded from s3")},
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
    use_scan: bool = field(
        default=False,
        metadata={"help": "Use scan on the model layers."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login`"
                " (necessary to use this script with private models)."
            )
        },
    )
    restore_state: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Restore optimizer and training state. Can be True (will retrieve"
                " associated wandb artifact) or a local directory."
            )
        },
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
            assert self.tokenizer_name is not None, "Tokenizer name or model name/path needs to be specified"
        if self.restore_state is True:
            assert (
                self.model_name_or_path is not None
            ), "If you want to restore state, you must provide a model name or path."
        if self.use_scan:
            assert self.model_name_or_path is None, "use_scan can only be defined when training from scratch."

    def get_metadata(self):
        if self.model_name_or_path is not None and ":" in self.model_name_or_path:
            if jax.process_index() == 0:
                artifact = wandb.run.use_artifact(self.model_name_or_path)
            else:
                artifact = wandb.Api().artifact(self.model_name_or_path)
            return artifact.metadata
        else:
            return dict()

    def get_opt_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # avoid multiple artifact copies
            if self.restore_state is True:
                # wandb artifact
                state_artifact = self.model_name_or_path.replace("/model-", "/state-", 1)
                if jax.process_index() == 0:
                    artifact = wandb.run.use_artifact(state_artifact)
                else:
                    artifact = wandb.Api().artifact(state_artifact)
                if artifact.metadata.get("bucket_path"):
                    # we will read directly file contents
                    self.restore_state = artifact.metadata["bucket_path"]
                else:
                    artifact_dir = artifact.download(tmp_dir)
                    self.restore_state = Path(artifact_dir)

            if self.restore_state.startswith("gs://"):
                bucket_path = Path(self.restore_state[5:]) / "opt_state.msgpack"
                bucket, blob_name = str(bucket_path).split("/", 1)
                assert (
                    storage is not None
                ), 'Could not find google.storage. Install with "pip install google-cloud-storage"'
                client = storage.Client()
                bucket = client.bucket(bucket)
                blob = bucket.blob(blob_name)
                return blob.download_as_bytes()

            with (Path(self.restore_state) / "opt_state.msgpack").open("rb") as f:
                return f.read()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_folder: str = field(metadata={"help": "Path to the root training directory which contains tfrecords."})
    valid_folder: str = field(metadata={"help": "Path to the root validation directory which contains tfrecords."})
    image_size: Optional[int] = field(
        default=0,
        metadata={"help": " The dimension images need to be resized to, if needed."},
    )
    min_original_image_size: Optional[int] = field(
        default=None,
        metadata={"help": (" The minimum size (resolution) of each original image from training" " set.")},
    )
    max_original_aspect_ratio: Optional[float] = field(
        default=None,
        metadata={"help": (" The maximum aspect ratio of each original image from training set.")},
    )
    seed_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "The seed used to augment the dataset."},
    )
    format: Optional[str] = field(default="rgb", metadata={"help": "The format of the images (rgb or lab)."})
    key_image: Optional[str] = field(default="webp", metadata={"help": "Name of the key containing the webp images."})
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
    flat = flatten_dict(unfreeze(data))
    split = {"standard": {}, "scanned_text": {}, "scanned_vision": {}}
    for k, v in flat.items():
        if "scanned" in k:
            if "text_model" in k:
                split["scanned_text"][k] = v
            else:
                split["scanned_vision"][k] = v
        else:
            split["standard"][k] = v
    # remove empty keys
    split = {k: v for k, v in split.items() if v}
    for k, v in split.items():
        split[k] = freeze(unflatten_dict(v))
    return split


def unsplit_scanned_params(data):
    flat = {}
    for k in data.keys():
        flat.update(flatten_dict(unfreeze(data[k])))
    return freeze(unflatten_dict(flat))


assert jax.local_device_count() == 8


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check arguments
    if training_args.mp_devices > jax.local_device_count():
        assert (
            data_args.seed_dataset is not None
        ), "Seed dataset must be provided when model is split over multiple hosts"

    # Use jax cache
    if not training_args.no_cache:
        cc.initialize_cache("jax_cache")

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

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # set seed for random transforms and torch dataloaders
    set_seed(training_args.seed_model)

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # Initialize datasets and pre-processing transforms
    dataset = Dataset(
        train_batch_size=training_args.train_batch_size,
        valid_batch_size=training_args.valid_batch_size,
        valid_batch_size_per_step=training_args.valid_batch_size_per_step,
        node_groups=training_args.node_groups,
        **asdict(data_args),
    )

    # Info on local devices
    logger.info(f"Local TPUs/GPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs/GPUs: {jax.device_count()}")

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            job_type=training_args.wandb_job_type,
            config=flat_args(model_args, data_args, training_args),
        )

    # Set up model configs
    if model_args.config_name:
        config = CLIPConfig.from_pretrained(
            model_args.config_name,
            use_scan=model_args.use_scan,
            gradient_checkpointing=training_args.gradient_checkpointing,
        )
    else:
        config = None

    # Load or create new models
    if model_args.model_name_or_path:
        model, params = FlaxCLIPModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            use_scan=model_args.use_scan,
            gradient_checkpointing=training_args.gradient_checkpointing,
            _do_init=False,  # we overwrite them with loaded checkpoint
        )
        params = freeze(params)
    else:
        model = FlaxCLIPModel(
            config,
            seed=training_args.seed_model,
            dtype=getattr(jnp, model_args.dtype),
            _do_init=False,
        )
        params = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
    )
    if model_args.normalize_text:
        tn = TextNormalizer()

    # get model metadata
    model_metadata = model_args.get_metadata()

    # get PartitionSpec and shape for model params
    params_shape = freeze(model.params_shape_tree)
    params_spec = set_partitions(unfreeze(params_shape), model.config.text_config.use_scan)

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed_model)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = training_args.num_train_epochs
    num_params = model.num_params(params_shape)

    # log some info
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Batch size per node = {training_args.batch_size_per_node}")
    logger.info(f"  Number of devices = {jax.device_count()}")
    logger.info(f"  Gradient accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Batch size per update = {training_args.batch_size_per_step}")
    logger.info(f"  Model parameters = {num_params:,}")

    if jax.process_index() == 0:
        # set default x-axis as 'train/step'
        wandb.define_metric("*", step_metric="train/step")

        # add interesting config parameters
        wandb.config.update(
            {
                "batch_size_per_step": training_args.batch_size_per_step,
                "num_params": num_params,
                "model_config": model.config.to_dict(),
                "num_devices": jax.device_count(),
                "versions": {
                    "jax": jax.__version__,
                    "jaxlib": jaxlib.__version__,
                    "flax": flax.__version__,
                    "optax": optax.__version__,
                    "transformers": transformers.__version__,
                    "wandb": wandb.__version__,
                },
            }
        )

    # Create learning rate schedule
    def create_learning_rate_fn(learning_rate) -> Callable[[int], jnp.array]:
        """Create the learning rate function."""
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=training_args.warmup_steps + 1,  # ensure not 0
        )
        last_boundary = training_args.warmup_steps
        # offset step when resuming
        if training_args.lr_offset:
            warmup_fn = optax.join_schedules(
                schedules=[optax.constant_schedule(0.0), warmup_fn],
                boundaries=[training_args.lr_offset],
            )
            last_boundary += training_args.lr_offset
        if training_args.lr_decay is None:
            return warmup_fn
        elif training_args.lr_decay == "linear":
            assert (
                training_args.num_train_steps is not None
            ), "linear decay requires specifying explicitly num_train_steps"
            assert training_args.num_train_steps > training_args.warmup_steps, (
                "linear decay requires number of training steps > warmup steps, got"
                f" {training_args.num_train_steps} < {training_args.warmup_steps}"
            )
            decay_fn = optax.linear_schedule(
                init_value=learning_rate,
                end_value=0,
                transition_steps=training_args.num_train_steps - training_args.warmup_steps,
            )
        elif training_args.lr_decay == "exponential":
            decay_fn = optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[last_boundary],
        )
        return schedule_fn

    learning_rate_fn = create_learning_rate_fn(training_args.learning_rate)

    # create optimizer
    if training_args.optim == "distributed_shampoo":
        # parameters from https://github.com/tensorflow/lingvo/blob/03ee9d7cd50764b0424c7c863733c91fc0b053ec/lingvo/jax/optimizers.py#L729
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
            else PartitionSpec(None, "dp", "mp")
        )
        _opt = partial(
            distributed_shampoo,
            block_size=training_args.block_size,
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
            preconditioner_partition_spec=PartitionSpec(training_args.shard_shampoo_across, None, None)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(
                "mp" if training_args.mp_devices > training_args.dp_devices else "dp",
                None,
                None,
            ),
            num_devices_for_pjit=training_args.dp_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=None,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
        )
        # get the real optimizer and helper functions
        opt = _opt(learning_rate_fn)
        update_fn = opt.update

        # for main optimizer, we need to allow scanned layers
        optimizer = {}
        opt_fn = {}
        for k, p in split_scanned_params(params_shape).items():
            if ("scanned_text" in k) or ("scanned_vision" in k):
                # extract 1 layer
                p = jax.eval_shape(lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p)
            optimizer[k] = opt.init(p)
            opt_fn[k] = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
                optimizer[k].pspec_fn, optimizer[k].shape_and_dtype_fn
            )
            optimizer[k] = optax.GradientTransformation(optimizer[k].init_fn, update_fn)

    elif training_args.optim == "adam":
        _opt = partial(
            optax.adamw,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        optimizer = {k: _opt(learning_rate=learning_rate_fn) for k in split_scanned_params(params_shape)}

    # get PartitionSpec and shape of optimizer state
    def get_opt_state_spec_and_shape():
        # get opt_state shape without actual init
        opt_state_shape = {}
        for k, p in split_scanned_params(params_shape).items():
            if ("scanned_text" in k) or ("scanned_vision" in k):
                opt_state_shape[k] = jax.eval_shape(jax.vmap(optimizer[k].init), p)
            else:
                opt_state_shape[k] = jax.eval_shape(optimizer[k].init, p)

        # utility functions for Adam
        def _adam_opt_state_spec_per_leaf(x, spec):
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
        split_spec = split_scanned_params(params_spec)
        opt_state_spec = {}

        def _get_spec(**kwargs):
            """Get optimizer spec for a certain model portion"""
            if training_args.optim == "adam":
                return _adam_pspec_fn(kwargs["params_spec"], kwargs["opt_state_shape"])
            elif training_args.optim == "distributed_shampoo":
                return kwargs["opt_fn"].pspec_fn(
                    kwargs["params_shape"],
                    kwargs["params_spec"],
                    statistics_partition_spec,
                )
            else:
                raise NotImplementedError

        for k, p in split_scanned_params(params_shape).items():
            if ("scanned_text" in k) or ("scanned_vision" in k):
                # extract 1 layer
                p = jax.eval_shape(lambda x: jax.tree_util.tree_map(lambda y: y[0], x), p)
            _opt_fn = opt_fn[k] if training_args.optim == "distributed_shampoo" else None
            opt_state_spec[k] = _get_spec(
                params_spec=split_spec[k],
                opt_state_shape=opt_state_shape[k],
                opt_fn=_opt_fn,
                params_shape=p,
            )
        return opt_state_spec, opt_state_shape

    opt_state_spec, opt_state_shape = get_opt_state_spec_and_shape()

    # create a mesh
    mesh_shape = (training_args.dp_devices, training_args.mp_devices)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = maps.Mesh(devices, ("dp", "mp"))
    logger.info(f"  Mesh shape: {mesh_shape}")

    class TrainState(struct.PyTreeNode):
        step: int
        params: core.FrozenDict[str, Any]
        opt_state: optax.OptState
        dropout_rng: jnp.ndarray = None
        epoch: int = 0
        train_time: float = 0.0  # total time the model trained
        train_samples: int = 0  # number of samples seen

        def apply_gradients(self, *, grads, **kwargs):
            # apply gradients to model parameters
            grads = split_scanned_params(grads)
            params = split_scanned_params(self.params)
            new_opt_state = {}
            new_params = {}
            for k, param in params.items():
                update_fn = optimizer[k].update
                if ("scanned_text" in k) or ("scanned_vision" in k):
                    update_fn = jax.vmap(update_fn, in_axes=(0, 0, 0), out_axes=(0, 0))
                updates, new_opt_state[k] = update_fn(grads[k], self.opt_state[k], param)
                new_params[k] = optax.apply_updates(param, updates)
            new_params = unsplit_scanned_params(new_params)

            return self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                **kwargs,
            )

        @classmethod
        def create(cls, *, params, **kwargs):
            opt_state = {}
            for k, p in split_scanned_params(params).items():
                init_fn = optimizer[k].init
                if ("scanned_text" in k) or ("scanned_vision" in k):
                    init_fn = jax.vmap(init_fn)
                opt_state[k] = init_fn(p)
            return cls(
                step=0,
                params=params,
                opt_state=opt_state,
                **kwargs,
            )

    # define state spec
    state_spec = TrainState(
        params=params_spec,
        opt_state=opt_state_spec,
        dropout_rng=None,
        step=None,
        epoch=None,
        train_time=None,
        train_samples=None,
    )

    # init params if not available yet
    def maybe_init_params(params, m):
        if params is not None:
            # model params are correctly loaded
            return params
        else:
            # params have not been initialized yet
            return m.init_weights(m.key, m.input_shape)

    with mesh:
        logger.info("  Creating state")

        # restore metadata
        attr_state = {}
        keys = ["train_time", "train_samples"]
        if model_args.restore_state:
            keys += ["step", "epoch"]
        attr_state = {k: v for k, v in model_metadata.items() if k in keys}

        if not model_args.restore_state:

            def init_state(params):
                return TrainState.create(
                    params=maybe_init_params(params, model),
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                init_state,
                in_axis_resources=(params_spec,) if model_args.model_name_or_path else None,
                out_axis_resources=state_spec,
                donate_argnums=(0,),
            )(params)

        else:
            # load opt_state
            opt_state = model_args.get_opt_state()
            opt_state = from_bytes(opt_state_shape, opt_state)

            def restore_state(params, opt_state):
                return TrainState(
                    params=params,
                    opt_state=opt_state,
                    dropout_rng=dropout_rng,
                    **attr_state,
                )

            state = pjit(
                restore_state,
                in_axis_resources=(
                    params_spec,
                    opt_state_spec,
                ),
                out_axis_resources=state_spec,
                donate_argnums=(0, 1),
            )(params, opt_state)

            # remove opt_state from CPU
            del opt_state

    # free CPU memory
    del params, opt_state_spec, opt_state_shape

    # define batch spec
    batch_spec = PartitionSpec("dp")
    grad_batch_spec = PartitionSpec(None, "dp")

    # "vmap trick" avoids a crash when mp_devices > 1 (not sure why it happens)
    # lead to better perf: see https://wandb.ai/dalle-mini/dalle-mini/reports/JAX-pmap-vs-pjit--VmlldzoxNDg1ODA2
    if training_args.use_vmap_trick:
        grad_params_spec = jax.tree_util.tree_map(
            lambda x: PartitionSpec(*("dp",) + (x if x is not None else (None,))),
            params_spec,
        )

    # Define loss
    def cross_entropy(logits, axis):
        logprobs = jax.nn.log_softmax(logits, axis=axis)
        nll = jnp.diag(logprobs)
        # TODO: try to compute only necessary part of the loss per device
        # nll = with_sharding_constraint(nll, batch_spec)
        ce = -jnp.mean(nll)
        return ce

    def clip_loss(similarity):
        loss = (cross_entropy(similarity, axis=0) + cross_entropy(similarity, axis=1)) / 2
        return loss

    def compute_loss(params, minibatch, dropout_rng, model_fn, train):
        logits = model_fn(**minibatch, params=params, dropout_rng=dropout_rng, train=train)[0]
        loss = clip_loss(logits)
        return loss

    # Define gradient update step fn
    def train_step(state, batch, train_time):
        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        train_compute_loss = partial(compute_loss, train=True)
        grad_fn = jax.value_and_grad(train_compute_loss)

        def loss_and_grad(grad_idx, dropout_rng):
            # minibatch at grad_idx for gradient accumulation (None otherwise)
            minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            # ensure it is sharded properly
            minibatch = with_sharding_constraint(minibatch, batch_spec)
            # only 1 single rng per grad step, let us handle larger batch size (not sure why)
            dropout_rng, _ = jax.random.split(dropout_rng)

            if training_args.use_vmap_trick:
                # "vmap trick", calculate loss and grads independently per dp_device
                loss, grads = jax.vmap(grad_fn, in_axes=(None, 0, None, None), out_axes=(0, 0))(
                    state.params,
                    minibatch,
                    dropout_rng,
                    model,
                )
                # ensure they are sharded correctly
                loss = with_sharding_constraint(loss, batch_spec)
                grads = with_sharding_constraint(grads, grad_params_spec)

                # average across all devices
                # Note: we could average per device only after gradient accumulation, right before params update
                (loss, grads,) = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0),
                    (
                        loss,
                        grads,
                    ),
                )
            else:
                # "vmap trick" may not work in multi-hosts or require too much hbm
                loss, grads = grad_fn(
                    state.params,
                    minibatch,
                    dropout_rng,
                    model,
                )
            # ensure grads are sharded
            grads = with_sharding_constraint(grads, params_spec)
            # return loss and grads
            return loss, grads, dropout_rng

        if training_args.gradient_accumulation_steps == 1:
            loss, grads, dropout_rng = loss_and_grad(None, state.dropout_rng)
        else:
            # create initial state for cumul_minibatch_step loop
            init_minibatch_step = (
                0.0,
                with_sharding_constraint(jax.tree_util.tree_map(jnp.zeros_like, state.params), params_spec),
                state.dropout_rng,
            )

            # accumulate gradients
            def cumul_minibatch_step(grad_idx, cumul_loss_grad_dropout):
                (
                    cumul_loss,
                    cumul_grads,
                    dropout_rng,
                ) = cumul_loss_grad_dropout
                loss, grads, dropout_rng = loss_and_grad(grad_idx, dropout_rng)
                (cumul_loss, cumul_grads,) = jax.tree_util.tree_map(
                    jnp.add,
                    (cumul_loss, cumul_grads),
                    (loss, grads),
                )
                cumul_grads = with_sharding_constraint(cumul_grads, params_spec)
                return (
                    cumul_loss,
                    cumul_grads,
                    dropout_rng,
                )

            # loop over gradients
            loss, grads, dropout_rng = jax.lax.fori_loop(
                0,
                training_args.gradient_accumulation_steps,
                cumul_minibatch_step,
                init_minibatch_step,
            )
            grads = with_sharding_constraint(grads, params_spec)
            # sum -> mean
            loss, grads = jax.tree_util.tree_map(
                lambda x: x / training_args.gradient_accumulation_steps,
                (loss, grads),
            )

        grads = with_sharding_constraint(grads, params_spec)

        # update state
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=dropout_rng,
            train_time=train_time,
            train_samples=state.train_samples + training_args.batch_size_per_step,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }

        # extract norms and histograms

        def maybe_fn(fn, val, zeros, freq):
            """Call fn only if it is a logging step"""
            return jax.lax.cond(
                state.step % freq == 0,
                fn,
                lambda _: zeros,
                val,
            )

        if training_args.log_norm_steps:
            zeros_norm = jax.tree_util.tree_map(lambda _: jnp.float32(0), state.params)

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val)

            gradients_norm = maybe_fn(norm, grads, zeros_norm, training_args.log_norm_steps)
            params_norm = maybe_fn(norm, state.params, zeros_norm, training_args.log_norm_steps)

            metrics.update(
                {
                    "gradients_norm": gradients_norm,
                    "params_norm": params_norm,
                }
            )

        if training_args.log_histogram_steps:
            zeros_hist = jax.tree_util.tree_map(lambda _: jnp.histogram(jnp.zeros(1), density=True), state.params)

            def histogram(val):
                return jax.tree_util.tree_map(lambda x: jnp.histogram(x, density=True), val)

            gradients_hist = maybe_fn(histogram, grads, zeros_hist, training_args.log_histogram_steps)
            params_hist = maybe_fn(histogram, state.params, zeros_hist, training_args.log_histogram_steps)

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return state, metrics

    # Ensure eval_fn is in float32 to avoid numerical issues
    eval_model = (
        model
        if model_args.dtype == "float32"
        else FlaxCLIPModel(
            model.config,
            seed=training_args.seed_model,
            dtype=jnp.float32,
            _do_init=False,
        )
    )

    # Evaluation step
    def eval_step(state, batch):
        def compute_eval_loss(batch):
            loss = compute_loss(
                state.params,
                batch,
                dropout_rng=None,
                model_fn=eval_model,
                train=False,
            )
            return {
                "loss": loss,
            }

        if training_args.use_vmap_trick:
            metrics = jax.vmap(compute_eval_loss)(batch)
            # ensure they are sharded correctly
            metrics = with_sharding_constraint(metrics, batch_spec)
            # average across all devices
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        else:
            metrics = compute_eval_loss(batch)

        return metrics

    # Create parallel version of the train and eval step
    p_train_step = pjit(
        train_step,
        in_axis_resources=(
            state_spec,
            grad_batch_spec if training_args.gradient_accumulation_steps > 1 else batch_spec,
            None,
        ),
        out_axis_resources=(state_spec, None),
        donate_argnums=(0,),
    )
    p_eval_step = pjit(
        eval_step,
        in_axis_resources=(state_spec, batch_spec),
        out_axis_resources=None,
    )

    # define metrics logger
    class MetricsLogger:
        def __init__(self, step):
            # keep state to use any key as a custom x-axis
            self.state_dict = {}
            # estimate speed
            self.step = step
            self.time = time.perf_counter()
            self.offset_time = 0.0

        def update_state_metrics(self, state):
            """Update internal state metrics (logged at each call to be used as x-axis)"""
            self.state_dict = {
                f'train/{k.split("_")[-1]}': state[k] for k in ["step", "epoch", "train_time", "train_samples"]
            }
            # timing metrics
            new_step = int(state["step"])
            new_time = time.perf_counter()
            if new_step > self.step:
                # remove time for eval & save
                delta_time = new_time - self.time - self.offset_time
                self.offset_time = 0
                time_per_step = delta_time / (new_step - self.step)
                self.step = new_step
                self.time = new_time
                self.log_time("train_per_step", time_per_step, offset=False)
                self.log_time("train_per_log", delta_time, offset=False)

        def log_time(self, key, duration, offset=True):
            if jax.process_index() == 0:
                wandb.log({f"time/{key}": duration, **self.state_dict})
            if offset:
                self.offset_time += duration

        def log(self, metrics, prefix=None):
            if jax.process_index() == 0:
                log_metrics = {}
                for k, v in metrics.items():
                    if "_norm" in k:
                        if self.step % training_args.log_norm_steps == 0:
                            log_metrics[f"{k}/"] = unfreeze(v)
                    elif "_hist" in k:
                        if self.step % training_args.log_histogram_steps == 0:
                            v = jax.tree_util.tree_map(lambda x: jax.device_get(x), unfreeze(v))
                            v = jax.tree_util.tree_map(
                                lambda x: wandb.Histogram(np_histogram=x),
                                v,
                                is_leaf=lambda x: isinstance(x, tuple),
                            )
                            log_metrics[f"{k}/"] = v
                    else:
                        if prefix is not None:
                            k = f"{prefix}/{k}"
                        log_metrics[k] = v
                wandb.log({**log_metrics, **self.state_dict})

    # keep local copy of state to avoid communication
    local_state = {
        k: jax.device_get(getattr(state, k)).item() for k in ["step", "epoch", "train_time", "train_samples"]
    }

    # init variables
    start_time = time.perf_counter() - local_state["train_time"]
    train_metrics = None
    evaluation_ran = False
    save_model_ran = False
    profile_status = "not started"
    profile_step_start = 102  # so we start seeing the preconditioner stats
    profile_step_end = profile_step_start + 20
    metrics_logger = MetricsLogger(local_state["step"])
    epochs = tqdm(
        range(local_state["epoch"], num_epochs),
        desc=f"Epoch ... (1/{num_epochs})",
        position=0,
        disable=jax.process_index() > 0,
    )

    def run_evaluation():
        # ======================== Evaluating ==============================
        if training_args.do_eval:

            # defragment memory
            jax.lib.xla_bridge.get_backend().defragment()

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
                if model_args.normalize_text:
                    captions = [tn(c) for c in captions]
                txt_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    truncation=True,
                    max_length=model.config.text_config.max_position_embeddings,
                    return_tensors="np",
                )
                # keep only input_ids and attention_mask
                txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
                batch = {"pixel_values": batch[0], **txt_inputs}

                # add dp dimension when using "vmap trick"
                if training_args.use_vmap_trick:
                    bs_shape = (
                        training_args.local_dp_devices,
                        training_args.batch_size_per_local_dp_device,
                    )
                    batch = jax.tree_util.tree_map(lambda x: x.reshape(bs_shape + x.shape[1:]), batch)

                # accumulate losses async
                metrics.append(p_eval_step(state, batch))

            # get the mean of the metrics
            metrics = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *metrics)
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)

            # log metrics
            metrics_logger.log(metrics, prefix="valid")

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Valid Loss: {metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            # log time
            metrics_logger.log_time("valid", time.perf_counter() - start_eval_time)

            # defragment memory
            jax.lib.xla_bridge.get_backend().defragment()

            return metrics

    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:
            start_save_time = time.perf_counter()
            output_dir = training_args.output_dir
            use_bucket = output_dir.startswith("gs://")
            if use_bucket:
                bucket_path = Path(output_dir[5:]) / wandb.run.id / f"step_{state.step}"
                bucket, dir_path = str(bucket_path).split("/", 1)
                tmp_dir = tempfile.TemporaryDirectory()
                output_dir = tmp_dir.name

            # save model
            params = jax.device_get(state.params)
            model.save_pretrained(output_dir, params=params)

            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # copy to bucket
            if use_bucket:
                client = storage.Client()
                bucket = client.bucket(bucket)
                for filename in Path(output_dir).glob("*"):
                    blob_name = str(Path(dir_path) / "model" / filename.name)
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(filename))
                tmp_dir.cleanup()

            # save state
            opt_state = jax.device_get(state.opt_state)
            if use_bucket:
                blob_name = str(Path(dir_path) / "state" / "opt_state.msgpack")
                blob = bucket.blob(blob_name)
                blob.upload_from_file(io.BytesIO(to_bytes(opt_state)))
            else:
                with (Path(output_dir) / "opt_state.msgpack").open("wb") as f:
                    f.write(to_bytes(opt_state))

            # save to HF hub
            if training_args.push_to_hub:
                repo.push_to_hub(
                    commit_message=f"Saving weights and logs of epoch {epoch}",
                    blocking=False,
                )

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("10GB"))

                metadata = {
                    k: jax.device_get(getattr(state, k)).item()
                    for k in ["step", "epoch", "train_time", "train_samples"]
                }
                metadata["num_params"] = num_params
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics

                # create model artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/model"
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="CLIP",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact.add_reference(metadata["bucket_path"])
                else:
                    for filename in [
                        "config.json",
                        "flax_model.msgpack",
                        # tokenizer
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "vocab.json",
                        "merges.txt",
                        "added_tokens.json",
                        "tokenizer.json",
                    ]:
                        if (Path(training_args.output_dir) / filename).exists():
                            artifact.add_file(Path(training_args.output_dir) / filename)
                        artifact.add_file(f"{Path(training_args.output_dir) / filename}")
                wandb.run.log_artifact(artifact)

                # create state artifact
                if use_bucket:
                    metadata["bucket_path"] = f"gs://{bucket_path}/state"
                artifact_state = wandb.Artifact(
                    name=f"state-{wandb.run.id}",
                    type="state",
                    metadata=metadata,
                )
                if use_bucket:
                    artifact_state.add_reference(metadata["bucket_path"])
                else:
                    artifact_state.add_file(f"{Path(training_args.output_dir) / 'opt_state.msgpack'}")
                wandb.run.log_artifact(artifact_state)
            metrics_logger.log_time("save_model", time.perf_counter() - start_save_time)

    # defragment memory
    jax.lib.xla_bridge.get_backend().defragment()

    logger.info("  Ready to start training")
    with mesh:
        for epoch in epochs:
            state = state.replace(epoch=epoch)
            local_state["epoch"] = epoch
            # ======================== Training ================================
            metrics_logger.update_state_metrics(local_state)
            metrics_logger.log({})

            # train
            if training_args.do_train:
                for batch in tqdm(
                    dataset.train,
                    desc="Training...",
                    position=1,
                    leave=False,
                    disable=jax.process_index() > 0,
                ):
                    # calculate delta time (we have a lag of one step but it's ok)
                    train_time = time.perf_counter() - start_time

                    # reset control variables
                    evaluation_ran = False
                    save_model_ran = False

                    # set correct shape to batch
                    bs_shape = (
                        (training_args.batch_size_per_node * training_args.node_groups,)
                        if not training_args.use_vmap_trick
                        else (
                            training_args.local_dp_devices,
                            training_args.batch_size_per_local_dp_device,
                        )
                    )
                    if training_args.gradient_accumulation_steps > 1:
                        # reshape data into (gradient_accumulation_steps, batch_per_node, ...)
                        # to avoid any data redistribution when sharding
                        bs_shape = (training_args.gradient_accumulation_steps,) + bs_shape

                    # preprocess batch
                    captions = [caption.decode("utf-8") for caption in batch[1]]
                    if model_args.normalize_text:
                        captions = [tn(c) for c in captions]
                    txt_inputs = tokenizer(
                        captions,
                        padding="max_length",
                        truncation=True,
                        max_length=model.config.text_config.max_position_embeddings,
                        return_tensors="np",
                    )
                    # keep only input_ids and attention_mask
                    txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
                    batch = {"pixel_values": batch[0], **txt_inputs}

                    # reshape batch
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape(bs_shape + x.shape[1:]),
                        batch,
                    )

                    # train step
                    state, train_metrics = p_train_step(state, batch, train_time)
                    local_state["step"] += 1
                    local_state["train_time"] = train_time
                    local_state["train_samples"] += training_args.batch_size_per_step

                    if local_state["step"] % training_args.logging_steps == 0 and jax.process_index() == 0:
                        metrics_logger.update_state_metrics(local_state)
                        metrics_logger.log(train_metrics, prefix="train")

                    eval_metrics = None
                    if local_state["step"] % training_args.eval_steps == 0:
                        eval_metrics = run_evaluation()
                        evaluation_ran = True

                    if local_state["step"] % training_args.save_steps == 0:
                        run_save_model(state, eval_metrics)
                        save_model_ran = True

                    # profile
                    if training_args.do_profile:
                        if profile_status == "not started" and local_state["step"] % profile_step_start == 0:
                            jax.profiler.start_trace("./profiles")
                            profile_status = "started"
                        elif profile_status == "started" and local_state["step"] % profile_step_end == 0:
                            jax.profiler.stop_trace()
                            profile_status = "stopped"

                # log final train metrics
                if train_metrics is not None:
                    metrics_logger.update_state_metrics(local_state)
                    metrics_logger.log(train_metrics, prefix="train")

                    epochs.write(
                        f"Epoch... ({epoch + 1}/{num_epochs} | Loss:"
                        f" {train_metrics['loss']}, Learning Rate:"
                        f" {train_metrics['learning_rate']})"
                    )

            # Final evaluation at the end of each epoch
            if not evaluation_ran:
                eval_metrics = run_evaluation()

            # save checkpoint after each epoch
            if not save_model_ran:
                run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()
