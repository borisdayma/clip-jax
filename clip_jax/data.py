import random
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from einops import rearrange


@dataclass
class Dataset:
    train_folder: str = None
    valid_folder: str = None
    train_batch_size: int = 64
    valid_batch_size: int = 64
    image_size: int = 0  # no resizing if set to 0, (data should be at right dimensions)
    min_original_image_size: int = None
    max_original_aspect_ratio: float = None
    seed_dataset: int = None
    format: str = "rgb"  # rgb or lab
    key_image: str = "webp"  # name of key containing image
    mean: list[float] = (0.0, 0.0, 0.0)  # do not apply mean
    std: list[float] = (1.0, 1.0, 1.0)  # do not apply std
    train: tf.data.Dataset = field(init=False)
    valid: tf.data.Dataset = field(init=False)
    rng: tf.random.Generator = field(init=False)
    multi_hosts: bool = field(init=False)

    def __post_init__(self):
        # verify valid args
        assert self.format in ["rgb", "lab"], f"Invalid format: {self.format}"

        # define rng
        if self.seed_dataset is None:
            self.seed_dataset = random.randint(0, 2**32 - 1)
        self.rng = tf.random.Generator.from_seed(self.seed_dataset, alg="philox")

        # check if we are on multi-hosts
        self.multi_hosts = jax.process_count() > 1

        # define parsing function
        features = {
            self.key_image: tf.io.FixedLenFeature([], tf.string),
            "original_width": tf.io.FixedLenFeature([], tf.int64),
            "original_height": tf.io.FixedLenFeature([], tf.int64),
            "caption": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return (
                parsed_features[self.key_image],
                parsed_features["original_width"],
                parsed_features["original_height"],
                parsed_features["caption"],
            )

        def _filter_function(image, width, height, caption):
            # filter out images that are too small
            if self.min_original_image_size is not None and (
                tf.minimum(width, height) < self.min_original_image_size
            ):
                return False
            # filter out images that have wrong aspect ratio
            if self.max_original_aspect_ratio is not None and (
                tf.divide(tf.maximum(width, height), tf.minimum(width, height))
                > self.max_original_aspect_ratio
            ):
                return False
            return True

        def _parse_image(image, width, height, caption):
            return tfio.image.decode_webp(image)[..., :3], caption

        def _parse_no_filter(example_proto):
            # we can combine parsing functions into one
            return _parse_image(*_parse_function(example_proto))

        def _augment(image, seed):
            # create a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # apply random crop
            return tf.image.stateless_random_crop(
                image, size=[self.image_size, self.image_size, 3], seed=new_seed
            )

        # augmentation wrapper
        def _augment_wrapper(image, caption):
            seed = self.rng.make_seeds(2)[0]
            return _augment(image, seed), caption

        # center crop (for validation)
        def _center_crop(image, caption):
            return (
                tf.image.resize_with_crop_or_pad(
                    image, self.image_size, self.image_size
                ),
                caption,
            )

        # normalization
        def _normalize(image, caption):
            if self.format == "rgb":
                image = (
                    tf.cast(image, tf.float32) / 255.0
                    - tf.convert_to_tensor(self.mean, dtype=tf.float32)
                ) / tf.convert_to_tensor(self.std, dtype=tf.float32)
                image = rearrange(image, "b h w c -> b c h w")

                return image, caption
            elif self.format == "lab":
                raise NotImplementedError("LAB not implemented")

        for folder, dataset, augment, batch_size in zip(
            [self.train_folder, self.valid_folder],
            ["train", "valid"],
            [True, False],
            [self.train_batch_size, self.valid_batch_size],
        ):
            if folder is not None:
                # load files
                if "gs://" in folder:
                    if folder[-1] != "/":
                        folder += "/"
                    files = tf.io.gfile.glob(f"{folder}*.tfrecord")
                else:
                    files = [f"{Path(f)}" for f in Path(folder).glob("*.tfrecord")]
                assert len(files) > 0, f"No files found at folder: {folder}"

                # shuffle files
                if augment:
                    random.shuffle(files)

                # load dataset
                ds = tf.data.TFRecordDataset(
                    files,
                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                    if dataset == "train"
                    else None,
                )

                # non deterministic read (faster)
                if augment:
                    ignore_order = tf.data.Options()
                    ignore_order.deterministic = False
                    ds = ds.with_options(ignore_order)

                if self.multi_hosts and dataset == "train":
                    # repeat indefinitely
                    ds = ds.repeat()

                # parse dataset
                if (
                    self.min_original_image_size is None
                    and self.max_original_aspect_ratio is None
                ):
                    ds = ds.map(
                        _parse_no_filter,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )
                else:
                    ds = ds.map(
                        _parse_function,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )

                    # filter dataset
                    ds = ds.filter(_filter_function)

                    # parse image
                    ds = ds.map(
                        _parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )

                if augment:
                    ds = ds.shuffle(1000)
                    if self.image_size:
                        ds = ds.map(
                            _augment_wrapper,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        )
                elif self.image_size:
                    ds = ds.map(
                        _center_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )

                # batch, normalize and prefetch
                ds = ds.batch(batch_size, drop_remainder=True)
                ds = ds.map(
                    _normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                setattr(self, dataset, ds)


def logits_to_image(logits, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), format="rgb"):
    logits = np.asarray(logits, dtype=np.float32)
    logits = rearrange(logits, "c h w -> h w c")
    if format == "rgb":
        logits = (logits * np.asarray(std, dtype=np.float32)) + np.asarray(
            mean, dtype=np.float32
        )
        logits = logits.clip(0.0, 1.0)
    else:
        raise NotImplementedError("LAB not implemented")
    return logits
