import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


@dataclass
class Dataset:
    train_folder: str = None
    valid_folder: str = None
    train_batch_size: int = 64
    valid_batch_size: int = 64
    image_crop_size: int = None  # crops image, no cropping if set to 0, (data should be at right dimensions)
    image_crop_resize: int = None  # resize cropped image to a fixed size
    min_original_image_size: int = None
    max_original_aspect_ratio: float = None
    seed_dataset: int = None
    format: str = "rgb"  # rgb or lab
    key_image: str = "webp"  # name of key containing image
    key_caption: str = "caption"  # name of key containing captions
    key_assistant: str = None  # name of key when using chat template data
    mean: list[float] = (0.5, 0.5, 0.5)  # rescale between -1 and 1 by default
    std: list[float] = (0.5, 0.5, 0.5)  # rescale between -1 and 1 by default
    _train: tf.data.Dataset = field(init=False)
    _valid: tf.data.Dataset = field(init=False)
    rng: tf.random.Generator = field(init=False)
    multi_hosts: bool = field(init=False)
    process_count: int = field(init=False)  # number of groups for validation set (multi-host)
    process_index: int = field(init=False)  # group number to use for validation set (multi-host)

    def __post_init__(self):
        # verify valid args
        assert self.format in ["rgb", "lab"], f"Invalid format: {self.format}"

        # define rng
        if self.seed_dataset is None:
            self.seed_dataset = random.randint(0, 2**32 - 1)
        self.rng = tf.random.Generator.from_seed(self.seed_dataset, alg="philox")

        # check if we are on multi-hosts
        self.multi_hosts = jax.process_count() > 1
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()

        # define parsing function
        features = {
            self.key_image: tf.io.FixedLenFeature([], tf.string),
            "original_width": tf.io.FixedLenFeature([], tf.int64),
            "original_height": tf.io.FixedLenFeature([], tf.int64),
            self.key_caption: tf.io.FixedLenFeature([], tf.string, default_value=""),
        }
        if self.key_assistant:
            features[self.key_assistant] = tf.io.FixedLenFeature([], tf.string, default_value="")

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return (
                parsed_features[self.key_image],
                parsed_features["original_width"],
                parsed_features["original_height"],
                parsed_features[self.key_caption],
                parsed_features[self.key_assistant] if self.key_assistant else None,
            )

        def _filter_function(image, width, height, caption, caption_assistant):
            # filter out images that are too small
            if self.min_original_image_size is not None and (tf.minimum(width, height) < self.min_original_image_size):
                return False
            # filter out images that have wrong aspect ratio
            if self.max_original_aspect_ratio is not None and (
                tf.divide(tf.maximum(width, height), tf.minimum(width, height)) > self.max_original_aspect_ratio
            ):
                return False
            return True

        def _parse_image(image, width, height, caption, caption_assistant):
            return tfio.image.decode_webp(image)[..., :3], caption, caption_assistant

        def _parse_no_filter(example_proto):
            # we can combine parsing functions into one
            return _parse_image(*_parse_function(example_proto))

        def _augment_crop(image, seed):
            # create a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # apply random crop
            return tf.image.stateless_random_crop(
                image, size=[self.image_crop_size, self.image_crop_size, 3], seed=new_seed
            )

        # augmentation wrapper
        def _augment_crop_wrapper(image, caption, caption_assistant):
            seed = self.rng.make_seeds(2)[0]
            return _augment_crop(image, seed), caption, caption_assistant

        # center crop (for validation)
        def _center_crop(image, caption, caption_assistant):
            return (
                tf.image.resize_with_crop_or_pad(image, self.image_crop_size, self.image_crop_size),
                caption,
                caption_assistant,
            )

        def _resize(image, caption, caption_assistant):
            # NOTE: area as we will typically be downsampling
            return (
                tf.image.resize(image, [self.image_crop_resize, self.image_crop_resize], method="area"),
                caption,
                caption_assistant,
            )

        # normalization
        def _normalize(image, caption, caption_assistant):
            if self.format == "rgb":
                image = (
                    tf.cast(image, tf.float32) / 255.0 - tf.convert_to_tensor([self.mean], dtype=tf.float32)
                ) / tf.convert_to_tensor([self.std], dtype=tf.float32)
            elif self.format == "lab":
                raise NotImplementedError("LAB not implemented")

            res = {"images": image, "captions": caption}
            if caption_assistant[0] is not None:
                res["captions_assistant"] = caption_assistant
            return res

        for folder, dataset, augment, batch_size in zip(
            [self.train_folder, self.valid_folder],
            ["_train", "_valid"],
            [True, False],
            [self.train_batch_size, self.valid_batch_size],
        ):
            if folder is not None:
                # load files
                if folder.endswith(".pkl"):
                    with open(folder, "rb") as f:
                        files = pickle.load(f)
                elif "gs://" in folder:
                    if folder[-1] != "/":
                        folder += "/"
                    files = tf.io.gfile.glob(f"{folder}*.tfrecord")
                else:
                    files = [f"{Path(f)}" for f in Path(folder).glob("*.tfrecord")]
                assert len(files) > 0, f"No files found at folder: {folder}"

                # sort files
                files = sorted(files)

                # shuffle files and select subset
                if augment:
                    files = files[self.process_index :: self.process_count]
                    random.shuffle(files)

                # load dataset
                ds = tf.data.TFRecordDataset(
                    files,
                    num_parallel_reads=tf.data.experimental.AUTOTUNE,
                )

                # non deterministic read (faster)
                if augment:
                    ignore_order = tf.data.Options()
                    ignore_order.deterministic = False
                    ds = ds.with_options(ignore_order)

                    if self.multi_hosts:
                        # repeat indefinitely
                        ds = ds.repeat()
                        # shuffle files
                        ds = ds.shuffle(len(files))

                # parse dataset
                if self.min_original_image_size is None and self.max_original_aspect_ratio is None:
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
                    ds = ds.map(_parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                if augment:
                    ds = ds.shuffle(1000)
                    if self.image_crop_size:
                        ds = ds.map(
                            _augment_crop_wrapper,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        )
                elif self.image_crop_size:
                    ds = ds.map(_center_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                # resize
                if self.image_crop_resize:
                    ds = ds.map(_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                # batch, normalize and prefetch
                ds = ds.batch(batch_size, drop_remainder=True)
                ds = ds.map(_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                setattr(self, dataset, ds)

    @property
    def train(self):
        return self._train.as_numpy_iterator()

    @property
    def valid(self):
        if not self.multi_hosts:
            yield from self._valid.as_numpy_iterator()
        else:
            # we need to return only a subset of the validation set
            for i, batch in enumerate(self._valid.as_numpy_iterator()):
                if i % self.process_count == self.process_index:
                    # this is the batch to yield for this host
                    batch_group = batch
                if i % self.process_count == (self.process_count - 1):
                    # all nodes have a batch
                    yield batch_group


def logits_to_image(logits, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), format="rgb"):
    if format == "rgb":
        logits = (logits * np.asarray(std, dtype=np.float32)) + np.asarray(mean, dtype=np.float32)
        logits = np.asarray(logits * 255.0, dtype=np.uint8)
        logits = logits.clip(0, 255)
    else:
        raise NotImplementedError("LAB not implemented")
    return logits


def image_to_logits(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), format="rgb"):
    image = np.asarray(image)
    if format == "rgb":
        image = (image / 255.0 - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    else:
        raise NotImplementedError("LAB not implemented")
    return image


def shift_tokens_left(logits, pad_token_id):
    shifted_logits = np.zeros_like(logits)
    shifted_logits[:, :-1] = logits[:, 1:]
    shifted_logits[:, -1] = pad_token_id
    return shifted_logits


def preprocess_batch(batch, tokenizer, max_length, is_decoder):
    # preprocess batch
    captions = [caption.decode("utf-8") for caption in batch["captions"]]
    captions_assistant = batch.get("captions_assistant", None)
    if captions_assistant is not None:
        captions_assistant = [caption.decode("utf-8") for caption in captions_assistant]
        messages = [
            [{"role": "user", "content": caption}, {"role": "assistant", "content": caption_assistant}]
            for caption, caption_assistant in zip(captions, captions_assistant)
        ]
        txt_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
            return_dict=True,
        )
        txt_inputs_only_mask = tokenizer.apply_chat_template(
            [msg[:1] for msg in messages],
            tokenize=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
            add_generation_prompt=False,
            return_dict=True,
        ).attention_mask
        label_mask = np.logical_not(txt_inputs_only_mask) * txt_inputs.attention_mask
        # get vision position ids
        target_id = tokenizer.unk_token_id  # TODO: configure it
        vision_start_ids = np.where(txt_inputs.input_ids == target_id, 1, 0).argmax(axis=-1)
    else:
        txt_inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        label_mask = txt_inputs.attention_mask
        vision_start_ids = None
    # keep only input_ids and attention_mask
    txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
    # add labels for decoder
    if is_decoder:
        txt_inputs["labels"] = shift_tokens_left(txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id)
        txt_inputs["label_mask"] = shift_tokens_left(label_mask, pad_token_id=0)
    batch = {"pixel_values": batch["images"], **txt_inputs}
    if vision_start_ids is not None:
        batch["vision_start_ids"] = vision_start_ids
    return batch
