import hashlib
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
    key_caption_2: str = None  # name of 2nd key containing captions for chat templates
    key_assistant: str = None  # name of key when using chat template data
    key_assistant_2: str = None  # name of 2nd key when using chat template data
    key_class: str = None  # used for classification
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
        if self.key_caption_2:
            features[self.key_caption_2] = tf.io.FixedLenFeature([], tf.string, default_value="")
        if self.key_assistant:
            features[self.key_assistant] = tf.io.FixedLenFeature([], tf.string, default_value="")
        if self.key_assistant_2:
            features[self.key_assistant_2] = tf.io.FixedLenFeature([], tf.string, default_value="")
        if self.key_class:
            features[self.key_class] = tf.io.FixedLenFeature([], tf.int64, default_value=0)

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return (
                parsed_features[self.key_image],
                parsed_features["original_width"],
                parsed_features["original_height"],
                parsed_features[self.key_caption],
                parsed_features[self.key_caption_2] if self.key_caption_2 else None,
                parsed_features[self.key_assistant] if self.key_assistant else None,
                parsed_features[self.key_assistant_2] if self.key_assistant_2 else None,
                parsed_features[self.key_class] if self.key_class else None,
            )

        def _filter_function(
            image,
            width,
            height,
            caption,
            caption_2,
            caption_assistant,
            caption_assistant_2,
            class_id,
        ):
            # filter out images that are too small
            if self.min_original_image_size is not None and (tf.minimum(width, height) < self.min_original_image_size):
                return False
            # filter out images that have wrong aspect ratio
            if self.max_original_aspect_ratio is not None and (
                tf.divide(tf.maximum(width, height), tf.minimum(width, height)) > self.max_original_aspect_ratio
            ):
                return False
            return True

        def _parse_image(
            image,
            width,
            height,
            caption,
            caption_2,
            caption_assistant,
            caption_assistant_2,
            class_id,
        ):
            return (
                tfio.image.decode_webp(image)[..., :3],
                caption,
                caption_2,
                caption_assistant,
                caption_assistant_2,
                class_id,
            )

        def _parse_no_filter(example_proto):
            # we can combine parsing functions into one
            return _parse_image(*_parse_function(example_proto))

        def _augment_crop(image, seed):
            # create a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # apply random crop
            return tf.image.stateless_random_crop(
                image,
                size=[self.image_crop_size, self.image_crop_size, 3],
                seed=new_seed,
            )

        # augmentation wrapper
        def _augment_crop_wrapper(image, caption, caption_2, caption_assistant, caption_assistant_2, class_id):
            seed = self.rng.make_seeds(2)[0]
            return (
                _augment_crop(image, seed),
                caption,
                caption_2,
                caption_assistant,
                caption_assistant_2,
                class_id,
            )

        # center crop (for validation)
        def _center_crop(image, caption, caption_2, caption_assistant, caption_assistant_2, class_id):
            return (
                tf.image.resize_with_crop_or_pad(image, self.image_crop_size, self.image_crop_size),
                caption,
                caption_2,
                caption_assistant,
                caption_assistant_2,
                class_id,
            )

        def _resize(image, caption, caption_2, caption_assistant, caption_assistant_2, class_id):
            # NOTE: area as we will typically be downsampling
            return (
                tf.image.resize(
                    image,
                    [self.image_crop_resize, self.image_crop_resize],
                    method="area",
                ),
                caption,
                caption_2,
                caption_assistant,
                caption_assistant_2,
                class_id,
            )

        # normalization
        def _normalize(image, caption, caption_2, caption_assistant, caption_assistant_2, class_id):
            if self.format == "rgb":
                image = (
                    tf.cast(image, tf.float32) / 255.0 - tf.convert_to_tensor([self.mean], dtype=tf.float32)
                ) / tf.convert_to_tensor([self.std], dtype=tf.float32)
            elif self.format == "lab":
                raise NotImplementedError("LAB not implemented")

            res = {"images": image, "captions": caption}
            if caption_2 is not None:
                res["captions_2"] = caption_2
            if caption_assistant is not None:
                res["captions_assistant"] = caption_assistant
            if caption_assistant_2 is not None:
                res["captions_assistant_2"] = caption_assistant_2
            if class_id is not None:
                res["class_id"] = class_id
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


def preprocess_batch(
    batch,
    tokenizer,
    max_length,
    is_decoder,
    is_prediction_batch=False,
    is_validation_batch=False,
):
    # for classification
    if "class_id" in batch.keys():
        batch = {k: v for k, v in batch.items() if k in ["class_id", "images"]}
        return batch
    # preprocess batch
    captions = [" ".join(caption.decode("utf-8").strip().split()) for caption in batch["captions"]]
    captions_assistant = batch.get("captions_assistant", None)
    if captions_assistant is not None:
        captions_2 = batch.get("captions_2", None)
        captions_assistant_2 = batch.get("captions_assistant_2", None)
        captions_assistant = [" ".join(caption.decode("utf-8").strip().split()) for caption in captions_assistant]
        # create random "choice" that can be leveraged by template
        if is_prediction_batch or is_validation_batch or (captions_assistant_2 is None and captions_2 is None):
            # alternate between 0 and 1
            choices = [i % 2 for i in range(len(captions))]
        else:
            # alternate at each epoch
            choices = [choiceDataset.batch_to_choice(pixel_values) for pixel_values in batch["images"]]
        if captions_2 is not None:
            captions_2 = [" ".join(caption.decode("utf-8").strip().split()) for caption in captions_2]
        else:
            captions_2 = [None] * len(captions_assistant)
        if captions_assistant_2 is not None:
            captions_assistant_2 = [
                " ".join(caption.decode("utf-8").strip().split()) for caption in captions_assistant_2
            ]
        else:
            captions_assistant_2 = [None] * len(captions_assistant)
        messages = [
            [
                {
                    "role": "user",
                    "content": caption,
                    "content_2": caption_2,
                    "choice": choice,
                },
                {
                    "role": "assistant",
                    "content": caption_assistant,
                    "content_2": caption_assistant_2,
                    "choice": choice,
                },
            ]
            for caption, caption_2, caption_assistant, caption_assistant_2, choice in zip(
                captions, captions_2, captions_assistant, captions_assistant_2, choices
            )
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
        txt_inputs_only = tokenizer.apply_chat_template(
            [msg[:1] for msg in messages],
            tokenize=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
            add_generation_prompt=False,
            return_dict=True,
        )
        label_mask = np.logical_not(txt_inputs_only.attention_mask) * txt_inputs.attention_mask
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
    if is_prediction_batch and captions_assistant is not None:
        # for prediction we just want inputs
        txt_inputs = {
            "input_ids": txt_inputs_only.input_ids,
            "attention_mask": txt_inputs_only.attention_mask,
            "labels": txt_inputs.input_ids,
        }
    else:
        txt_inputs = {k: txt_inputs[k] for k in ["input_ids", "attention_mask"]}
        # add labels for decoder
        if is_decoder:
            txt_inputs["labels"] = shift_tokens_left(txt_inputs["input_ids"], pad_token_id=tokenizer.pad_token_id)
            txt_inputs["label_mask"] = shift_tokens_left(label_mask, pad_token_id=0)
    if captions_assistant is not None:
        # get vision position ids
        target_id = tokenizer.unk_token_id  # TODO: configure it
        vision_start_ids = np.where(txt_inputs["input_ids"] == target_id, 1, 0).argmax(axis=-1)
        txt_inputs["vision_start_ids"] = vision_start_ids
    batch = {"pixel_values": batch["images"], **txt_inputs}
    return batch


class ChoiceDataset:
    def __init__(self):
        self.last_choice = {}

    def batch_to_choice(self, pixel_values):
        md5 = hashlib.md5(pixel_values).hexdigest()
        self.last_choice[md5] = (self.last_choice.get(md5, random.randint(0, 1)) + 1) % 2
        return self.last_choice[md5]


choiceDataset = ChoiceDataset()
