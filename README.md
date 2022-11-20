# CLIP-JAX

This repository is used to CLIP models from 🤗 transformers using JAX.

## Installation

```bash
pip install -e .
```

## Usage

1. Use [dataset/prepare_dataset.ipynb](dataset/prepare_dataset.ipynb) to prepare your dataset.
1. Train the model with [training/train_clip.py](training/train_clip.py).

## Supported downstream tasks

- [x] Image classification with `FlaxCLIPVisionModelForImageClassification`

## TODO

- [ ] Add guides
- [ ] Add pre-trained models
- [ ] Add more downstream tasks
