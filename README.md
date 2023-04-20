# CLIP-JAX

This repository is used to train CLIP models from 🤗 transformers using JAX.

## Installation

```bash
pip install clip-jax
```

Note: this package is under active development, install from source for latest version.

## Usage

1. Use [utils/prepare_dataset.ipynb](utils/prepare_dataset.ipynb) to prepare your dataset.
1. Train the model with [training/train.py](training/train.py).

## Supported downstream tasks

- [ ] Image classification with `CLIPVisionModelForImageClassification`
- [ ] Text encoder with `CLIPTextModelForFineTuning`

## TODO

- [ ] Add guides: download LAION, train CLIP
- [ ] Add pre-trained models
- [ ] Add script for downstream tasks
