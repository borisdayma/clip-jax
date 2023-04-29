# CLIP-JAX

This repository is used to train custom CLIP models using JAX.

## Installation

```bash
pip install clip-jax
```

Note: this package is currently under active development, install from source for latest version.

## Usage

1. Use [utils/prepare_dataset.ipynb](utils/prepare_dataset.ipynb) to prepare your dataset.
1. Train the model with [training/train.py](training/train.py).

## TODO

- [ ] Add guides: download LAION, train CLIP
- [ ] Add pre-trained models
- [ ] Add script for downstream tasks

## Features

- [x] Custom model architectures
- [x] Custom sharding strategies
- [x] Training with constrastive loss or [chunked sigmoid loss](https://arxiv.org/abs/2303.15343)
- [ ] Downstream tasks
  - [ ] Image classification with `CLIPVisionModelForImageClassification`
  - [ ] Text encoder with `CLIPTextModelForFineTuning`

## Acknowledgements

- 🤗 Hugging Face for CLIP reference implementation and training scripts
- Google [TPU Research Cloud (TRC) program](https://sites.research.google/trc/) for providing computing resources
- [Weights & Biases](https://wandb.com/) for providing the infrastructure for experiment tracking and model management

## Citations

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training}, 
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
