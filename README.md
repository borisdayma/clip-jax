# CLIP-JAX

This repository is used to train custom [CLIP models](https://arxiv.org/abs/2103.00020) with JAX:

- custom model architectures
- custom sharding strategies
- training with constrastive loss or [chunked sigmoid loss](https://arxiv.org/abs/2303.15343)
- downstream fine-tuning

## Installation

```bash
pip install clip-jax
```

Note: this package is currently under active development, install from source for latest version.

## Usage

### Download training data

You can download training data from [DataComp](https://github.com/mlfoundations/datacomp):

```bash
# clone and install datacomp

# download data
python download_upstream.py \
    --scale small --data_dir gs://my_bucket/datacomp/small metadata_dir metadata \
    --image_size 256 --resize_mode center_crop --skip_bbox_blurring \
    --output_format tfrecord
```

Alternatively, you can use your own dataset. In that case you should use [img2dataset](https://github.com/rom1504/img2dataset) with `output_format="tfrecord"`.

### Train a model

TODO

### Use a trained model

TODO

### Downstream tasks

TODO:

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
