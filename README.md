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
    --image_size 256 --resize_mode center_crop --skip_bbox_blurring --no_resize_only_if_bigger \
    --encode_format webp --output_format tfrecord
```

Alternatively, you can use your own dataset. In that case you should use [img2dataset](https://github.com/rom1504/img2dataset) with `output_format="tfrecord"`.

### Train a model

Use [`training/train.py`](training/train.py) to train a model:

Here is an example command to train a model on a TPU v3-8:

```bash
python train.py \
    --assert_TPU_available \
    --config_name ../configs/small-patch16.json --dtype float32 \
    --do_train --train_folder gs://my_bucket/datacomp/small/shards \
    --output_dir gs://my_bucket/clip_model/$(date +"%Y%m%d%H%M%S") \
    --num_train_epochs 10 \
    --tokenizer_name openai/clip-vit-base-patch32 \
    --batch_size_per_node 4096 --gradient_accumulation_steps 1 \
    --learning_rate 0.00001 --warmup_steps 2000 --lr_offset 0 \
    --optim distributed_shampoo --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
    --block_size_text 512 --block_size_vision 512 --nesterov \
    --graft_type rmsprop_normalized --preconditioning_compute_steps 20 \
    --mp_devices 1 --shard_shampoo_across 2d \
    --activation_partitioning_dims 1 --parameter_partitioning_dims 1 \
    --loss_type sigmoid \
    --gradient_checkpointing \
    --unroll 100 \
    --logging_steps 100 --save_steps 5000
```

### Use a trained model

Refer to [`utils/demo.ipynb`](utils/demo.ipynb).

TODO: train a benchmark model and load it from the hub.

### Downstream tasks

TODO: add demo's:

- [ ] Image classification with `CLIPVisionModelForImageClassification`
- [ ] Text encoder with `CLIPTextModelForFineTuning`

## Acknowledgements

- [Lucas Beyer](https://twitter.com/giffmana) for helping with clarifications on the [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) paper
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
