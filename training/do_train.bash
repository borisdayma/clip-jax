UUID=$(uuidgen | cut -c1-8)

WANDB_NOTES=$(cat ~/instance_name) python train.py \
    --assert_TPU_available \
    --unroll \
    --output_dir gs://craiyon_models_us_central2/clip/$UUID \
    --config_name ../configs/tiny-debug.json --dtype float32 \
    --key_image webp_256 \
    --train_folder train_files.pkl --valid_folder valid_files.pkl \
    --tokenizer_name craiyon/clip/craiyon_tokenizer:v0 \
    --do_train --do_eval \
    --batch_size_per_node 64 --gradient_accumulation_steps 2 \
    --learning_rate 0.00001 --warmup_steps 0 --lr_offset 0 \
    --optim distributed_shampoo --beta1 0.9 --beta2 0.99 --weight_decay 0.0 \
    --block_size_text 1024 --block_size_vision 1024 --nesterov \
    --graft_type rmsprop_normalized --preconditioning_compute_steps 20 \
    --mp_devices 1 --shard_shampoo_across data \
    --activation_partitioning_dims 1 --parameter_partitioning_dims 1 \
    --logging_steps 20 --eval_steps 100 --save_steps 110 --do_test_steps 120