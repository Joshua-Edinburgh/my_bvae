#! /bin/sh
#========= Baseline, using FVAE
python main.py --model_type VQVAE\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999\
    --batch_size 128 --z_dim 256 --a_dim 256 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name noit_VQVAE_z256_a256\
    --save_gifs False



