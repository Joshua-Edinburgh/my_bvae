#! /bin/sh
#========= Baseline, using FVAE
python main.py --model_type FCVAE\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 500\
    --batch_size 128 --z_dim 8 --a_dim 40 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name noit_FCVAE_gam500_z8_R3



