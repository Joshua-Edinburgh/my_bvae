#! /bin/sh
#========= Baseline, using FVAE
python main.py --model_type FVAE\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 30\
    --batch_size 64 --z_dim 10 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name noit_FVAE_gam30_z10_testmetric



