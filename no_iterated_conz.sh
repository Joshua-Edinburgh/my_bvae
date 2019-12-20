#! /bin/sh
#========= Baseline, using BetaVAE_H
python main.py --discrete_z False\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 10 --a_dim 40 --max_iter_per_gen 1500000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name no_iterated_test_baseline_lr5e_4



