#! /bin/sh

python main.py --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 10 --max_iter_per_gen 1500000 --max_gen 1\
    --exp_name no_iterated_test
