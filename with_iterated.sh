#! /bin/sh

python main.py --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 10 --max_iter_per_gen 40000 --max_gen 40\
    --nb_preENDE 100 --niter_preEN 5000 --niter_preDE 5000\
    --exp_name iteration_test
