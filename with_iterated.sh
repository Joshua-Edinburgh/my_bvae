#! /bin/sh

python main.py --seed 2 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 5 --a_dim 40 --max_iter_per_gen 10000 --max_gen 100\
    --nb_preENDE 100 --niter_preEN 2000 --niter_preDE 800 --metric_step 5000\
    --exp_name iteration_test
