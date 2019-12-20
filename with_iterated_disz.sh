#========= Baseline, using CVAE (discrete_z, Beta-VAE=1, Catogorical implementation, temperature annealing)
python main.py --discrete_z True --beta 1\
    --seed 1 --lr 1e-3 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 5 --a_dim 100 \
    --max_iter_per_gen 200000 --max_gen 40\
    --nb_preENDE 400 --niter_preEN 6000 --niter_preDE 2000\
    --nb_preENDE 200\
    --save_step 50000 --metric_step 50000 \
    --exp_name with_iterated_test_CVAE_EN6K_DE2K_z5

