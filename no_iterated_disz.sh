#========= Baseline, using CVAE (discrete_z, Beta-VAE=1, Catogorical implementation, temperature annealing)
python main.py --discrete_z True --beta 1\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 10 --a_dim 20 --max_iter_per_gen 1500000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name no_iterated_test_CVAE_a20_beta1

#========= Baseline, using CVAE (discrete_z, Beta-VAE=1, Catogorical implementation, temperature annealing)
python main.py --discrete_z True --beta 1\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --batch_size 64 --z_dim 5 --a_dim 20 --max_iter_per_gen 1500000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name no_iterated_test_CVAE_a200_beta1_z5
