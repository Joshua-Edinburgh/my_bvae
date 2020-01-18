#========= 3dshapes, using FVAE =======================
#python main.py --model_type FVAE --data_type 3dshapes\
#    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 20\
#    --batch_size 128 --z_dim 10 --max_iter_per_gen 1000000 --max_gen 1 \
#    --save_step 50000 --metric_step 50000 \
#    --exp_name NOIT_3D_FVAE_gam20_z10

#========= 3dshapes, using FCVAE =======================
python main.py --model_type FCVAE --data_type 3dshapes\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 20\
    --batch_size 128 --z_dim 10 --a_dim 20 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name NOIT_3D_FCVAE_gam20_z10_a20

#========= 3dshapes, using FCVAE =======================
python main.py --model_type FCVAE --data_type dsprites\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 20\
    --batch_size 128 --z_dim 10 --a_dim 20 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name NOIT_ds_FCVAE_gam20_z10_a20

#========= 3dshapes, using FVAE =======================
python main.py --model_type FVAE --data_type dsprites\
    --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 --gamma 20\
    --batch_size 128 --z_dim 10 --max_iter_per_gen 1000000 --max_gen 1 \
    --save_step 50000 --metric_step 50000 \
    --exp_name NOIT_ds_FVAE_gam20_z10

