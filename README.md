# my_bvae
my test version of beta-vae

To run the code, we should download dataset (only the .npz file) at 
<url>https://github.com/deepmind/dsprites-dataset<url> and save this file as following:

```
.  
└── data  
    └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz  
```

To do list:

+ [x] Make a simple beta-vae version that can run.
+ [ ] Implement different metrics that can measure the quality of embeddings
  + [ ] Qualitative, i.e., direct show the figure or .gif
  + [ ] Quantitive 1 ($\rho$), topological similarity
  + [ ] Quantitive 2 (matrix $R$), from Cian's paper
+ [ ] Observe the relationship between quantitive metric 1 and 2
+ [ ] Implement the encoder pre-train phase, and observe the learning curves of embeddings with different values of $\rho$ or $R$.
+ [ ] Try to embed iterated learning to this framework
