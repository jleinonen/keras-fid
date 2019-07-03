## Frechet Inception Distance for Keras models

This module contains an implementation of the Frechet Inception Distance
metric for Keras-based generative adversarial network (GAN) generators.

### Usage

A basic example:
```
import fid

generator = ... # Your code for creating the GAN generator
noise = ... # Your generator inputs (usually random noise)
real_images = ... # Your training dataset

# change (0,1) to the range of values in your dataset
fd = fid.FrechetInceptionDistance(generator, (0,1)) 
gan_fid = fd(real_images, noise)
```

If you already have the means and covariances:
```
gan_fid = fid.frechet_distance(mean1, cov1, mean2, cov2)
```
