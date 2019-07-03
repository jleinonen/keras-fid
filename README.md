## Frechet Inception Distance for Keras GANs

This module contains an implementation of the Frechet Inception Distance (FID)
metric for Keras-based generative adversarial network (GAN) generators.

The FID is defined here: https://arxiv.org/abs/1706.08500

### Usage

A basic example:
```python
import fid

generator = ... # Your code for creating the GAN generator
noise = ... # Your generator inputs (usually random noise)
real_images = ... # Your training dataset

# change (0,1) to the range of values in your dataset
fd = fid.FrechetInceptionDistance(generator, (0,1)) 
gan_fid = fd(real_images, noise)
```

If you already have the means and covariances:
```python
gan_fid = fid.frechet_distance(mean1, cov1, mean2, cov2)
```

### More information

See the docstrings for the `fid.FrechetInceptionDistance` class and
the `fid.frechet_distance` function for more detailed documentation.
