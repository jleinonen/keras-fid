import warnings

from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import numpy as np


def update_mean_cov(mean, cov, N, batch):
	batch_N = batch.shape[0]

	x = batch
	N += batch_N
	x_norm_old = batch-mean
	mean = mean + x_norm_old.sum(axis=0)/N
	x_norm_new = batch-mean
	cov = ((N-batch_N)/N)*cov + x_norm_old.T.dot(x_norm_new)/N

	return (mean, cov, N)


def frechet_distance(mean1, cov1, mean2, cov2):
	"""Frechet distance between two multivariate Gaussians.

	Arguments:
		mean1, cov1, mean2, cov2: The means and covariances of the two
			multivariate Gaussians.

	Returns:
		The Frechet distance between the two distributions.
	"""
	
	def check_nonpositive_eigvals(l):
		nonpos = (l < 0)
		if nonpos.any():
			warnings.warn('Rank deficient covariance matrix, '
				'Frechet distance will not be accurate.', Warning)
		l[nonpos] = 0

	(l1,v1) = np.linalg.eigh(cov1)
	check_nonpositive_eigvals(l1)
	cov1_sqrt = (v1*np.sqrt(l1)).dot(v1.T)
	cov_prod = cov1_sqrt.dot(cov2).dot(cov1_sqrt)
	lp = np.linalg.eigvalsh(cov_prod)
	check_nonpositive_eigvals(lp)

	trace = l1.sum() + np.trace(cov2) - 2*np.sqrt(lp).sum()
	diff_mean = mean1-mean2
	fd = diff_mean.dot(diff_mean) + trace

	return fd


class InputIterator(object):
	def __init__(self, inputs, batch_size=64, shuffle=True, seed=None):
		self._inputs = inputs
		self._inputs_list = isinstance(inputs, list)
		self._N = self._inputs[0].shape[0] if self._inputs_list else \
			self._inputs.shape[0]
		self.batch_size = batch_size
		self._shuffle = shuffle
		self._prng = np.random.RandomState(seed=seed)
		self._next_indices = np.array([], dtype=np.uint)

	def __iter__(self):
		return self

	def __next__(self):
		while len(self._next_indices) < self.batch_size:
			next_ind = np.arange(self._N, dtype=np.uint)
			if self._shuffle:
				self._prng.shuffle(next_ind)
			self._next_indices = np.concatenate((
				self._next_indices, next_ind))

		ind = self._next_indices[:self.batch_size]
		self._next_indices = self._next_indices[self.batch_size:]

		if self._inputs_list:
			batch = [inp[ind,...] for inp in self._inputs]
		else:
			batch = self._inputs[ind,...]

		return batch


class FrechetInceptionDistance(object):
	"""Frechet Inception Distance.
	
	Class for evaluating Keras-based GAN generators using the Frechet
	Inception Distance (Heusel et al. 2017, 
	https://arxiv.org/abs/1706.08500).

	Arguments to constructor:
		generator: a Keras model trained as a GAN generator
		image_range: A tuple giving the range of values in the images output
			by the generator. This is used to rescale to the (-1,1) range
			expected by the Inception V3 network. 
		generator_postprocessing: A function, preserving the shape of the
			output, to be applied to all generator outputs for further 
			postprocessing. If None (default), no postprocessing will be
			done.

	Attributes: The arguments above all have a corresponding attribute
		with the same name that can be safely changed after initialization.

	Arguments to call:
		real_images: An 4D NumPy array of images from the training dataset,
			or a Python generator outputting training batches. The number of
			channels must be either 3 or 1 (in the latter case, the single
			channel is distributed to each of the 3 channels expected by the
			Inception network).
		generator_inputs: One of the following:
			1. A NumPy array with generator inputs, or
			2. A list of NumPy arrays (if the generator has multiple inputs)
			3. A Python generator outputting batches of generator inputs
				(either a single array or a list of arrays)
		batch_size: The size of the batches in which the data is processed.
			No effect if Python generators are passed as real_images or
			generator_inputs.
		num_batches_real: Number of batches to use to evaluate the mean and
			the covariance of the real samples.
		num_batches_gen: Number of batches to use to evaluate the mean and
			the covariance of the generated samples. If None (default), set
			equal to num_batches_real.
		shuffle: If True (default), samples are randomly selected from the
			input arrays. No effect if real_images or generator_inputs is
			a Python generator.
		seed: A random seed for shuffle (to provide reproducible results)

	Returns (call):
		The Frechet Inception Distance between the real and generated data.
	"""

	def __init__(self, generator, image_range=, 
		generator_postprocessing=None):

		self._inception_v3 = None
		self.generator = generator
		self.generator_postprocessing = generator_postprocessing
		self.image_range = image_range
		self._channels_axis = \
			-1 if K.image_data_format()=="channels_last" else -3

	def _setup_inception_network(self):
		self._inception_v3 = InceptionV3(
			include_top=False, pooling='avg')
		self._pool_size = self._inception_v3.output_shape[-1]

	def _preprocess(self, images):
		if self.image_range != (-1,1):
			images = images - self.image_range[0]
			images /= (self.image_range[1]-self.image_range[0])/2.0
			images -= 1.0
		if images.shape[self._channels_axis] == 1:
			images = np.concatenate([images]*3, axis=self._channels_axis)
		return images

	def _stats(self, inputs, input_type="real", postprocessing=None,
		batch_size=64, num_batches=128, shuffle=True, seed=None):

		mean = np.zeros(self._pool_size)
		cov = np.zeros((self._pool_size,self._pool_size))
		N = 0

		for i in range(num_batches):
			try:
				# draw a batch from generator input iterator
				batch = next(inputs)
			except TypeError:
				# assume that an array or a list of arrays was passed
				# instead
				inputs = InputIterator(inputs,
					batch_size=batch_size, shuffle=shuffle, seed=seed)
				batch = next(inputs)

			if input_type=="generated":
				batch = self.generator.predict(batch)
			if postprocessing is not None:
				batch = postprocessing(batch)
			batch = self._preprocess(batch)
			pool = self._inception_v3.predict(batch, batch_size=batch_size)

			(mean, cov, N) = update_mean_cov(mean, cov, N, pool)

		return (mean, cov)

	def __call__(self,
			real_images,
			generator_inputs,
			batch_size=64,
			num_batches_real=128,
			num_batches_gen=None,
			shuffle=True,
			seed=None
		):

		if self._inception_v3 is None:
			self._setup_inception_network()

		(real_mean, real_cov) = self._stats(real_images,
			"real", batch_size=batch_size, num_batches=num_batches_real,
			shuffle=shuffle, seed=seed)
		if num_batches_gen is None:
			num_batches_gen = num_batches_real
		(gen_mean, gen_cov) = self._stats(generator_inputs,
			"generated", batch_size=batch_size, num_batches=num_batches_gen,
			postprocessing=self.generator_postprocessing,
			shuffle=shuffle, seed=seed)

		return frechet_distance(real_mean, real_cov, gen_mean, gen_cov)
