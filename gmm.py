from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import utils

from typing import Any

Array = Any


@struct.dataclass
class GaussianMixtureModel:
    mixture_logits: Array
    components_loc: Array
    components_scale_tril: Array

    @property
    def distribution(self):
        # https://github.com/tensorflow/probability/issues/1271
        mixture_distribution = tfp.distributions.Categorical(self.mixture_logits)
        components_distribution = tfp.distributions.MultivariateNormalTriL(self.components_loc,
                                                                           self.components_scale_tril)
        return tfp.distributions.MixtureSameFamily(mixture_distribution, components_distribution)

    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)

    def mean_valid_log_prob(self, samples: utils.PaddedArray):
        return jnp.sum(jax.vmap(self.log_prob, -2, -1)(samples.padded) * samples.valid_mask, -1) / samples.num_valid


def sample_gmm_training_data(key,
                             point_dim=2,
                             num_samples_minval=300,
                             num_samples_maxval=600,
                             num_components_minval=4,
                             num_components_maxval=6,
                             mixture_logits_minval=-2.0,
                             mixture_logits_maxval=2.0,
                             components_loc_minval=-4.0,
                             components_loc_maxval=4.0,
                             components_scale_minval=0.1,
                             components_scale_maxval=0.5):
    (num_samples_key, num_components_key, mixture_logits_key, components_loc_key, components_scale_key,
     components_rotation_key, sample_key) = jax.random.split(key, 7)
    num_samples = jax.random.randint(num_samples_key, (), num_samples_minval, num_samples_maxval)
    num_components = jax.random.randint(num_components_key, (), num_components_minval, num_components_maxval)
    mixture_logits = jax.random.uniform(mixture_logits_key, (num_components_maxval,),
                                        minval=mixture_logits_minval,
                                        maxval=mixture_logits_maxval)
    components_loc = jax.random.uniform(components_loc_key, (num_components_maxval, point_dim),
                                        minval=components_loc_minval,
                                        maxval=components_loc_maxval)
    components_scale = jax.random.uniform(components_scale_key, (num_components_maxval, point_dim, 1),
                                          minval=components_scale_minval,
                                          maxval=components_scale_maxval)
    components_scale_tril = jnp.swapaxes(
        jnp.linalg.qr(
            components_scale *
            jnp.linalg.qr(jax.random.normal(components_rotation_key,
                                            (num_components_maxval, point_dim, point_dim)))[0], "r"), -1, -2)

    mixture_logits = jnp.where(utils.sequence_mask(num_components, num_components_maxval), mixture_logits, -np.inf)
    gaussian_mixture_model = GaussianMixtureModel(mixture_logits, components_loc, components_scale_tril)
    samples = utils.PaddedArray(
        gaussian_mixture_model.sample(num_samples_maxval, sample_key) *
        utils.sequence_mask(num_samples, num_samples_maxval)[:, None], num_samples)
    return {"samples": samples, "ground_truth_gmm": gaussian_mixture_model}
