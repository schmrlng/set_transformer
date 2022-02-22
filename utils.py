import hashlib

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from typing import Any

Array = Any


def sequence_mask(lengths, maxlen):
    return jnp.arange(maxlen) < lengths[..., None]


@struct.dataclass
class PaddedArray:
    padded: Array
    num_valid: Array

    @property
    def valid_mask(self):
        return sequence_mask(self.num_valid, self.padded.shape[self.num_valid.ndim])


def jensen_shannon_divergence_estimate(key, distribution_p, distribution_q, num_samples=1000):
    p_key, q_key = jax.random.split(key)
    p_samples = distribution_p.sample(num_samples, p_key)
    q_samples = distribution_q.sample(num_samples, q_key)
    p_log_p = distribution_p.log_prob(p_samples)
    p_log_q = distribution_q.log_prob(p_samples)
    q_log_p = distribution_p.log_prob(q_samples)
    q_log_q = distribution_q.log_prob(q_samples)
    return (jnp.mean(p_log_p, -1) - jnp.mean(jnp.logaddexp(p_log_p, p_log_q), -1) + jnp.mean(q_log_q, -1) -
            jnp.mean(jnp.logaddexp(q_log_q, q_log_p), -1)) / 2 + np.log(2)


def deterministic_hash(string):
    h = hashlib.sha1()
    h.update(string.encode("utf-8"))
    return h.hexdigest()
