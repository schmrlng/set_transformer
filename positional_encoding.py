from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from typing import Any

Array = Any


class SinusouidalEncoding(nn.Module):
    features: int
    include_identity: bool = True

    @nn.compact
    def __call__(self, x: Array):
        max_degree = -(self.features // (-2 * x.shape[-1]))
        scales = np.array([2**i for i in range(max_degree)])
        a = jnp.reshape(x[..., None, :] * scales[:, None], x.shape[:-1] + (-1,))
        return jnp.concatenate(
            ([x] if self.include_identity else []) + [jnp.sin(jnp.concatenate([a, a + np.pi / 2], -1))],
            -1)[..., :self.features]


class FourierFeatureEncoding(nn.Module):
    features: int
    scale: float = 10.0
    include_identity: bool = True

    @nn.compact
    def __call__(self, x: Array):
        mapping = self.variable(
            "input_encoding", "mapping",
            lambda shape, scale: 2 * np.pi * scale * jax.random.normal(self.make_rng("params"), shape),
            (x.shape[-1], self.features // 2), self.scale)
        a = x @ mapping.value
        return jnp.concatenate(
            ([x] if self.include_identity else []) + [jnp.sin(jnp.concatenate([a, a + np.pi / 2], -1))],
            -1)[..., :self.features]
