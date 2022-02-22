from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp

import attention
import gmm
import positional_encoding


class GMMSetTransformer(nn.Module):
    num_components: int = 6

    input_encoding: str = "fourier_feature"
    input_encoding_scale: float = 0.1
    num_encoder_set_attention_blocks: int = 3
    num_decoder_set_attention_blocks: int = 2
    hidden_dim: int = 128
    num_inducing_points: int = 32
    num_heads: int = 8

    @nn.compact
    def __call__(self, samples):
        point_dim = samples.padded.shape[-1]
        mask = samples.valid_mask

        if self.input_encoding == "fourier_feature":
            x = positional_encoding.FourierFeatureEncoding(self.hidden_dim, self.input_encoding_scale)(samples.padded)
        elif self.input_encoding == "sinusoidal":
            x = positional_encoding.SinusouidalEncoding(self.hidden_dim)(samples.padded)
        elif self.input_encoding == "affine":
            x = nn.Dense(self.hidden_dim)(samples.padded)
        else:
            raise ValueError(f"Unknown input encoding: {self.input_encoding}")

        for _ in range(self.num_encoder_set_attention_blocks):
            x = attention.InducedSetAttentionBlock(self.num_inducing_points, self.num_heads)(x, mask)
        x = attention.PoolingByMultiHeadAttention(self.num_components, self.num_heads)(x, mask)
        for _ in range(self.num_decoder_set_attention_blocks):
            x = attention.SetAttentionBlock(self.num_heads)(x)

        mixture_logits = nn.Dense(1)(x)[..., 0]
        components_loc = nn.Dense(point_dim)(x)
        components_scale_tril = tfp.bijectors.FillScaleTriL()(nn.Dense(point_dim * (point_dim + 1) // 2)(x))
        return gmm.GaussianMixtureModel(mixture_logits, components_loc, components_scale_tril)
