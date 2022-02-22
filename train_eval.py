import functools
import os

from absl import logging
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg
import numpy as np
import optax

import gmm
import utils
import models

from typing import Any

Array = Any


def compute_metrics(output_gmm, example):
    return {
        "nll":
            -output_gmm.mean_valid_log_prob(example["samples"]),
        "nll_diff":
            -output_gmm.mean_valid_log_prob(example["samples"]) +
            example["ground_truth_gmm"].mean_valid_log_prob(example["samples"]),
        "jsd":
            utils.jensen_shannon_divergence_estimate(jax.random.PRNGKey(0), output_gmm, example["ground_truth_gmm"])
    }


class TrainState(train_state.TrainState):
    non_trainable_variables: Any
    next_train_step_key: Array

    def __call__(self, samples, params=None):
        return self.apply_fn({
            **self.non_trainable_variables, "params": self.params if params is None else params
        }, samples)


@jax.jit
def apply_model(state, samples, params=None):
    return state(samples, params)


def create_train_state(key, config):
    model = models.GMMSetTransformer(
        config.num_components,
        config.input_encoding,
        config.input_encoding_scale,
        config.num_encoder_set_attention_blocks,
        config.num_decoder_set_attention_blocks,
        config.hidden_dim,
        config.num_inducing_points,
        config.num_heads,
    )
    init_key, next_train_step_key = jax.random.split(key)
    example = gmm.sample_gmm_training_data(jax.random.PRNGKey(0))
    non_trainable_variables, params = model.init(init_key, example["samples"]).pop("params")
    return TrainState.create(apply_fn=model.apply,
                             params=params,
                             tx=optax.adam(config.learning_rate),
                             non_trainable_variables=non_trainable_variables,
                             next_train_step_key=next_train_step_key)


@jax.jit
def train_step(state, batch):

    def loss_fn(params):
        output_gmm = state(batch["samples"], params)
        return -jnp.mean(output_gmm.mean_valid_log_prob(batch["samples"])), output_gmm

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, output_gmm), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = jax.vmap(compute_metrics)(output_gmm, batch)
    return state, metrics


@jax.jit
def compute_contour_XYZ(output_gmm):
    XY = jnp.stack(jnp.meshgrid(jnp.linspace(-6, 6, 100), jnp.linspace(-6, 6, 100)), -1)
    return XY[..., 0], XY[..., 1], output_gmm.log_prob(XY)


def figure_to_image(fig):
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]


def eval_image(output_gmm, example):
    fig = plt.Figure(figsize=(6, 6), dpi=72)
    ax = fig.add_subplot(111)
    ax.scatter(*example["samples"].padded[:example["samples"].num_valid].T)
    ax.contour(*compute_contour_XYZ(output_gmm), levels=50, cmap="jet")
    ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel=None, ylabel=None)
    return figure_to_image(fig)


@functools.partial(jax.jit, static_argnums=1)
def sample_training_batch(key, batch_size):
    return jax.vmap(lambda key: gmm.sample_gmm_training_data(key))(jax.random.split(key, batch_size))


def train_epoch(state, num_steps, batch_size):
    batch_metrics = []
    for i in range(num_steps):
        next_train_step_key, batch_key = jax.random.split(state.next_train_step_key)
        state = state.replace(next_train_step_key=next_train_step_key)
        batch = sample_training_batch(batch_key, batch_size)
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)
    epoch_metrics = jax.tree_map(lambda *x: np.mean(np.concatenate(x)), *batch_metrics)
    return state, epoch_metrics


def train_and_evaluate(config, checkpoint_dir=None, checkpoint_dir_prefix=None):
    if checkpoint_dir is None:
        if checkpoint_dir_prefix is None:
            checkpoint_dir_prefix = "checkpoints"
        checkpoint_dir = os.path.join(checkpoint_dir_prefix, utils.deterministic_hash(config.to_json()))
    state = create_train_state(jax.random.PRNGKey(config.training_prng_key), config)
    state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    eval_batch = sample_training_batch(jax.random.PRNGKey(config.eval_prng_key), config.batch_size)

    summary_writer = tensorboard.SummaryWriter(checkpoint_dir)
    summary_writer.hparams(dict(config))

    while state.step < config.num_train_steps:
        state, epoch_metrics = train_epoch(state, config.steps_per_epoch, config.batch_size)
        checkpoints.save_checkpoint(checkpoint_dir, state, state.step)
        for logger in (logging.info, print):
            logger(f"Step {state.step}: NLL={epoch_metrics['nll']:.4f}, JSD={epoch_metrics['jsd']:.4f}")
        for k, v in epoch_metrics.items():
            summary_writer.scalar(f"train_{k}", v, state.step)
        eval_output_gmm = apply_model(state, eval_batch["samples"])
        summary_writer.image("eval_plots",
                             np.stack([
                                 eval_image(*jax.tree_map(lambda x: x[i], (eval_output_gmm, eval_batch)))
                                 for i in range(min(config.batch_size, 12))
                             ]),
                             state.step,
                             max_outputs=12)

    summary_writer.flush()
    return state
