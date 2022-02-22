import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.num_components = 6

    config.input_encoding = "fourier_feature"
    config.input_encoding_scale = 0.1
    config.num_encoder_set_attention_blocks = 2
    config.num_decoder_set_attention_blocks = 1
    config.hidden_dim = 128
    config.num_inducing_points = 32
    config.num_heads = 8

    config.training_prng_key = 1234
    config.eval_prng_key = 9876
    config.learning_rate = 3e-4
    config.batch_size = 64
    config.num_train_steps = 40000
    config.steps_per_epoch = 200

    return config
