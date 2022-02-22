from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf

import train_eval

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config.py")


def main(_):
    tf.config.set_visible_devices([], "GPU")
    logging.info(FLAGS.config)
    train_eval.train_and_evaluate(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
