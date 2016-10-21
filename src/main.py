import tensorflow as tf
import time
import os
import utils.ptb_reader as ptb_reader
from models.adaptive_computation_time import adaptive_computation_time
from tensorflow.contrib.learn.python.learn.monitors import ValidationMonitor
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.learn.python.learn.estimators import Estimator

# TODO Pass the initial state as the final output state
# TODO Try using dynamic rnn for sentence-wise rnn

tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("data_path", 'data/', "data_path (/data)")
tf.flags.DEFINE_integer("batch_size", 10, "batch size (defaults to 32)")
tf.flags.DEFINE_integer("num_time_steps", 20, "number of steps to unroll RNN and apply BPTT (20)")
tf.flags.DEFINE_boolean("debug", True, "set to true for loading smaller dataset")
tf.flags.DEFINE_float("init_scale", 0.1, "Weights are initialized between [-init_scale, init_scale] (0.1)")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_integer("max_grad_norm", 5, "Gradient Clipping (5)")
tf.flags.DEFINE_integer("hidden_size", 200, "Size of hidden layer (200)")
tf.flags.DEFINE_integer("max_epoch", 4, "Max number of epochs (4)")
tf.flags.DEFINE_integer("max_max_epoch", 13, "Max max number of epochs (13)")
tf.flags.DEFINE_integer("keep_prob", 1.0, "Dropout keep probablity (1.0)")
tf.flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay (0.5)")
tf.flags.DEFINE_integer("vocab_size", 10000, "Vocabulary size to use (10000)")
tf.flags.DEFINE_integer("max_computation", 50, "Max number of computations per time steps (50)")
tf.flags.DEFINE_float("epsilon", 0.01, 'such that hault happend when 1-epsilon is reached at every time step (0.1)')
tf.flags.DEFINE_float("ponder_time_penalty", 0.01, "Pondering penalty (0.01)")
tf.flags.DEFINE_boolean("use_lstm", False, "whether to use LSTM (False)")

# config flags
tf.flags.DEFINE_integer("save_checkpoints_secs", 60, "Save checkpoints every this many seconds")
tf.flags.DEFINE_integer("save_summary_steps", 10, "Save summaries every this many steps")
tf.flags.DEFINE_integer("keep_checkpoint_max", 10, "The maximum number of recent checkpoint files to keep. " +
                        "As new files are created, older files are deleted. If None or 0, all checkpoint" +
                        " files are kept.")
tf.flags.DEFINE_integer("tf_random_seed", 42, "Random seed for TensorFlow initializers. Setting this value " +
                        "allows consistency between reruns")

# validation
tf.flags.DEFINE_integer("eval_steps", None, "Number of batches to evaluate in one run of validation")
tf.flags.DEFINE_integer("every_n_steps", 5, "Validation is run after this many steps")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    TIMESTAMP = int(time.time())
    MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))


def main(unused_argv):
    trn_x, trn_y, val_x, val_y, _, _, _, _ = ptb_reader.load_ptb_dataset(FLAGS.data_path, FLAGS.batch_size,
                                                                         FLAGS.num_time_steps, FLAGS.debug)

    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=FLAGS.save_checkpoints_secs,
                                        save_summary_steps=FLAGS.save_summary_steps,
                                        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                                        tf_random_seed=FLAGS.tf_random_seed)

    estimator = Estimator(model_fn=adaptive_computation_time, model_dir=MODEL_DIR, config=config,
                          params=FLAGS.__dict__['__flags'])

    metrics = {
        'perplexity': MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean,
                                 prediction_key="perplexity")
    }

    val_monitor = ValidationMonitor(x=val_x, y=val_y, batch_size=FLAGS.batch_size,
                                    eval_steps=FLAGS.eval_steps,
                                    every_n_steps=FLAGS.every_n_steps,
                                    metrics=metrics)

    estimator.fit(x=trn_x, y=trn_y, batch_size=FLAGS.batch_size, monitors=[val_monitor])

if __name__ == "__main__":
    tf.app.run()
