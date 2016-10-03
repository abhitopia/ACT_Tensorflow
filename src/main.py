import tensorflow as tf
import time
import os
import utils.ptb_reader as reader
from src.implementations.ACTCell import ACTCell
from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq
from tensorflow.contrib.learn.python.learn.monitors import ValidationMonitor
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

# TODO Pass the initial state as the final output state
# TODO Try using dynamic rnn for sentence-wise rnn
# TODO remove explicit dependence on batch_size in ACT_cell tensor.get_shape()

tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_string("data_path", 'data/', "data_path (/data)")
tf.flags.DEFINE_integer("batch_size", 32, "batch size (defaults to 32)")
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
tf.flags.DEFINE_integer("save_checkpoints_secs", 600, "Save checkpoints every this many seconds")
tf.flags.DEFINE_integer("save_summary_steps", 10, "Save summaries every this many steps")
tf.flags.DEFINE_integer("keep_checkpoint_max", 10, "The maximum number of recent checkpoint files to keep. " +
                        "As new files are created, older files are deleted. If None or 0, all checkpoint" +
                        " files are kept.")
tf.flags.DEFINE_integer("tf_random_seed", 42, "Random seed for TensorFlow initializers. Setting this value " +
                                              "allows consistency between reruns")

# validation
tf.flags.DEFINE_integer("eval_steps", 10, "Number of batches to evaluate in one run of validation")
tf.flags.DEFINE_integer("every_n_steps", 100, "Validation is run after this many steps")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    TIMESTAMP = int(time.time())
    MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))


def model_fn(features, targets, mode):
    # features = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])
    # targets = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.num_steps])

    tf.Assert(tf.equal(tf.shape(features)[0], FLAGS.batch_size), [features, targets])
    embedding = tf.get_variable('embedding', [FLAGS.vocab_size, FLAGS.hidden_size])
    # set up ACT cell and inner rnn-type cell for use inside the ACT cell
    with tf.variable_scope("rnn"):
        if FLAGS.use_lstm:
            inner_cell = rnn_cell.BasicLSTMCell(FLAGS.hidden_size, state_is_tuple=True)
        else:
            inner_cell = rnn_cell.GRUCell(FLAGS.hidden_size)

    with tf.variable_scope("ACT"):
        act = ACTCell(inner_cell, FLAGS.epsilon, max_computation=FLAGS.max_computation, batch_size=FLAGS.batch_size)

    inputs = tf.nn.embedding_lookup(embedding, features)
    inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, FLAGS.num_time_steps, inputs)]

    outputs, final_state = rnn(act, inputs, dtype=tf.float32)

    # softmax to get probability distribution over vocab
    output = tf.reshape(tf.concat(1, outputs), [-1, FLAGS.hidden_size])
    softmax_w = tf.get_variable("softmax_w", [FLAGS.hidden_size, FLAGS.vocab_size])
    softmax_b = tf.get_variable("softmax_b", [FLAGS.vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b  # dim (400, 10,000)(numsteps*batchsize, vocabsize)

    # by default averages across time steps and batches
    loss = seq2seq.sequence_loss(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([FLAGS.batch_size * FLAGS.num_time_steps])])

    loss_batch = tf.reduce_sum(loss)
    perplexity = tf.exp(loss_batch, name='perplexity') # average perplexity per word
    # add up loss and retrieve batch-normalised ponder cost: sum N + sum Remainder
    cost = loss_batch + act.calculate_ponder_cost(time_penalty=FLAGS.ponder_time_penalty)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        # TO DO implement learning_rate_decay
        train_op = tf.contrib.layers.optimize_loss(
            loss=cost,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=FLAGS.learning_rate,
            clip_gradients=FLAGS.max_grad_norm,
            optimizer='Adam')
        return {'perplexity': perplexity}, cost, train_op
    else:
        return {'perplexity': perplexity}, cost, tf.no_op()


def load_dataset():
    print('Loading dataset...')
    if not FLAGS.debug:
        raw_data = reader.ptb_raw_data(FLAGS.data_path, "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    else:
        raw_data = reader.ptb_raw_data(FLAGS.data_path + 'debug/', "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    train_data, val_data, test_data, vocab, word_to_id = raw_data
    trn_x, trn_y = reader.ptb_split_to_features_and_targets(train_data, FLAGS.num_time_steps, FLAGS.batch_size)
    val_x, val_y = reader.ptb_split_to_features_and_targets(val_data, FLAGS.num_time_steps, FLAGS.batch_size)
    tst_x, tst_y = reader.ptb_split_to_features_and_targets(test_data, FLAGS.num_time_steps, FLAGS.batch_size)
    return trn_x, trn_y, val_x, val_y, tst_x, tst_y, vocab, word_to_id


def get_config():
    config = tf.contrib.learn.RunConfig(save_checkpoints_secs=FLAGS.save_checkpoints_secs,
                                        save_summary_steps=FLAGS.save_summary_steps,
                                        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
                                        tf_random_seed=FLAGS.tf_random_seed)
    return config


def get_metrics():
    metrics = {
        'perplexity': MetricSpec(metric_fn=tf.contrib.metrics.streaming_mean,
                                 prediction_key="perplexity")
    }
    return metrics


def main(unused_argv):
    trn_x, trn_y, val_x, val_y, _, _, _, _ = load_dataset()

    estimator = tf.contrib.learn.Estimator(model_fn=model_fn,
                                           model_dir=MODEL_DIR,
                                           config=get_config())

    val_monitor = ValidationMonitor(x=val_x, y=val_y, batch_size=FLAGS.batch_size,
                                    eval_steps=FLAGS.eval_steps,
                                    every_n_steps=FLAGS.every_n_steps,
                                    metrics=get_metrics())

    estimator.fit(x=trn_x, y=trn_y, batch_size=FLAGS.batch_size,
                  monitors=[val_monitor])


if __name__ == "__main__":
    tf.app.run()
