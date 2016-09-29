from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import tensorflow as tf

import config as cf
import reader as reader
import saveload
from epoch import run_epoch
from models.AdaptiveComputationTime import ACTModel


# TODO
# Configure moving average (perhaps not needed with tensor board)
# Try using contrib.learn (skflow)
# Add Tensor board logging
# Pass the initial state as the final output state
# Try using dynamic rnn for sentence-wise rnn
# untie loss ( with Ponder Cost) and perplexity


def get_config(conf):
    if conf == "small":
        return cf.SmallConfig
    elif conf == "medium":
        return cf.MediumConfig
    elif conf == "large":
        return cf.LargeConfig
    elif conf == "titanx":
        return cf.TitanXConfig
    else:
        raise ValueError('did not enter acceptable model size:', conf)


def main():
    config = get_config(FLAGS.model_size)
    eval_config = get_config(FLAGS.model_size)
    saved_model_path = FLAGS.model_path
    weights_dir = FLAGS.weights_dir
    verbose = FLAGS.verbose
    debug = FLAGS.debug

    if weights_dir is not None:
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

    if not debug:
        raw_data = reader.ptb_raw_data(FLAGS.data_path, "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    else:
        raw_data = reader.ptb_raw_data(FLAGS.data_path + 'debug/', "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")

    # load up PTB data
    train_data, val_data, test_data, vocab, word_to_id = raw_data

    with tf.Graph().as_default(), tf.Session() as session:
        initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initialiser):
            m = ACTModel(config, is_training=True)

            # if we have a saved/pre-trained model, load it.
            if saved_model_path is not None:
                saveload.main(saved_model_path, session)

        tf.initialize_all_variables().run()

        print("starting training")
        for i in range(config.max_max_epoch):

            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            session.run(tf.assign(m.lr, config.learning_rate * lr_decay))
            train_loss = run_epoch(session, m, train_data, m.train_op, verbose=True)
            valid_loss = run_epoch(session, m, val_data, tf.no_op())

            if verbose:
                print("Epoch: {} Learning rate: {}".format(i + 1, session.run(m.lr)))
                print("Epoch: {} Train Loss: {}".format(i + 1, train_loss))
                print("Epoch: %d Valid Loss: %.3f" % (i + 1, valid_loss))

            # save weights in a pickled dictionary format
            if weights_dir is not None:
                date = "{:%m.%d.%H.%M}".format(datetime.now())
                saveload.main(weights_dir + "/Epoch_{:02}Train_{:0.3f}Val_{:0.3f}date{}.pkl"
                              .format(i+1, train_loss,valid_loss, date), session)

        test_loss = run_epoch(session, m, test_data, tf.no_op())

    if verbose:
        print("Test Perplexity: %.3f" % test_loss)


if __name__ == '__main__':

    flags = tf.flags
    # logging = tf.logging
    flags.DEFINE_string("model_size", "small", "Size of model to train, either small, medium or large")
    flags.DEFINE_string("data_path", 'data/', "data_path")
    flags.DEFINE_string("model_path", None, "full path of a saved model to load")
    flags.DEFINE_string("weights_dir", 'model/', "full directory path to save weights into per epoch")
    flags.DEFINE_boolean("verbose", True, "Verbosity of the training")
    flags.DEFINE_boolean("debug", True, "Uses small corpuses for debugging purposes")
    FLAGS = flags.FLAGS

    # from sys import argv
    flags.FLAGS._parse_flags()
    main()
