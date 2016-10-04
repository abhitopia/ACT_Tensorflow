import tensorflow as tf
from src.implementations.ACTCell import ACTCell
from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq


def adaptive_computation_time(features, targets, mode, params):
    # features = tf.placeholder(tf.int32, [batch_size, num_steps])
    # targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    hidden_size = params['hidden_size']
    vocab_size = params['vocab_size']
    use_lstm = params['use_lstm']
    max_computation = params['max_computation']
    ponder_time_penalty = params['ponder_time_penalty']
    epsilon = params['epsilon']
    learning_rate = params['learning_rate']
    max_grad_norm = params['max_grad_norm']

    features = tf.Print(features, [tf.shape(features)], 'features')
    embedding = tf.get_variable('embedding', [vocab_size, hidden_size])

    # set up ACT cell and inner rnn-type cell for use inside the ACT cell
    with tf.variable_scope("rnn"):
        if use_lstm:
            inner_cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        else:
            inner_cell = rnn_cell.GRUCell(hidden_size)

    with tf.variable_scope("ACT"):
        act = ACTCell(inner_cell, epsilon, max_computation=max_computation)

    inputs = tf.nn.embedding_lookup(embedding, features)
    inputs_split = list(tf.unpack(inputs, axis=1))

    outputs, final_state = rnn(act, inputs_split, dtype=tf.float32)

    # softmax to get probability distribution over vocab
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b

    # by default averages across time steps and batches
    loss = seq2seq.sequence_loss(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.reshape(tf.ones_like(targets, dtype=tf.float32), [-1])])

    loss_batch = tf.reduce_sum(loss)
    perplexity = tf.exp(loss_batch, name='perplexity')  # average perplexity per word
    # add up loss and retrieve batch-normalised ponder cost: sum N + sum Remainder
    cost = loss_batch + act.calculate_ponder_cost(time_penalty=ponder_time_penalty)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        # TO DO implement learning_rate_decay
        train_op = tf.contrib.layers.optimize_loss(
            loss=cost,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            clip_gradients=max_grad_norm,
            optimizer='Adam')
        return {'perplexity': perplexity}, cost, train_op
    else:
        return {'perplexity': perplexity}, cost, tf.no_op()
