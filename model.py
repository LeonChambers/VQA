#!/usr/bin/env python
import sys
import json
import time
import argparse
import tensorflow as tf

# Enum for types of question channels
LSTM = "LSTM"
DeeperLSTM = "DeeperLSTM"
BLSTM = "BLSTM"
question_channel_types = [
    LSTM, DeeperLSTM, BLSTM
]

# Parse metadata
with open("metadata.json", "r") as f:
    _metadata = json.load(f)
K = _metadata["K"]
max_question_length = _metadata["max_question_length"]
input_vocabulary_size = _metadata["input_vocabulary_size"]
itow = _metadata["itow"]
itoa = _metadata["itoa"]
itoa = {int(k):v for k,v in itoa.items()}

#######################
# Read the input data #
#######################

train_filename = "training_data.tfrecords"
validation_filename = "validation_data.tfrecords"

def read_training_question():
    reader = tf.TFRecordReader()
    train_queue = tf.train.string_input_producer([train_filename])
    _, serialized_example = reader.read(train_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'question': tf.FixedLenFeature([max_question_length], tf.int64),
            'question_length': tf.FixedLenFeature([1], tf.int64),
            'answer': tf.FixedLenFeature([1], tf.int64),
            'image_features': tf.FixedLenFeature([4096], tf.float32)
        }
    )
    return (
        features['question'], features['question_length'][0],
        features['answer'][0], features['image_features']
    )

def read_validation_question():
    reader = tf.TFRecordReader()
    validation_queue = tf.train.string_input_producer([validation_filename])
    _, serialized_example = reader.read(validation_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'question_id': tf.FixedLenFeature([1], tf.int64),
            'question': tf.FixedLenFeature([max_question_length], tf.int64),
            'question_length': tf.FixedLenFeature([1], tf.int64),
            'answer_choices': tf.FixedLenFeature([18], tf.int64),
            'image_features': tf.FixedLenFeature([4096], tf.float32),
        }
    )
    return (
        features['question_id'], features['question'],
        features['question_length'][0], features['answer_choices'],
        features['image_features']
    )

def training_inputs(batch_size):
    val = read_training_question()
    vals = tf.train.shuffle_batch(
        val, batch_size=batch_size, capacity=1000+3*batch_size,
        min_after_dequeue=1000, allow_smaller_final_batch=True
    )
    return vals

def validation_inputs(batch_size):
    val = read_validation_question()
    vals = tf.train.shuffle_batch(
        val, batch_size=batch_size, capacity=1000+3*batch_size,
        min_after_dequeue=1000
    )
    return vals

##############
# Full model #
##############

word_embedding_size = 300
lstm_width = 512
keep_prob = tf.placeholder(tf.float32)

# Helper function for processing the lstm output
def network_output(
        questions, question_lengths, image_features, question_channel_type
    ):
    # word embedding layer
    embedding_weights = tf.get_variable(
        "embedding_weights", [input_vocabulary_size, word_embedding_size],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    word_embedding = tf.tanh(
        tf.nn.dropout(
            tf.nn.embedding_lookup(embedding_weights, questions), keep_prob
        )
    )

    # lstm layer
    if question_channel_type == LSTM:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_width)
        _, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell,
            word_embedding,
            sequence_length=question_lengths,
            dtype=tf.float32,
        )
        lstm_output = tf.concat(1, [lstm_state.c, lstm_state.h])
        lstm_output_size = 2*lstm_width
    elif question_channel_type == DeeperLSTM:
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(lstm_width)]*2
        )
        _, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell,
            word_embedding,
            sequence_length=question_lengths,
            dtype=tf.float32,
        )
        lstm_output = tf.concat(
            1, [
                lstm_state[0].c, lstm_state[0].h, lstm_state[1].c,
                lstm_state[1].h
            ]
        )
        lstm_output_size = 4*lstm_width
    elif question_channel_type == BLSTM:
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_width)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_width)
        _, lstm_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            word_embedding,
            sequence_length=question_lengths,
            dtype=tf.float32,
        )
        lstm_output = tf.concat(
            1, [
                lstm_state[0].c, lstm_state[0].h, lstm_state[1].c,
                lstm_state[1].h
            ]
        )
        lstm_output_size = 4*lstm_width
    else:
        raise ValueError("Invalid question_channel_type")

    question_channel_output = tf.nn.dropout(
        lstm_output, keep_prob
    )

    # Fully connected layer after the lstm
    fc_lstm_weights = tf.get_variable(
        "fc_lstm_weights", [lstm_output_size, 1024],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    fc_lstm_biases = tf.get_variable(
        "fc_lstm_biases", [1024],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    fc_lstm_output = tf.tanh(
        tf.nn.dropout(
            tf.nn.bias_add(
                tf.matmul(
                    lstm_output, fc_lstm_weights
                ), fc_lstm_biases
            ), keep_prob
        )
    )

    # Fully connected layer for the image embedding
    fc_image_weights = tf.get_variable(
        "fc_image_weights", [4096, 1024],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    fc_image_biases = tf.get_variable(
        "fc_image_biases", [1024],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    fc_image_output = tf.tanh(
        tf.nn.dropout(
            tf.nn.bias_add(
                tf.matmul(
                    image_features, fc_image_weights
                ), fc_image_biases
            ), keep_prob
        )
    )

    # Merge the question and image channels
    channel_merge_output = tf.nn.dropout(
        tf.multiply(fc_lstm_output, fc_image_output), keep_prob
    )

    final_weights = tf.get_variable(
        "final_weights", [1024, 1000],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )

    final_biases = tf.get_variable(
        "final_biases", [1000],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )

    final_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                channel_merge_output, final_weights
            ), final_biases
        )
    )

    return final_output

######################################
# Training and validation operations #
######################################

def training_ops(batch_size, learning_rate, question_channel_type):
    questions, question_lengths, answers, image_features = training_inputs(
        batch_size
    )
    truncated_question_lengths = tf.minimum(
        question_lengths, max_question_length
    )
    final_output = network_output(
        questions, truncated_question_lengths, image_features,
        question_channel_type
    )

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(final_output, answers))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_output,1), answers), tf.float32))

    return optimizer, cost, accuracy

def run_training(sess, optimizer, cost, accuracy, batch_size):
    step = 0
    num_batches = 60000/batch_size
    while step < num_batches:
        sys.stdout.write(
            "Running training {}/{} ({:02.2f}% done)\r".format(
                step, num_batches, step*100.0/num_batches
            )
        )

        _, _cost, _accuracy= sess.run(
            [optimizer, cost, accuracy], feed_dict={keep_prob: 0.5}
        )
        print "Step: {}, cost: {}, accuracy: {}".format(
            step, _cost, _accuracy
        )
        step += 1

def validation_ops(batch_size, question_channel_type):
    question_ids, questions, question_lengths, answer_choices, image_features = validation_inputs(batch_size)
    truncated_question_lengths = tf.minimum(
        question_lengths, max_question_length
    )
    final_output = tf.nn.softmax(network_output(
        questions, truncated_question_lengths, image_features, question_channel_type
    ))
    chosen_answers = tf.argmax(
        tf.multiply(
            final_output, tf.reduce_sum(
                tf.one_hot(
                    answer_choices, K, axis=1
                ), 2
            )
        ), 1
    )

    return question_ids, chosen_answers

def run_validation(sess, question_ids, chosen_answers, output_filename):
    res = {}
    step = 0
    while step < 30:
        sys.stdout.write(
            "Processing validation dataset {}/{} ({:02.2f}% done)\r".format(
                step, 30, step*100.0/30
            )
        )
        sys.stdout.flush()
        ids, answers = sess.run(
            [question_ids, chosen_answers], feed_dict={keep_prob: 1}
        )
        for i in range(len(ids)):
            res[ids[i][0]] = itoa[answers[i]]
        step += 1
    print ""

    res = [{
        "question_id": k,
        "answer": v
    } for k,v in res.items()]

    with open(output_filename, 'w+') as f:
        json.dump(res, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'question_channel_type', choices=question_channel_types
    )
    args = parser.parse_args()

    print "Using a {} for the question".format(args.question_channel_type)

    training_batch_size = 500

    with tf.variable_scope("model") as scope:
        # Define all variables
        optimizer, cost, accuracy = training_ops(
            training_batch_size, 0.0005, args.question_channel_type
        )
        scope.reuse_variables()
        question_ids, chosen_answers = validation_ops(
            1000, args.question_channel_type
        )

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(100):
            print "Starting epoch", epoch
            results_filename = "Results/{}/epoch_{:03d}.json".format(
                args.question_channel_type, epoch
            )
            run_validation(
                sess, question_ids, chosen_answers, results_filename
            )
            run_training(sess, optimizer, cost, accuracy, training_batch_size)

        coord.request_stop()
        coord.join(threads)
