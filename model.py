#!/usr/bin/env python
import json
import time
import tensorflow as tf

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

# Helper function for processing the lstm output
def network_output(questions, question_lengths, image_features):
    # word embedding layer
    embedding_weights = tf.get_variable(
        "embedding_weights", [input_vocabulary_size, word_embedding_size],
        initializer=tf.random_normal_initializer()
    )
    word_embedding = tf.nn.embedding_lookup(embedding_weights, questions)

    # lstm layer
    _, lstm_state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_width),
        word_embedding,
        sequence_length=question_lengths,
        dtype=tf.float32,
    )

    lstm_output = tf.concat(1, [lstm_state.c, lstm_state.h])

    # Fully connected layer after the lstm
    fc_lstm_weights = tf.get_variable(
        "fc_lstm_weights", [2*lstm_width, 1024],
        initializer=tf.random_normal_initializer()
    )
    fc_lstm_biases = tf.get_variable(
        "fc_lstm_biases", [1024],
        initializer=tf.random_normal_initializer()
    )
    fc_lstm_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                lstm_output, fc_lstm_weights
            ), fc_lstm_biases
        )
    )

    # Fully connected layer for the image embedding
    fc_image_weights = tf.get_variable(
        "fc_image_weights", [4096, 1024],
        initializer=tf.random_normal_initializer()
    )
    fc_image_biases = tf.get_variable(
        "fc_image_biases", [1024], initializer=tf.random_normal_initializer()
    )
    fc_image_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                image_features, fc_image_weights
            ), fc_image_biases
        )
    )

    # Merge the question and image channels
    channel_merge_output = tf.multiply(fc_lstm_output, fc_image_output)

    # MLP for the combined embedding
    fc1_weights = tf.get_variable(
        "fc1_weights", [1024, 1024], initializer=tf.random_normal_initializer()
    )
    fc1_biases = tf.get_variable(
        "fc1_biases", [1024], initializer=tf.random_normal_initializer()
    )
    fc1_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                channel_merge_output, fc1_weights
            ), fc1_biases
        )
    )
    fc1_output_drop = tf.nn.dropout(fc1_output, 0.5)

    fc2_weights = tf.get_variable(
        "fc2_weights", [1024, 1024], initializer=tf.random_normal_initializer()
    )
    fc2_biases = tf.get_variable(
        "fc2_biases", [1024], initializer=tf.random_normal_initializer()
    )
    fc2_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                fc1_output_drop, fc2_weights
            ), fc2_biases
        )
    )
    fc2_output_drop = tf.nn.dropout(fc2_output, 0.5)

    final_weights = tf.get_variable(
        "final_weights", [1024, 1000],
        initializer=tf.random_normal_initializer()
    )
    final_biases = tf.get_variable(
        "final_biases", [1000], initializer=tf.random_normal_initializer()
    )
    final_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                fc2_output_drop, final_weights
            ), final_biases
        )
    )

    return final_output

######################################
# Training and validation operations #
######################################

def training_ops(batch_size, learning_rate):
    questions, question_lengths, answers, image_features = training_inputs(
        batch_size
    )
    final_output = network_output(questions, question_lengths, image_features)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(final_output, answers))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_output,1), answers), tf.float32))

    return optimizer, cost, accuracy

def run_training(sess, optimizer, cost, accuracy):
    step = 0
    while step < 60:
        _, _cost, _accuracy= sess.run([optimizer, cost, accuracy])
        print "Step: {}, cost: {}, accuracy: {}".format(step, _cost, _accuracy)
        step += 1

def validation_ops(batch_size):
    question_ids, questions, question_lengths, answer_choices, image_features = validation_inputs(batch_size)
    final_output = tf.nn.softmax(network_output(questions, question_lengths, image_features))
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
        ids, answers = sess.run([question_ids, chosen_answers])
        for i in range(len(ids)):
            res[ids[i][0]] = itoa[answers[i]]
        step += 1

    res = [{
        "question_id": k,
        "answer": v
    } for k,v in res.items()]

    with open(output_filename, 'w+') as f:
        json.dump(res, f)

if __name__ == '__main__':
    with tf.variable_scope("model") as scope:
        # Define all variables
        optimizer, cost, accuracy = training_ops(1000, 0.5)
        scope.reuse_variables()
        question_ids, chosen_answers = validation_ops(1000)

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        epoch = 0

        run_validation(sess, question_ids, chosen_answers, "Results/test.json")
        run_training(sess, optimizer, cost, accuracy)
        run_training(sess, optimizer, cost, accuracy)

        coord.request_stop()
        coord.join(threads)
