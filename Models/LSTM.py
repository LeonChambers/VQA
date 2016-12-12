#!/usr/bin/env python
import json
import tensorflow as tf

# Parse metadata
with open("metadata.json", "r") as f:
    _metadata = json.load(f)
max_question_length = _metadata["max_question_length"]
input_vocabulary_size = _metadata["input_vocabulary_size"]
itow = _metadata["itow"]
itoa = _metadata["itoa"]

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
            'image_features': tf.FixedLenFeature([4096], tf.float64)
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
            'image_features': tf.FixedLenFeatures([4096], tf.float64),
        }
    )
    return (
        features['question_id'], features['question'], 
        features['question_length'][0], features['answer_choices'],
        features['image_features']
    )

def training_inputs(batch_size, num_epochs):
    question, question_length, answer, image_features = read_training_questions()
    questions, question_lengths, answers, image_features = tf.train.shuffle_batch(
        [question, question_length, answer, image_features],
        batch_size=batch_size, capacity=1000+3*batch_size,
        min_after_dequeue=1000
    )
    return questions, question_lengths, answers, image_features

##############
# Full model #
##############

word_embedding_size = 300
lstm_width = 512

# Helper function for processing the lstm output
def last_relevant(x):
    batch_size = tf.shape(x)[0]
    index = tf.range(0, batch_size) * max_question_length + (max_question_length-1)
    return tf.gather(tf.reshape(x, [-1, lstm_width]), index)

def network_output(questions, question_lengths, image_features):
    # word embedding layer
    embedding_weights = tf.Variable(tf.random_normal([input_vocabulary_size, word_embedding_size]))
    word_embedding = tf.nn.embedding_lookup(embedding_weights, questions)

    # lstm layer
    output, state = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_width),
        word_embedding,
        sequence_length=question_lengths,
        dtype=tf.float32,
    )

    last_lstm_output = last_relevant(output)
    last_lstm_state = last_relevant(state)

    merged_lstm_output = tf.concat(1, [last_lstm_output, last_lstm_state])

    # Fully connected layer after the lstm
    fc_lstm_weights = tf.Variable(tf.random_normal([2*lstm_width, 1024]))
    fc_lstm_biases = tf.Variable(tf.random_normal([2*lstm_width]))
    fc_lstm_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                merged_lstm_output, fc_lstm_weights
            ), fc_lstm_biases
        )
    )

    # Fully connected layer for the image embedding
    fc_image_weights = tf.Variable(tf.random_normal([4096, 1024]))
    fc_image_biases = tf.Variable(tf.random_normal([1024]))
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
    fc1_weights = tf.Variable(tf.random_normal([1024, 1000]))
    fc1_biases = tf.Variable(tf.random_normal([1000]))
    fc1_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                channel_merge_output, fc1_weights
            ), fc1_biases
        )
    )
    fc1_output_drop = tf.nn.dropout(fc1_output, 0.5)

    fc2_weights = tf.Variable(tf.random_normal([1000, 1000]))
    fc2_biases = tf.Variable(tf.random_normal([1000]))
    fc2_output = tf.tanh(
        tf.nn.bias_add(
            tf.matmul(
                fc1_output_drop, fc2_weights
            ), fc2_biases
        )
    )
    fc2_output_drop = tf.nn.dropout(fc2_output_drop, 0.5)

    final_weights = tf.Variable(tf.random_normal([1000, 1000]))
    final_biases = tf.Variable(tf.random_normal([1000]))
    final_output = tf.nn.softmax(
        tf.tanh(
            tf.nn.bias_add(
                tf.matmul(
                    fc2_output_drop, final_weights
                ), final_biases
            )
        )
    )

    return final_output

# tf Graph input
# questions = tf.placeholder(tf.int32, [None, max_question_length])
# question_lengths = tf.placeholder(tf.int32, [None])
# image_features = tf.placeholder(tf.float32, [None, 4096])
# question_answers = tf.placeholder(tf.float32, [None, 1000])

# Evaluate model
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_output,1), tf.argmax(question_answers)), tf.float32))

def run_training(batch_size, num_epochs):
    questions, question_lengths, answers, image_features = training_inputs(
        batch_size, num_epochs
    )
    final_output = network_output(questions, question_lengths, image_features)

    # Define optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_output, answers))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    step = 0
    with tf.Session() as sess:
        _, cost = sess.run([optimizer, cost])
        print "Step: {}, cost: {}".format(step, cost)
        step += 1

def run_validation():
    pass

if __name__ == '__main__':
    run_training(10, 1)
