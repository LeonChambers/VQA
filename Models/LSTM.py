#!/usr/bin/env python
import json
import tensorflow as tf

max_question_length = 20

# Parse metadata
with open("metadata.json", "r") as f:
    _metadata = json.load(f)
input_vocabulary_size = _metadata["input_vocabulary_size"]
itow = _metadata["itow"]
itoa = _metadata["itoa"]

train_filename = "training_data.tfrecords"
validation_filename = "validation_data.tfrecords"

train_queue = tf.train.string_input_producer([train_filename])
validation_queue = tf.train.string_input_producer([validation_filename])

def read_training_question():
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(train_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'question': tf.FixedLenFeature([max_question_length], tf.int64),
            'question_length': tf.FixedLenFeature([1], tf.int64),
            'answer': tf.FixedLenFeature([1], tf.int64),
        }
    )
    return (
        features['question'], features['question_length'][0], features['answer'][0]
    )

def read_validation_question():
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(validation_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'question': tf.FixedLenFeature([max_question_length], tf.int64),
            'question_length': tf.FixedLenFeature([1], tf.int64),
            'answer_choices': tf.FixedLenFeature([1], tf.int64),
        }
    )
    return (
        features['question'], features['question_length'][0], features['answer_choices'][0]
    )

def inputs(train, batch_size, num_epochs):
    question, question_length, answer = read_training_questions()
    questions, question_lengths, answers = tf.train.shuffle_batch(
        [question, question_length, answer], batch_size=batch_size,
        capacity=1000+3*batch_size, min_after_dequeue=1000
    )
    return questions, question_lengths, answers

word_embedding_size = 300
lstm_width = 512

# tf Graph input
questions = tf.placeholder(tf.float32, [None, max_question_length, input_vocabulary_size])
question_lengths = tf.placeholder(tf.float32, [None])

# word embedding layer
embedding_weights = tf.Variable(tf.random_normal([input_vocabulary_size, word_embedding_size]))
embedding_biases = tf.Variable(tf.random_normal([word_embedding_size]))
word_embedding = tf.tanh(
    tf.reshape(
        tf.nn.bias_add(
            tf.matmul(
                tf.reshape(questions, [-1, input_vocabulary_size]), embedding_weights
            ),
            embedding_biases
        ),
        [-1, max_question_length, input_vocabulary_size]
    )
)

# lstm layer
output, state = tf.nn.dynamic_rnn(
    tf.nn.rnn_cell.BasicLSTMCell(lstm_width),
    word_embedding,
    sequence_length=question_lengths,
    dtype=tf.float32,
)

def last_relevant(x):
    batch_size = tf.shape(x)[0]
    index = tf.range(0, batch_size) * max_question_length + (max_question_length-1)
    return tf.gather(tf.reshape(x, [-1, lstm_width]), index)

last_lstm_output = last_relevant(output)
last_lstm_state = last_relevant(state)

merged_lstm_output = tf.concat(1, [last_lstm_output, last_lstm_state])

fc_lstm_weights = tf.Variable(tf.random_normal([2*lstm_width, 1024]))
fc_lstm_biases = tf.Variable(tf.random_normal([2*lstm_width]))
fc_lstm_output = tf.tanh(
    tf.nn.bias_add(
        tf.matmul(
            merged_lstm_output, fc_lstm_weights
        ), fc_lstm_biases
    )
)

# TODO: Image channel and merging
channel_merge_output = fc_lstm_output

fc1_weights = tf.Variable(tf.random_normal([1024, 1000]))
fc1_biases = tf.Variable(tf.random_normal([1000]))
fc1_output = tf.tanh(
    tf.nn.bias_add(
        tf.matmul(
            channel_merge_output, fc1_weights
        ), fc1_biases
    )
)

fc2_weights = tf.Variable(tf.random_normal([1000, 1000]))
fc2_biases = tf.Variable(tf.random_normal([1000]))
fc2_output = tf.tanh(
    tf.nn.bias_add(
        tf.matmul(
            fc1_output, fc2_weights
        ), fc2_biases
    )
)

final_weights = tf.Variable(tf.random_normal([1000, 1000]))
final_biases = tf.Variable(tf.random_normal([1000]))
final_output = tf.tanh(
    tf.nn.bias_add(
        tf.matmul(
            fc2_output, final_weights
        ), final_biases
    )
)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run([final_output])
