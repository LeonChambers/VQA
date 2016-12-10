#!/usr/bin/env python
"""
Preprocesses the raw json data and outputs it in standard TensorFlow format
"""
import sys
import json
import numpy as np
import spacy.en
import random
import tensorflow as tf

K = 1000
UNKNOWN = 'UNK' # Special token for question words not in the vocabulary
max_question_length = 20

def load_data(data_subtype):
    """
    Load the question and annotation files for the given data subtype and return a list of question dictionaries
    """
    print "Loading data for {}".format(data_subtype)
    question_filename = "Questions/MultipleChoice_abstract_v002_{}_questions.json".format(data_subtype)
    annotation_filename = "Annotations/abstract_v002_{}_annotations.json".format(data_subtype)

    with open(question_filename, 'r') as f:
        questions = json.load(f)["questions"]
    with open(annotation_filename, 'r') as f:
        annotations = json.load(f)["annotations"]

    res = {question["question_id"]:question for question in questions}
    for annotation in annotations:
        assert res[annotation["question_id"]]["image_id"] == annotation["image_id"]
        res[annotation["question_id"]].update(annotation)

    return res.values()

def get_top_answers(questions):
    """
    Gets a list of the top K answers for the given list of questions
    """
    print "Getting a list of the top {} answers".format(K)
    counts = {}
    for question in questions:
        ans = question["multiple_choice_answer"]
        counts[ans] = counts.get(ans, 0) + 1
    ans_list = sorted([(count, w) for w,count in counts.items()], reverse=True)
    return [ans_list[i][1] for i in range(K)]

def tokenize_questions(questions):
    print "Tokenizing questions"
    nlp = spacy.en.English()
    for i, question in enumerate(questions):
        txt = [token.norm_ for token in nlp(question["question"])]
        question["question_tokens"] = txt
        if i % 1000 == 0:
            sys.stdout.write("Processing {}/{} ({:02.2f}% done)\r".format(
                i, len(questions), i*100.0/len(questions)
            ))
            sys.stdout.flush()
    print ""
    return questions

def build_question_vocab(questions):
    print "Building a vocabulary of question words"
    vocab = set()
    vocab.add(UNKNOWN)
    for question in questions:
        for token in question["question_tokens"]:
            vocab.add(token)
    return list(vocab)

def encode_questions(questions, wtoi):
    N = len(questions)
    encoded_questions = np.zeros((N, max_question_length), dtype='int32')
    question_lengths = np.zeros(N, dtype='int32')
    truncated_questions = 0
    for i,question in enumerate(questions):
        question_lengths[i] = len(question["question_tokens"])
        for j,word in enumerate(question["question_tokens"]):
            if j >= max_question_length:
                truncated_questions += 1
                break
            encoded_questions[i][j] = wtoi.get(word, wtoi[UNKNOWN])
    print "Truncated {}/{} questions".format(truncated_questions, N)
    return encoded_questions, question_lengths

def encode_answers(questions, atoi):
    N = len(questions)
    encoded_answers = np.zeros(N, dtype='int32')
    for i,question in enumerate(questions):
        encoded_answers[i] = atoi[question["multiple_choice_answer"]]
    return encoded_answers

def encode_answer_choices(questions, atoi):
    N = len(questions)
    encoded_answer_choices = np.zeros((N, 18), dtype='int32')
    for i,question in enumerate(questions):
        for j,ans in enumerate(questions[i]["multiple_choices"]):
            encoded_answer_choices[i][j] = atoi.get(ans, atoi[UNKNOWN])
    return encoded_answer_choices

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

if __name__ == '__main__':
    training_questions = load_data("train2015")
    validation_questions = load_data("val2015")

    # Get the top K answers
    top_ans = get_top_answers(training_questions)
    top_ans.append(UNKNOWN)
    atoi = {w:i for i,w in enumerate(top_ans)}
    itoa = {i:w for i,w in enumerate(top_ans)}

    # Only train on questions whose answers are in top_ans
    print "Filtering questions"
    training_questions = [
        q for q in training_questions if q["multiple_choice_answer"] in atoi
    ]

    # Shuffle the order of the training questions
    print "Shuffling questions"
    random.seed(123) # Make reproducible
    random.shuffle(training_questions) 

    # Tokenize the questions
    training_questions = tokenize_questions(training_questions)
    validation_questions = tokenize_questions(validation_questions)

    # Create the vocab for the questions
    vocab = build_question_vocab(training_questions)
    wtoi = {w:i for i,w in enumerate(vocab)}
    itow = {i:w for i,w in enumerate(vocab)}

    # Encode all the things
    print "Encoding question data"
    training_questions_encoded, training_question_lengths = encode_questions(training_questions, wtoi)
    training_answers = encode_answers(training_questions, atoi)
    validation_questions_encoded, validation_question_lengths = encode_questions(validation_questions, wtoi)
    validation_answer_choices = encode_answer_choices(validation_questions, atoi)

    # Make sure the question and answer encodings make sense
    print "A sampling of some decoded questions:"
    for _ in range(10):
        i = random.randint(0, len(training_question_lengths)-1)
        question = " ".join([
            itow[training_questions_encoded[i][j]] for j in range(training_question_lengths[i])
        ])
        answer = itoa[training_answers[i]]
        print question, answer
    for _ in range(10):
        i = random.randint(0, len(validation_question_lengths)-1)
        question = " ".join([
            itow[validation_questions_encoded[i][j]] for j in range(validation_question_lengths[i])
        ])
        choices = [
            itoa[validation_answer_choices[i][j]] for j in range(18)
        ]
        print question, choices

    # Write the training data to the file system
    print "Writing training data"
    writer = tf.python_io.TFRecordWriter("training_data.tfrecords")
    for i in range(len(training_question_lengths)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'question': _int64_feature(
                training_questions_encoded[i].tolist()
            ),
            'question_length': _int64_feature([
                np.asscalar(training_question_lengths[i])
            ]),
            'answer': _int64_feature([
                np.asscalar(training_answers[i])
            ])
        }))
        writer.write(example.SerializeToString())
        if i % 1000 == 0:
            sys.stdout.write("Processing {}/{} ({:02.2f}% done)\r".format(
                i, len(training_question_lengths), 
                i*100.0/len(training_question_lengths)
            ))
            sys.stdout.flush()
    writer.close()
    print ""

    # Write the validation data to the file system
    print "Writing validation data"
    writer = tf.python_io.TFRecordWriter("validation_data.tfrecords")
    for i in range(len(validation_question_lengths)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'question': _int64_feature(
                validation_questions_encoded[i].tolist()
            ),
            'question_length': _int64_feature([
                np.asscalar(validation_question_lengths[i])
            ]),
            'answer_choices': _int64_feature(
                validation_answer_choices[i].tolist()
            )
        }))
        writer.write(example.SerializeToString())
        if i % 1000 == 0:
            sys.stdout.write("Processing {}/{} ({:02.2f}% done)\r".format(
                i, len(validation_question_lengths), 
                i*100.0/len(validation_question_lengths)
            ))
            sys.stdout.flush()
    writer.close()
    print ""
