"""
Generates intermediary pickle files that consist of (word_vector, char_vector,
label) triplets.
"""

from __future__ import print_function, absolute_import, division

import csv
import os
import pickle
import sys

import constants
import dataset_parser


def generate(train_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading Glove... Please wait!")
    glove = dataset_parser.loadGlove(constants.GLOVE_PATH)
    print("Glove loaded! Generating vectors...")
    input_files = os.listdir(train_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in input_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    for hashtag in target_hashtags:
        input_filename = os.path.join(train_dir, hashtag + '.tsv')
        output_filename = os.path.join(output_dir, hashtag + '.pickle')
        tweets = load_input_file(
            input_filename)  # (tweet_id, tweet_text, tweet_level)

        data = []
        for tweet_id, tweet_text, tweet_level in tweets:
            word_vector = get_word_vector(glove, tweet_text)
            char_vector = get_char_vector(tweet_text)
            data.append((word_vector, char_vector, tweet_level))

        write_output_file(output_filename, data)

    print("Stored pickles to: " + output_dir)


def load_input_file(filename):
    tweets_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweets_list.append(row)
    return tweets_list


def get_word_vector(embed_dict, tweet_text):
    return dataset_parser.createGlovefromTweet(embed_dict, tweet_text)


def get_char_vector(tweet_text):
    return dataset_parser.tweet_to_integer_vector(tweet_text)


def write_output_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<train_data> <output_dir>')
        print('train_data directory contains train tsv files for each theme.')
        print('output directory for storing vector pickles.')
        print('Example call:')
        print('python3 hybrid_vector_generator.py ../dataset/train_data output')
        sys.exit(1)

    _, train_dir, output_dir = sys.argv
    generate(train_dir, output_dir)
