#!/usr/bin/env python

from __future__ import print_function, absolute_import, division

import csv
import os
import sys

import numpy as np

import constants
import dataset_parser
import model_evaluation


def generate(input_dir, output_dir):
    glove = dataset_parser.loadGlove(constants.GLOVE_PATH)
    print("loaded glove file")

    model = model_evaluation.Model(os.path.join(constants.TF_WEIGHTS,
                                                "CNN_BILSTM_FC_model.ckpt-1320"))  # TODO
    input_files = os.listdir(input_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in input_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    for hashtag in target_hashtags:
        input_filename = os.path.join(input_dir, hashtag + '.tsv')
        output_filename = os.path.join(output_dir, hashtag + '_PREDICT.tsv')
        tweets = load_input_file(input_filename)  # (tweetID, tweetText)

        results = []
        # make tweet combinations and get result
        index = 1
        ind = 0
        for tweetID1, tweet_text1 in tweets:
            for tweetID2, tweet_text2 in tweets[index:]:
                if tweetID1 == tweetID2:
                    continue

                word_vect1, char_vect1 = get_feature_vector(glove, tweet_text1)
                word_vect2, char_vect2 = get_feature_vector(glove, tweet_text2)
                word_merged = np.concatenate((word_vect1, word_vect2), axis=1)
                char_merged = np.concatenate((char_vect1, char_vect2), axis=0)

                network_result = get_classification(model,
                                                    word_merged,
                                                    char_merged)
                results.append((tweetID1, tweetID2, network_result))

                if ind % 2000 == 0:
                    print("{}/{}".format(ind,
                                         len(tweets) * (len(tweets) - 1) / 2))
                ind += 1

            index += 1

        write_output_file(output_filename, results)


def get_feature_vector(embed_dict, tweet_text):
    return (dataset_parser.createGlovefromTweet(embed_dict, tweet_text),
            dataset_parser.tweet_to_integer_vector(tweet_text))


def get_classification(model, word_merged, char_merged):
    return model.predict(word_merged, char_merged)


def load_input_file(filename):
    tweets_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweets_list.append(row)
    return tweets_list


def write_output_file(filename, results):
    with open(filename, 'w') as f:
        for tweetID1, tweetID2, result in results:
            f.write(tweetID1 + "\t" + tweetID2 + "\t" + str(result) + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<input_dir> <output_dir>')
        print('Input directory contains tsv files for each theme.')
        sys.exit(1)

    _, input_dir, output_dir = sys.argv
    generate(input_dir, output_dir)
