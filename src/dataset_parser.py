import logging
import os
import pickle
import re

import nltk
import numpy as np


def filterText(tweets):
    """
     Removes unnecessary data from the tweet such as extra hashtags, links...
    :param tweets: List of all tweets
    :return:
    """
    result = []
    for tweet in tweets:
        filtered = []
        for token in tweet:
            if (not (token.startswith('#') or token.startswith(
                    '@') or token.startswith('.@') or token.startswith(
                'http'))):
                filtered.append(token)
        while filtered.__contains__(''):
            filtered.remove('')
        result.append(filtered)
    return result


def read_file_by_line_and_tokenize(file_path):
    """
    Reads the twee document file and tokenizes it.
    :param file_path:
    :return: List of tokenized tweets
    """
    tweets = [line.rstrip('\n') for line in
              open(file_path, 'r', encoding="utf8")]
    return [re.split('\t', tweet) for tweet in tweets]


def create_pair_combs(lst):
    """
    Create all pair combinations from tweets of different humor ranking.
    :param lst:
    :return: List of docuement pairs and the list of label pairs
    """
    index = 1
    pairs_matrix = []
    pairs_labels = []
    for element1 in lst:
        for element2 in lst[index:]:
            if element1[1] == element2[1]:
                continue
            if element1[1] < element2[1]:
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element1[0], element2[0]),
                                                  axis=1)
                    pairs_matrix.append(concatMatrix)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.concatenate((element2[0], element1[0]),
                                                  axis=1)
                    pairs_matrix.append(concatMatrix)
                    pairs_labels.append(1)
            else:
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element2[0], element1[0]),
                                                  axis=1)
                    pairs_matrix.append(concatMatrix)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.concatenate((element1[0], element2[0]),
                                                  axis=1)
                    pairs_matrix.append(concatMatrix)
                    pairs_labels.append(1)
        index += 1
    return pairs_matrix, pairs_labels


def parse_data(glove_file,
               data_path,
               pickleDir,
               embedding_dim=100,
               timestep=25):
    """
    Creates per token tweet embeddings using the Glove 100-D vectors.
    Exports embeddings to a pickle file.

    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param timestep: Maximum sentence length
    :param glove_file: Glove file path
    :param data_path: Files directory path
    :param pickleDir: Export pickle directory
    """
    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    topicsMatrix = []

    for i, filename in enumerate(os.listdir(data_path)):
        print("Parsing file number:", i)

        if filename.endswith(".tsv"):
            tweetsMatrix = []
            current_rows = read_file_by_line_and_tokenize(
                os.path.join(data_path, filename))
            ranks = [word[2] for word in current_rows]
            tokenized = [nltk.word_tokenize(word[1]) for word in current_rows]
            tokenizedCopy = []
            for token in tokenized:
                tokenizedCopy.append(
                    [word.lower() for word in token if word.isalpha()])
            for k, tweet in enumerate(tokenized):
                sentenceRow = np.zeros((embedding_dim, timestep))
                for j, token in enumerate(tweet[:timestep]):
                    if token in embed_dict:
                        sentenceRow[:, j] = embed_dict[token.lower()]

                tweetsMatrix.append((sentenceRow, ranks[k]))
            topicsMatrix.append(tweetsMatrix)

    with open(pickleDir, "wb") as f:
        pickle.dump(topicsMatrix, f)


if __name__ == "__main__":
    parse_data("./resources/glove/glove.twitter.27B.100d.txt",
               "../dataset/train_data", "./train_pairs.pkl")
