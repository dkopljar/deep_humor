import logging
import os
import pickle
import re

import nltk
import numpy as np

import char_mapper


def clear_tweet(tweet):
    """
    Clear given tweet from hashtags, mails, links...
    :param tweet: Tweet
    :return: cleared tweet
    """
    tweet = tweet
    for word in tweet.split():
        if word.startswith('@') or word.startswith(
                '.@') or word.startswith('http') or word.startswith('#'):
            tweet = tweet.replace(' ' + word, "")  # if it is on the end
            tweet = tweet.replace(word + ' ', "")  # if it is on the begining
    return tweet


def tweet_to_integer_vector(tweet, tweet_char_count=70):
    """
    Maps given tweet (it will lowercase it) to np.array of constants.TWEET_CHARACTER_COUNT dimension, 
    with zeros as padding and ending vector with defined character (41)
    :param tweet: Tweet
    :return: Integer vector for given tweet
    """
    tweet = clear_tweet(tweet.lower())
    vector = np.zeros(tweet_char_count, dtype=np.int)
    last_index = 0
    for index, character in enumerate(tweet):
        last_index = index
        if index == tweet_char_count - 1:  # leave space for ending character
            break
        vector[index] = char_mapper.map_letter_to_int(character)
    vector[last_index] = char_mapper.map_letter_to_int('end')
    return vector


def loadGlove(glove_file):
    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    return embed_dict


def read_file_by_line_and_tokenize(file_path):
    """
    Reads the twee document file and tokenizes it.
    :param file_path:
    :return: List of tokenized tweets
    """
    tweets = [line.rstrip('\n') for line in
              open(file_path, 'r', encoding="utf8")]
    return [re.split('\t', tweet) for tweet in tweets]


def prepare_dataset_for_taskB(glove_file,
                              data_path,
                              pickleInputDir,
                              pickleLabelDir,
                              embedding_dim=100,
                              timestep=25):
    """
    Creates per token tweet embeddings using the Glove 100-D vectors.
    Exports embeddings for tweets and labels to a pickle file.

    Labels: [0,0,1] = 2   [0,1,0] = 1  [1,0,0] = 0

    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param timestep: Maximum sentence length
    :param glove_file: Glove file path
    :param data_path: Files directory path
    :param pickleInputDir: Export pickle directory for inputs
    :param pickleLabelDir: Export pickle directory for inputs
    """

    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    inputMatrix = []
    labelMatrix = []

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
                label = np.zeros(3)
                label[int(ranks[k])] = 1
                for j, token in enumerate(tweet[:timestep]):
                    if token in embed_dict:
                        sentenceRow[:, j] = embed_dict[token.lower()]

                inputMatrix.append(sentenceRow)
                labelMatrix.append(label)

    with open(pickleInputDir, "wb") as f:
        pickle.dump(inputMatrix, f)

    with open(pickleLabelDir, "wb") as f:
        pickle.dump(labelMatrix, f)


def createGlovefromTweet(embed_dict, tweetText, embedding_dim=100,
                         timestep=25):
    """
    Method takes tweet and converts it in Glove embedding

    :param embed_dict: Glove dictionary
    :param timestep: Maximum sentence length
    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param tweetText: Tweeter text

    """
    tokens = nltk.word_tokenize(tweetText)
    tokens = [word.lower() for word in tokens]
    sentenceRow = np.zeros((embedding_dim, timestep))
    for j, token in enumerate(tokens[:timestep]):
        if token in embed_dict:
            sentenceRow[:, j] = embed_dict[token.lower()]
        else:
            tmp = np.ones(embedding_dim) / 20
            sentenceRow[:, j] = tmp
    return sentenceRow


def create_pair_combs(lst):
    """
    Create all pair combinations from tweets of different humor ranking.
    :param lst:
    :return: List of docuement pairs and the list of label pairs
    """
    index = 1
    pairs_words = []
    pairs_labels = []
    pairs_chrs = []

    for element1 in lst:
        for element2 in lst[index:]:
            if int(element1[-1]) == int(element2[-1]):
                continue
            if int(element1[-1]) < int(element2[-1]):
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element1[0], element2[0]),
                                                  axis=1)
                    chr_merged = np.concatenate((element1[1], element2[1]),
                                                axis=0)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.concatenate((element2[0], element1[0]),
                                                  axis=1)
                    chr_merged = np.concatenate((element2[1], element1[1]),
                                                axis=0)
                    pairs_labels.append(1)
            else:
                if np.random.random() > 0.5:
                    concatMatrix = np.concatenate((element2[0], element1[0]),
                                                  axis=1)
                    chr_merged = np.concatenate((element2[1], element1[1]),
                                                axis=0)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.concatenate((element1[0], element2[0]),
                                                  axis=1)
                    chr_merged = np.concatenate((element1[1], element2[1]),
                                                axis=0)
                    pairs_labels.append(1)

            # Add word and char level information
            pairs_words.append(concatMatrix)
            pairs_chrs.append(chr_merged)

        index += 1
    return pairs_words, pairs_chrs, pairs_labels


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

        # glove = loadGlove("./resources/glove/glove.twitter.27B.100d.txt")
        # createGlovefromTweet(glove, "Gugi is smart boy")

        # if __name__ == "__main__":
        #   prepare_dataset_for_taskB("./resources/glove/glove.twitter.27B.100d.txt",
        #             "../dataset/", "./trainDamir.pkl", "./trainDamir2.pkl")
