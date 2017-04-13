import datetime
import logging
import os
import pickle
import sys

import numpy as np

import constants
import dataset_parser
import models
import utils


def create_data_pairs(topics_matrix):
    """
    Create data pair combinations
    """
    X, y = [], []
    for topic in topics_matrix:
        comb_m, comb_l = dataset_parser.combinantorial(topic)
        X.extend(comb_m)
        y.extend(
            [utils.int_to_one_hot(x, config['n_classes']) for x in comb_l])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.uint8)


def main(config):
    """
    :param config: Loaded configuration dictionary
    :return:
    """
    config['n_classes'] = 2
    pickle_pairs = os.path.join(constants.DATA, "pickled",
                                "train_pairs.pkl")
    with open(pickle_pairs, "rb") as f:
        topicsMatrix = pickle.load(f)
    X, y = create_data_pairs(topicsMatrix)

    # Shuffle and split
    X, y = utils.shuffle_data(X, y)
    x_train, x_valid, x_test, y_train, y_valid, y_test = utils.split_data(X, y)

    # Mock data
    config['train_examples'] = x_train.shape[0]
    config['validation_examples'] = x_valid.shape[0]
    config['test_examples'] = x_test.shape[0]
    config['save_dir'] = os.path.join(constants.TF_WEIGHTS)

    # Log configuration
    logging.info("CONFIG:")
    logging.info("\n".join([k + ": " + str(v) for k, v in config.items()]))

    # Add datasets to config
    config['train_word'] = x_train
    config['valid_word'] = x_valid
    config['train_label'] = y_train
    config['valid_label'] = y_valid

    net = models.BILSTM(config)
    net.train()

    # Evaluate on the TEST set
    net.eval(y_test, y_train)


if __name__ == "__main__":
    seed = 1337
    np.random.seed(seed)
    config = utils.read_config(os.path.join(constants.CONFIGS, "word_lstm.ini"))

    # Setup logging
    utils.dir_creator([constants.LOGS, constants.TF_WEIGHTS])

    log_name = config['domain'] + "_" + str(
        datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")) + ".log"
    log_file = os.path.join(constants.LOGS, log_name)
    print("Logging to", log_file)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p',
                        stream=sys.stdout)

    logging.info("Numpy random seed set to " + str(seed))
    main(config=config)
