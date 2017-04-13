import datetime
import logging
import os
import pdb
import pickle

import numpy as np

import constants
import dataset_parser
import models
import utils


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

    X, y = [], []
    for topic in topicsMatrix:
        comb_m, comb_l = dataset_parser.combinantorial(topic)
        X.extend(comb_m)
        y.extend(
            [utils.int_to_one_hot(x, config['n_classes']) for x in comb_l])

    X, y = np.array(X), np.array(y)

    # X, y = utils.shuffle_data(X, y)
    x_train, x_valid, x_test, y_train, y_valid, y_test = utils.split_data(X, y)

    # Mock data
    config['train_examples'] = x_train.shape[0]
    config['validation_examples'] = x_valid.shape[0]
    config['test_examples'] = 2

    config['train_word'] = x_train
    config['valid_word'] = x_valid[:5000]
    config['train_label'] = y_train
    config['valid_label'] = y_valid[:5000]
    config['save_dir'] = os.path.join(constants.TF_WEIGHTS)

    logging.info("CONFIG:")
    logging.info("\n".join([k + ": " + str(v) for k, v in config.items()]))

    net = models.LSTM(config)
    net.train()

    # Evaluate on the TEST set
    # net.eval(test_input, test_label)


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
                        level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')

    logging.info("Numpy random seed set to " + str(seed))
    main(config=config)
