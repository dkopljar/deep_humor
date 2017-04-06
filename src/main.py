import logging
import numpy as np
import utils
import constants
import os
import datetime
import models


def main(config):
    """

    :param config: Loaded configuration dictionary
    :return:
    """

    # Mock data
    config['n_classes'] = 2
    config['train_examples'] = 80000
    config['validation_examples'] = 10000
    config['test_examples'] = 5000

    config['train_word'] = None
    config['valid_word'] = None
    config['train_label'] = None
    config['valid_label'] = None
    config['save_dir'] = os.path.join(constants.TF_WEIGHTS)

    logging.info("CONFIG:")
    logging.info("\n".join([k + ": " + str(v) for k, v in config.items()]))

    net = models.LSTM(config)

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
