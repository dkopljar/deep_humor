import os

import numpy as np
import scipy.stats as st

import constants
import utils


class StatisticalTest:
    def __init__(self, results_file, k_folds=35):
        """
        :param results_file: File with results
        """
        acc, p, r, f1 = utils.read_log_file(results_file)

        self.file_name = results_file
        self.accuracies = acc
        self.precisions = p
        self.recalls = r
        self.f1s = f1

        assert len(self.accuracies) == k_folds
        assert len(self.precisions) == k_folds
        assert len(self.recalls) == k_folds
        assert len(self.f1s) == k_folds

    def get_mean(self, data, confidence=0.95):
        """
        Returns the mean +- SE for the given confidence.
        :return: Tuple of values for acc, prec, rec and f1
        """

        interval = st.norm.interval(confidence, loc=np.mean(data),
                                    scale=st.sem(data))
        mean = np.mean(interval)
        return str("{:.3f} +- {:.5f}".format(mean, interval[1] - mean))

    def p_test(self, test2, p_value=0.05):
        """str(mean)
        Does a statistical p test for the given StatisticalTest object.
        Compares all metrics.
        :param test2: StatisticalTest object to compare with
        :return: List of bool values for each metric. True if the test is
        the hypothesis is accepted, False otherwise
        """
        pass


if __name__ == "__main__":
    files = os.listdir(constants.LOGS)

    for file in files:
        test = StatisticalTest(os.path.join(constants.LOGS, file))
        print(test.file_name)
        print("Accuracy", test.get_mean(test.accuracies))
        print("Precision", test.get_mean(test.precisions))
        print("Recall", test.get_mean(test.recalls))
        print("F1", test.get_mean(test.f1s))
        print()
