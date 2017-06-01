class StatisticalTest:
    def __init__(self, results_file):
        """
        :param results_file: File with results
        """
        # TODO process results
        pass
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def get_mean(self, confidence):
        """
        Returns the mean +- SE for the given confidence.
        :return: Tuple of values for acc, prec, rec and f1
        """
        pass

    def p_test(self, test2, p_value=0.05):
        """
        Does a statistical p test for the given StatisticalTest object.
        Compares all metrics.
        :param test2: StatisticalTest object to compare with
        :return: List of bool values for each metric. True if the test is
        the hypothesis is accepted, False otherwise
        """
        pass
