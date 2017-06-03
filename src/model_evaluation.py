import os
import time

import numpy as np
import tensorflow as tf

import constants


class ModelEvaluator:
    def __init__(self, model_path, pred_op_name="softmax:0",
                 input_word_op_name="input:0",
                 input_char_op_name="input_char:0"):
        """
        :param input_op_name: Input tensor name from the TF Graph
        :param pred_op_name: Prediction operation name from the TF Graph
        :param model_path: ModelEvaluator path including the model name + step. E.g.
        'my-save-dir/my-model-10000'
        """

        # Create a new session and restore the graph + variables
        self.sess = tf.Session()
        new_saver = tf.train.import_meta_graph(model_path + ".meta")
        new_saver.restore(self.sess, model_path)
        graph = tf.get_default_graph()


        # Restore ops for input and prediction
        self.pred_op = graph.get_tensor_by_name(pred_op_name)
        self.input_op_char = graph.get_tensor_by_name(input_char_op_name)
        self.input_op_word = graph.get_tensor_by_name(input_word_op_name)

        print("ModelEvaluator restored from " + model_path)

    def predict(self, input_word, input_char, batch_size=32):
        """
        Outputs the model prediction class for all examples.
        :param batch_size: Mini-batch size
        :param input: Numpy array of the all input example
        :return: Prediction result indexes (class), varies on the model type
        """

        size = input_word.shape[0]
        num_batches = size // batch_size

        predictions = []
        for b in range(num_batches):
            input_word_b = input_word[b * batch_size:(b + 1) * batch_size]
            input_char_b = input_char[b * batch_size:(b + 1) * batch_size]
            prediction = self.sess.run([self.pred_op],
                                       feed_dict={
                                           self.input_op_char: input_char_b,
                                           self.input_op_word: input_word_b})[0]

            predictions.extend(np.argmax(prediction, axis=1))

            if b % 20 == 0:
                print("Processed batch {}/{}".format(b, num_batches))

        # Last few examples didn't fit into any of the batches
        if batch_size * num_batches < size:
            input_word_b = input_word[batch_size * num_batches:]
            input_char_b = input_char[batch_size * num_batches:]
            prediction = self.sess.run([self.pred_op],
                                       feed_dict={
                                           self.input_op_char: input_char_b,
                                           self.input_op_word: input_word_b})[0]
            predictions.extend(np.argmax(prediction, axis=1))

        assert len(
            predictions) == size, "Number of predicted and input examples does not match"

        return np.array(predictions)

    def predict_regression(self, input_word, input_char, batch_size=32,
                           class_num=3):
        """
        Outputs the model prediction probabilitse for all examples.
        :param batch_size: Mini-batch size
        :param input: Numpy array of the all input example
        :return: Prediction result indexes (class), varies on the model type
        """

        size = input_word.shape[0]
        num_batches = size // batch_size

        predictions = np.ones((input_word.shape[0], class_num))

        for b in range(num_batches):
            input_word_b = input_word[b * batch_size:(b + 1) * batch_size]
            input_char_b = input_char[b * batch_size:(b + 1) * batch_size]
            prediction = self.sess.run([self.pred_op],
                                       feed_dict={
                                           self.input_op_char: input_char_b,
                                           self.input_op_word: input_word_b})[0]

            predictions[b * batch_size: (b + 1) * batch_size] = prediction

            if b % 20 == 0:
                print("Processed batch {}/{}".format(b, num_batches))

        # Last few examples didn't fit into any of the batches
        if batch_size * num_batches < size:
            input_word_b = input_word[batch_size * num_batches:]
            input_char_b = input_char[batch_size * num_batches:]
            prediction = self.sess.run([self.pred_op],
                                       feed_dict={
                                           self.input_op_char: input_char_b,
                                           self.input_op_word: input_word_b})[0]
            predictions[num_batches * batch_size:] = prediction

        assert len(
            predictions) == size, "Number of predicted and input examples does not match"

        return predictions

    def compare_regression(self, pairs):
        """
        Compares all predicted softmax pairs. All pairs are compared using a
        specilizaed scoring function which takes both ranking and confidence into
        account.
        :param pairs: Input of shape [num_pairs, 2, num_classes]
        :return:
        """
        pair_l = (np.argmax(pairs[0, :], axis=1) + 1) * np.max(pairs[0, :],
                                                               axis=1)
        pair_r = (np.argmax(pairs[1, :], axis=1) + 1) * np.max(pairs[1, :],
                                                               axis=1)

        return np.array(pair_r > pair_l, dtype=np.uint8)


if __name__ == "__main__":
    # Example of usage
    path = os.path.join(constants.TF_WEIGHTS,
                        "CNN_BILSTM_FC_model_v_loss_0.289724265536.ckpt-1872")
    model = ModelEvaluator(path)

    start = time.time()

    # Simulate prediction
    num_examples = 100
    data = np.ones((num_examples, 100, 25))
    data_char = np.ones((num_examples, 80))
    result = model.predict_regression(data, data_char,
                                      batch_size=32)  # returns 0 or 1:
    print("Prediction time for all examples [s]", time.time() - start)

    # Test pair comparison
    print(model.compare_regression(
        np.array([[result[0], result[2]], [result[1], result[3]]])))
