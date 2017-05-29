import os
import time

import numpy as np
import tensorflow as tf

import constants


class Model:
    def __init__(self, model_path, pred_op_name="softmax:0",
                 input_word_op_name="input_word:0",
                 input_char_op_name="input_char:0"):
        """
        :param input_op_name: Input tensor name from the TF Graph
        :param pred_op_name: Prediction operation name from the TF Graph
        :param model_path: Model path including the model name + step. E.g.
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

        print("Model restored from " + model_path)

    def predict(self, input_word, input_char):
        """
        Outputs the model prediction class for one example.
        :param input: Numpy array of the ONE input example
        :return: Prediction result index (class), varies on the model type
        """

        # Create a batch of a size 1
        input_char = np.array([input_char], dtype=np.float32)
        input_word = np.array([input_word], dtype=np.float32)
        prediction = self.sess.run([self.pred_op],
                                   feed_dict={self.input_op_char: input_char,
                                              self.input_op_word: input_word})

        # TODO may need some editing due to the dataset conventions
        return np.argmax(prediction[0])


if __name__ == "__main__":
    # Example of usage
    path = os.path.join(constants.TF_WEIGHTS, "CNN_BILSTM_FC_model.ckpt-1320")
    model = Model(path)

    start = time.time()

    # Simulate prediction
    data = np.ones((100, 50))
    data_char = np.ones((140))
    num_examples = 1000
    for i in range(num_examples):
        result = model.predict(data, data_char)  # returns 0 or 1:

    print("Prediction time for all examples [s]", time.time() - start)
