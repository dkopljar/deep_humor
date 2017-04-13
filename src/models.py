import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import utils


class Net:
    def setup(self):
        """
        Sets up the
        :return:
        """
        raise NotImplementedError

    def load_model(self, model_path):
        """
        Restores the model from the checkpoint file
        :param model_path:
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        logging.info("Model restored from " + model_path)

    def train(self):
        """
        Sets up the model training and runs training.
        :return:
        """
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        num_batches = self.train_word.shape[0] // self.batch_size

        for epoch in range(self.num_epochs):
            # Shuffle training data in each epoch

            for b in range(num_batches):
                word = self.train_word[
                       b * self.batch_size:(b + 1) * self.batch_size]
                label = self.train_label[
                        b * self.batch_size:(b + 1) * self.batch_size]
                loss, _, lr = self.sess.run(
                    [self.loss, self.train_op, self.lr],
                    {self.word_embedding_input: word,
                     self.labels: label})

                if (b + 1) % 15 == 0:
                    logging.info(
                        "Iteration {}/{}, Batch Loss {:.4f}, LR: {:.4f}".format(
                            b * self.batch_size, num_batches * self.batch_size,
                            loss, lr))

            self.eval(self.valid_word, self.valid_label)
            logging.info("Finished epoch {}\n".format(epoch + 1))

            if (epoch + 1) % 10 == 0:
                # Save model every n epochs
                path = saver.save(self.sess,
                                  os.path.join(self.model_save_dir,
                                               "_cnn_bilstm_crf.ckpt"),
                                  global_step=self.global_step)
                logging.info("Model saved at: " + str(path))

    def eval(self, input, labels):
        """
        Evalutates the model using Accuracy, Precision, Recall and F1 metrics.
        :param input: Input of shape [batch_size, timestep, vector_dim]
        :param
        :return:
        """
        logging.info("Evaluating on the validation set...")
        # char, word, label = utils.shuffle_data(char, word, label)
        num_batches = input.shape[0] // self.batch_size
        acc, prec, rec, f1 = 0, 0, 0, 0
        for b in range(num_batches):
            word_b = input[
                     b * self.batch_size:(b + 1) * self.batch_size]
            label_b = labels[
                      b * self.batch_size:(b + 1) * self.batch_size]
            loss, pred = self.sess.run(
                [self.loss, self.softmax],
                {self.word_embedding_input: word_b,
                 self.labels: label_b})

            # Update metric
            a, p, r, f = utils.calc_metric(np.argmax(pred, axis=1),
                                           np.argmax(label_b, axis=1))
            acc += a
            prec += p
            rec += r
            f1 += f

        logging.info("Accuracy {:.3f}%".format(acc / num_batches * 100))
        logging.info("Macro Precision {:.3f}%".format(prec / num_batches * 100))
        logging.info("Macro Recall {:.3f}%".format(rec / num_batches * 100))
        logging.info("Macro F1 {:.3f}%\n".format(f1 / num_batches * 100))


class BILSTM(Net):
    """
    Glove word embeddings -> Bi-LSTM architecture, extract only the last BILSTM
    layer.
    """

    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector_dim"]
        self.max_word_size = config["max_word_size"]
        self.lstm_hidden = config["lstm_hidden"]
        self.n_classes = config["n_classes"]
        self.train_examples = config["train_examples"]
        self.batch_size = config["batch_size"]

        self.train_word = config['train_word']
        self.valid_word = config['valid_word']
        self.train_label = config['train_label']
        self.valid_label = config['valid_label']
        self.num_epochs = config['train_epochs']
        self.model_save_dir = config['save_dir']

        self.setup()

    def setup(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)

        """
        Word embeddings input of size (batch_size, timestep, word_embed_dim)
        """
        self.word_embedding_input = tf.placeholder(tf.float32,
                                                   (None, self.word_embd_vec,
                                                    self.timestep))
        # POS tags encoded in one-hot fashion (batch_size, num_classes)
        self.labels = tf.placeholder(tf.int32, (None, self.n_classes))

        # BI-BILSTM
        # Define weights for the Bi-directional BILSTM
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward +
            # backward cells
            'out': tf.Variable(
                tf.random_uniform([2 * self.lstm_hidden, self.n_classes],
                                  minval=-np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)),
                                  maxval=np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)))
            )
        }
        biases = {
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        net = tf.reshape(self.word_embedding_input,
                         [-1, self.timestep * self.word_embd_vec],
                         name="reshape1")
        net = tf.split(net, self.timestep, axis=1, name="split1")

        # Forward and backward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)

        net, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=net,
                                                 dtype=tf.float32)

        # Linear activation, using rnn inner loop on the final output
        logits = tf.matmul(net[-1], weights['out']) + biases['out']

        # Probabilites
        self.softmax = tf.nn.softmax(logits)

        # Loss and learning rate
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                    logits=logits))
        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=self.train_examples // self.batch_size,
                                             decay_rate=0.95)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.loss,
                                            global_step=self.global_step)


class CNNBiLSTM(Net):
    """
    Char embeddings -> CNN -> word + char features -> Bi-LSTM architecture,
    extract only the last BILSTM
    layer.
    """

    # TODO

    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector_dim"]
        self.max_word_size = config["max_word_size"]
        self.lstm_hidden = config["lstm_hidden"]
        self.n_classes = config["n_classes"]
        self.train_examples = config["train_examples"]
        self.batch_size = config["batch_size"]
        self.char_vocab_dim = config['chr_vocab_dim']
        self.char_features = config['chr_features']
        self.cnn_filter = config['cnn_filter']

        self.train_word = config['train_word']
        self.valid_word = config['valid_word']
        self.train_label = config['train_label']
        self.valid_label = config['valid_label']
        self.num_epochs = config['train_epochs']
        self.model_save_dir = config['save_dir']

        self.setup()

    def setup(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)

        """
        Word embeddings input of size (batch_size, timestep, word_embed_dim)
        """
        self.word_embedding_input = tf.placeholder(tf.float32,
                                                   (None, self.word_embd_vec,
                                                    self.timestep))

        # POS tags encoded in one-hot fashion (batch_size, num_classes)
        self.labels = tf.placeholder(tf.int32, (None, self.n_classes))

        """
        Character embeddings input of size (batch_size, max_sentence_length
         (a.k.a. timestep) * max_word_size)
        """
        self.char_embedding_input = tf.placeholder(tf.int32,
                                                   (None,
                                                    self.timestep * self.max_word_size))
        # POS tags encoded in one-hot fashion (batch_size, num_classes)
        self.labels = tf.placeholder(tf.int32,
                                     (None, self.timestep, self.n_classes))

        # Char embedding layer
        char_embed = tf.Variable(
            tf.random_uniform(
                [self.char_vocab_dim, self.char_features],
                minval=-np.sqrt(3 / self.char_features),
                maxval=np.sqrt(3 / self.char_features)),
        )
        net = tf.nn.embedding_lookup(char_embed, self.char_embedding_input)

        # 1-D Convolution on a character level
        net = tf.layers.conv1d(
            inputs=net,
            filters=self.cnn_filter,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            name="conv1")
        net = tf.layers.max_pooling1d(net,
                                      pool_size=2,
                                      strides=2,
                                      name="pool1")
        net = tf.reshape(net, [-1, self.timestep, self.cnn_filter * 10],
                         name="reshape1")

        # Concatenate word and char-cnn embeddings
        net = tf.concat([self.word_embedding_input, net], axis=2,
                        name="concat1")

        # Apply dropout and prepare input for the BI-LSTM net
        net = tf.reshape(net, [-1, self.cnn_filter * 10 + self.word_embd_vec],
                         name="reshape2")
        net = tf.split(net, self.timestep, axis=0, name="split1")

        # BI-BILSTM
        # Define weights for the Bi-directional BILSTM
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward +
            # backward cells
            'out': tf.Variable(
                tf.random_uniform([2 * self.lstm_hidden, self.n_classes],
                                  minval=-np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)),
                                  maxval=np.sqrt(6 / (
                                      2 * self.lstm_hidden + self.n_classes)))
            )
        }
        biases = {
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # Forward and backward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden, forget_bias=1.0)

        net, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=net,
                                                 dtype=tf.float32)

        # Linear activation, using rnn inner loop on the final output
        logits = tf.matmul(net[-1], weights['out']) + biases['out']

        # Probabilites
        self.softmax = tf.nn.softmax(logits)

        # Loss and learning rate
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                    logits=logits))
        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=self.train_examples // self.batch_size,
                                             decay_rate=0.95)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.loss,
                                            global_step=self.global_step)
