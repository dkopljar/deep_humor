import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
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
        logging.info("ModelEvaluator restored from " + model_path)

    def train(self):
        """
        Sets up the model training and runs training.
        :return:
        """
        logging.info("Training Started")
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        best_val_loss = self.eval(self.valid_word,
                                  self.valid_char,
                                  self.valid_label)
        early_stop_counter = 0

        num_batches = self.train_word.shape[0] // self.batch_size
        logging.info("Number of batches " + str(num_batches))

        for epoch in range(self.num_epochs):
            # Shuffle training data in each epoch
            self.train_word, self.train_char, self.train_label = utils.shuffle_data(
                [
                    self.train_word,
                    self.train_char,
                    self.train_label])

            for b in range(num_batches):
                word = self.train_word[
                       b * self.batch_size:(b + 1) * self.batch_size]
                char = self.train_char[
                       b * self.batch_size:(b + 1) * self.batch_size]
                label = self.train_label[
                        b * self.batch_size:(b + 1) * self.batch_size]
                loss, _, lr = self.sess.run(
                    [self.loss, self.train_op, self.lr],
                    {self.word_embedding_input: word,
                     self.chr_embedding_input: char,
                     self.labels: label})

                if (b + 1) % 15 == 0:
                    logging.info(
                        "Iteration {}/{}, Batch Loss {:.4f}, LR: {:.4f}".format(
                            b * self.batch_size, num_batches * self.batch_size,
                            loss, lr))

            current_val_loss = self.eval(self.valid_word,
                                         self.valid_char,
                                         self.valid_label)
            if current_val_loss < best_val_loss:
                # If the validation loss is better
                best_val_loss = current_val_loss
                early_stop_counter = 0
                # Save model every n epochs
                path = saver.save(self.sess,
                                  os.path.join(self.model_save_dir,
                                               self.model_name + "_v_loss_" + str(
                                                   best_val_loss) + ".ckpt"),
                                  global_step=self.global_step)
                logging.info("Model saved at: " + str(path))
            else:
                early_stop_counter += 1

            logging.info("Finished epoch {}\n".format(epoch + 1))

            if early_stop_counter >= self.early_stop_threshold:
                logging.info("Early stopping the model")
                break

    def eval(self, input, input_chr, labels):
        """
        Evalutates the model using Accuracy, Precision, Recall and F1 metrics.
        :param input: Input of shape [batch_size, timestep, vector_dim]
        :param
        :return:
        """
        logging.info("Evaluating on the validation set...")
        num_batches = input.shape[0] // self.batch_size
        input, input_chr, labels = utils.shuffle_data(
            [input, input_chr, labels])
        acc, prec, rec, f1, loss_sum = 0, 0, 0, 0, 0
        for b in range(num_batches):
            word_b = input[
                     b * self.batch_size:(b + 1) * self.batch_size]
            char_b = input_chr[
                     b * self.batch_size:(b + 1) * self.batch_size]
            label_b = labels[
                      b * self.batch_size:(b + 1) * self.batch_size]
            loss, pred = self.sess.run(
                [self.loss, self.softmax],
                {self.word_embedding_input: word_b,
                 self.chr_embedding_input: char_b,
                 self.labels: label_b})

            # Update metric
            a, p, r, f = utils.calc_metric(np.argmax(pred, axis=1),
                                           np.argmax(label_b, axis=1))
            acc += a
            prec += p
            rec += r
            f1 += f
            loss_sum += loss

        logging.info("Accuracy {:.3f}%".format(acc / num_batches * 100))
        logging.info("Macro Precision {:.3f}%".format(prec / num_batches * 100))
        logging.info("Macro Recall {:.3f}%".format(rec / num_batches * 100))
        logging.info("Macro F1 {:.3f}%".format(f1 / num_batches * 100))
        logging.info("Average loss {:.5f}\n".format(loss_sum / num_batches))

        return loss_sum / num_batches


class Baseline(Net):
    """
    Baseline model.

    The model uses random guessing to predict humor ranking.
    Expected metric results are ~50%.
    """

    def __init__(self, config):
        self.train_word = config['train_word']
        self.valid_word = config['valid_word']
        self.train_label = config['train_label']
        self.valid_label = config['valid_label']
        self.batch_size = config['batch_size']

    def train(self):
        print("Evaluation on the train set")
        x, y = self.train_word.shape[0], self.train_label.shape[1]
        prediction_train = self.random_guess(x, y)
        self.eval(prediction_train, self.train_label)

        print("Evaluation on the  validation set")
        x, y = self.valid_word.shape[0], self.valid_label.shape[1]
        prediction_valid = self.random_guess(x, y)
        self.eval(prediction_valid, self.valid_label)

    def random_guess(self, num_examples, num_classes):
        dim_input = num_examples * num_classes
        prediction_zeros = np.zeros(dim_input // 2)
        prediction_ones = np.ones(dim_input // 2)
        prediction = np.hstack((prediction_ones, prediction_zeros))
        np.random.shuffle(prediction)
        return np.reshape(prediction, (num_examples, num_classes))

    def eval(self, input, labels):
        """
        Evalutates the model using Accuracy, Precision, Recall and F1 metrics.
        :param input: Input of shape [batch_size, timestep, vector_dim]
        :param
        :return:
        """
        pred = input
        acc, prec, rec, f1 = utils.calc_metric(np.argmax(pred, axis=1),
                                               np.argmax(labels, axis=1))

        logging.info("Accuracy {:.3f}%".format(acc * 100))
        logging.info("Macro Precision {:.3f}%".format(prec * 100))
        logging.info("Macro Recall {:.3f}%".format(rec * 100))
        logging.info("Macro F1 {:.3f}%\n".format(f1 * 100))


class BILSTM_FC(Net):
    """
    Glove word embeddings -> Bi-LSTM -> FCs architecture, extract only the
    last BILSTM layer.
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
        self.model_name = config["domain"]

        self.early_stop_threshold = config['early_stopping']

        self.train_word = config['train_word']
        self.valid_word = config['valid_word']
        self.train_char = config['train_chr']
        self.valid_char = config['valid_chr']
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
                                                    self.timestep),
                                                   name="input")
        self.tag_embedding = tf.placeholder(tf.float32,
                                            (None, self.word_embd_vec),
                                            name="labels")
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
        net = tf.matmul(net[-1], weights['out']) + biases['out']

        # Concat Tag embedding with LSTM output
        # TODO Add tag embeddings
        # net = tf.concat([net, self.tag_embedding], axis=0, name="concat1")

        # FC Layers
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.dense(inputs=net,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              activation=tf.nn.relu,
                              units=512)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.dense(inputs=net,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              activation=tf.nn.relu,
                              units=256)
        net = tf.layers.dropout(net, rate=0.5)

        # Logits and softmax
        logits = tf.layers.dense(inputs=net,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 units=self.n_classes,
                                 name="logits")
        # Probabilites
        self.softmax = tf.nn.softmax(logits, name="softmax")

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


class CNN_FC(Net):
    """
    word_embeddings -> CNN -> -> FC architecture,
    extract only the last BILSTM
    layer.
    """

    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector_dim"]
        self.char_timestep = config['char_timestep']
        self.char_embedding_dim = config['char_embeddings_dim']
        self.char_vocab_size = config['char_vocab_size']
        self.early_stop_threshold = config['early_stopping']

        self.n_classes = config["n_classes"]
        self.train_examples = config["train_examples"]
        self.batch_size = config["batch_size"]
        self.model_name = config["domain"]

        # TODO FIx naming
        self.train_word = config['train_chr']
        self.valid_word = config['valid_chr']
        # self.train_char = config['train_chr']
        # self.valid_char = config['valid_chr']
        self.train_label = config['train_label']
        self.valid_label = config['valid_label']
        self.num_epochs = config['train_epochs']
        self.model_save_dir = config['save_dir']

        self.setup()

    def setup(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)

        # Define inputs
        """
        Char embeddings input of size (batch_size, timestep, word_embed_dim)
        """

        # TODO Change naming
        self.word_embedding_input = tf.placeholder(tf.int32,
                                                   (None, self.char_timestep),
                                                   name="input")
        self.labels = tf.placeholder(tf.int32, (None, self.n_classes))

        # Char embedding layer
        char_embed = tf.Variable(
            tf.random_uniform(
                [self.char_vocab_size, self.char_embedding_dim],
                minval=-np.sqrt(3 / self.char_embedding_dim),
                maxval=np.sqrt(3 / self.char_embedding_dim)),
            name="char_embedding")

        net = tf.nn.embedding_lookup(char_embed, self.word_embedding_input)
        net = slim.dropout(net, keep_prob=0.5, scope="dropout1")
        net = tf.expand_dims(net, axis=3)

        # Network layers
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.repeat(net, 2, slim.conv2d, 64,
                              [self.char_timestep, 5], scope='conv1')
            net = slim.max_pool2d(net, [1, 2], scope='pool1')

            net = slim.repeat(net, 1, slim.conv2d, 128,
                              [self.char_timestep, 3], scope='conv2')
            net = slim.max_pool2d(net, [1, 4], scope='pool1')

            # FC layers
            net = slim.flatten(net, scope="flatten3")
            net = slim.fully_connected(net, 512, scope='fc3')
            net = slim.dropout(net, keep_prob=0.5, scope="dropout4")
            logits = slim.fully_connected(net, self.n_classes,
                                          activation_fn=None,
                                          scope='logits')

        # Probabilities
        self.softmax = tf.nn.softmax(logits, name="softmax")

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


class CNN_BILST_FC(Net):
    """
    word_embeddings -> CNN -> -> FC architecture,
    extract only the last BILSTM
    layer.
    """

    def __init__(self, config):
        self.learning_rate = config["lr"]
        self.optimizer = config["optimizer"]
        self.timestep = config["timestep"]
        self.word_embd_vec = config["word_vector_dim"]
        self.char_timestep = config['char_timestep']
        self.char_embedding_dim = config['char_embeddings_dim']
        self.lstm_hidden = config["lstm_hidden"]
        self.char_vocab_size = config['char_vocab_size']

        self.n_classes = config["n_classes"]
        self.train_examples = config["train_examples"]
        self.batch_size = config["batch_size"]
        self.model_name = config["domain"]
        self.early_stop_threshold = config['early_stopping']

        self.train_word = config['train_word']
        self.valid_word = config['valid_word']
        self.train_char = config['train_chr']
        self.valid_char = config['valid_chr']
        self.train_label = config['train_label']
        self.valid_label = config['valid_label']
        self.num_epochs = config['train_epochs']
        self.model_save_dir = config['save_dir']

        self.setup()

    def setup(self):
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False)

        # Define inputs
        """
        Char embeddings input of size (batch_size, timestep, word_embed_dim)
        """

        self.word_embedding_input = tf.placeholder(tf.float32,
                                                   (None, self.word_embd_vec,
                                                    self.timestep),
                                                   name="input_word")

        self.chr_embedding_input = tf.placeholder(tf.int32,
                                                  (None, self.char_timestep),
                                                  name="input_char")
        self.labels = tf.placeholder(tf.int32, (None, self.n_classes))

        # Char embedding layer
        char_embed = tf.Variable(
            tf.random_uniform(
                [self.char_vocab_size, self.char_embedding_dim],
                minval=-np.sqrt(3 / self.char_embedding_dim),
                maxval=np.sqrt(3 / self.char_embedding_dim)),
            name="char_embedding")

        net = tf.nn.embedding_lookup(char_embed, self.chr_embedding_input)
        net = slim.dropout(net, keep_prob=0.4, scope="dropout1")
        net = tf.expand_dims(net, axis=3)

        # Network layers
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.repeat(net, 1, slim.conv2d, 64,
                              [self.char_timestep, 5], scope='conv1')
            net = slim.max_pool2d(net, [1, 4], scope='pool1')

            net = slim.repeat(net, 1, slim.conv2d, 128,
                              [self.char_timestep, 3], scope='conv2')
            net = slim.max_pool2d(net, [1, 2], scope='pool2')

            # FC layers
            net_cnn = slim.flatten(net, scope="flatten3")

        weights_output_dim = 300
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward +
            # backward cells
            'out': tf.Variable(
                tf.random_uniform([2 * self.lstm_hidden, weights_output_dim],
                                  minval=-np.sqrt(6 / (
                                      2 * self.lstm_hidden + weights_output_dim)),
                                  maxval=np.sqrt(6 / (
                                      2 * self.lstm_hidden + weights_output_dim)))
            )
        }
        biases = {
            'out': tf.Variable(tf.zeros([weights_output_dim]))
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
        net_rnn = slim.flatten(slim.dropout(
            tf.matmul(net[-1], weights['out']) + biases['out'], keep_prob=0.4))

        # Merge CNN and RNN features
        net = tf.concat([net_cnn, net_rnn], axis=1)

        # FC layers
        net = slim.fully_connected(net, 512, scope='fc3')
        net = slim.dropout(net, keep_prob=0.4, scope="dropout4")

        logits = slim.fully_connected(net, self.n_classes,
                                      activation_fn=None,
                                      scope='logits')

        # Probabilities
        self.softmax = tf.nn.softmax(logits, name="softmax")

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
