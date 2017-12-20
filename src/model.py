# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import logging


class LSTMSRLer:
    def __init__(self, configure):
        # initialize configuration
        self.config = configure

        # initialize embedding variables
        self.word_embedding = tf.get_variable("Embed_word",
                                              shape=(self.config['vocab_size'], self.config['embedding_dim']),
                                              dtype='float32', initializer=tf.random_normal_initializer())
        self.postag_embedding = tf.get_variable("Embed_postag", shape=(34, self.config['postag_dim']), dtype='float32',
                                                initializer=tf.random_normal_initializer())  # postag size is fixed to 32+2(eos, bos)
        self.distance_embedding = tf.get_variable("Embed_dist", shape=(240, self.config['distance_dim']),
                                                  dtype='float32',
                                                  initializer=tf.random_normal_initializer())  # we observed in training set max dist is 240

        # initialize fully connected variables
        total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim'] + self.config['distance_dim']
        self.W1 = tf.get_variable("W_1", shape=(total_embed_dim, self.config['fc_hidden1_dim']),
                                  initializer=tf.random_normal_initializer())
        self.b1 = tf.get_variable("b_1", shape=(self.config['fc_hidden1_dim'],),
                                  initializer=tf.random_normal_initializer())
        self.W2 = tf.get_variable("W_2", shape=(2 * self.config['RNN_dim'], self.config['fc_hidden2_dim']),
                                  initializer=tf.random_normal_initializer())
        self.b2 = tf.get_variable("b_2", shape=(self.config['fc_hidden2_dim'],),
                                  initializer=tf.random_normal_initializer())
        self.W3 = tf.get_variable("W_3", shape=(self.config['fc_hidden2_dim'], self.config['n_label']),
                                  initializer=tf.random_normal_initializer())
        self.b3 = tf.get_variable("b_3", shape=(self.config['n_label'],),
                                  initializer=tf.random_normal_initializer())

        # initialize RNN cell
        self.fw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim'])
        self.bw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim'])

        # initialize transition rule
        self.make_transition_rule()

    def make_transition_rule(self):
        # make a matrix to indicate whether a transition is legal
        trans = np.ones(shape=(67, 67), dtype="float32")
        # B flag can only be followed with I or E with same name
        trans[:17, :] = 0
        trans[:17, 17:34] = np.diag(np.ones(17))
        trans[:11, 34:45] = np.diag(np.ones(11))
        trans[12:17, 45:50] = np.diag(np.ones(5))

        # E, O, S, rel flag can be followed with any labels except E and I
        trans[17:34, 17:50] = 0
        trans[50:67, 17:50] = 0

        # I flag can only be followed with I or E with same name
        trans[34:50, :] = 0
        trans[34:50, 34:50] = np.diag(np.ones(16))
        trans[34:45, 17:28] = np.diag(np.ones(11))
        trans[45:50, 29:34] = np.diag(np.ones(5))

        self.transition_rules = trans

    def compile(self, training):
        # define placeholder
        self.curword = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.lastword = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.nextword = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.predicate = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.curpostag = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.lastpostag = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.nextpostag = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))
        self.distance = tf.placeholder(dtype="int32", shape=(None, self.config['max_len']))

        self.seq_length = tf.placeholder(dtype="int32", shape=(None,))

        if training:
            self.label = tf.placeholder(dtype="int32",
                                        shape=(None, self.config['max_len'],
                                               self.config['n_label']))  # one-hot encode
        self.mask = tf.placeholder(dtype="int32",
                                   shape=(None, self.config['max_len']))  # 0 for padding words

        # get representation
        curword_emb = tf.nn.embedding_lookup(self.word_embedding, self.curword)
        lastword_emb = tf.nn.embedding_lookup(self.word_embedding, self.lastword)
        nextword_emb = tf.nn.embedding_lookup(self.word_embedding, self.nextword)
        predicate_emb = tf.nn.embedding_lookup(self.word_embedding, self.predicate)
        curpos_emb = tf.nn.embedding_lookup(self.postag_embedding, self.curpostag)
        lastpos_emb = tf.nn.embedding_lookup(self.postag_embedding, self.lastpostag)
        nextpos_emb = tf.nn.embedding_lookup(self.postag_embedding, self.nextpostag)
        dist_emb = tf.nn.embedding_lookup(self.distance_embedding, self.distance)
        embedding = tf.concat(
            [curword_emb, lastword_emb, nextword_emb, predicate_emb, curpos_emb, lastpos_emb, nextpos_emb, dist_emb],
            axis=2)

        # first fully connected layer
        total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim'] + self.config['distance_dim']
        hidden_1 = tf.matmul(tf.reshape(embedding, (-1, total_embed_dim)), self.W1) + self.b1
        hidden_1 = tf.tanh(hidden_1)
        hidden_1 = tf.reshape(hidden_1, (-1, self.config['max_len'], self.config['fc_hidden1_dim']))

        # recurrent layer
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_lstmcell, self.bw_lstmcell, hidden_1, self.seq_length,
                                                     dtype="float32")
        hidden_rnn = tf.concat(outputs, axis=2)

        # second fully connected layer
        hidden_2 = tf.matmul(tf.reshape(hidden_rnn, (-1, 2 * self.config['RNN_dim'])), self.W2) + self.b2
        hidden_2 = tf.tanh(hidden_2)

        # output layer
        logits = tf.matmul(hidden_2, self.W3) + self.b3  # (batch_size * max_len, n_label)

        if training:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(tf.reshape(self.label, shape=(-1, self.config['n_label'])), "float32"), logits=logits)
            loss = tf.reshape(loss, (-1, self.config['max_len']))
            loss = loss * tf.cast(self.mask, "float32")
            self.loss = tf.reduce_sum(loss)

            # initialize training op
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.config['lrate']).minimize(self.loss)
        else:
            self.outputs = tf.reshape(logits, (-1, self.config['max_len'], self.config['n_label']))

    def train(self, training_data, feats, labels, save_per_epoch):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=self.config['num_spoch'])
        lengths = np.array([len(sent) for sent in training_data], dtype=np.int32)
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.config['num_spoch']):
                logging.info("Epoch %d started:" % epoch)
                sum_loss = 0.
                ids = np.arange(len(training_data))
                np.random.shuffle(ids)
                batch_num = len(training_data) // self.config['batch_size']
                for iter in range(batch_num):
                    data_id = ids[iter * self.config['batch_size']:(iter + 1) * self.config['batch_size']]
                    features = feats[data_id]
                    length = lengths[data_id]
                    label = labels[data_id]
                    onehot_label = np.zeros(
                        shape=(self.config['batch_size'], self.config['max_len'], self.config['n_label']),
                        dtype=np.int32)
                    masks = np.zeros(shape=(self.config['batch_size'], self.config['max_len']), dtype=np.int32)
                    # rel_masks = np.zeros(shape=(self.config['batch_size'], self.config['max_len']), dtype=np.int32)
                    for i in range(self.config['batch_size']):
                        masks[i, 0:length[i]] = 1
                        # rel_masks[i, 0:lengths[i]] = features[i, 0:lengths[i], 7] == 0
                        for j in range(self.config['max_len']):
                            onehot_label[i, j, label[i, j]] = 1
                    feed_dict = {
                        self.curword: features[:, :, 0],
                        self.lastword: features[:, :, 1],
                        self.nextword: features[:, :, 2],
                        self.predicate: features[:, :, 3],
                        self.curpostag: features[:, :, 4],
                        self.lastpostag: features[:, :, 5],
                        self.nextpostag: features[:, :, 6],
                        self.distance: features[:, :, 7],
                        self.seq_length: length,
                        self.label: onehot_label,
                        self.mask: masks,
                        # self.rel_mask: rel_masks,
                    }
                    iter_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                    sum_loss += iter_loss
                    if iter % 10 == 0:
                        logging.info("Iter %d, training loss: %f" % (
                            iter, sum_loss * 1. / (1 + iter) / self.config['batch_size']))
                logging.info(
                    "Epoch %d, training loss: %f" % (epoch, sum_loss * 1. / batch_num / self.config['batch_size']))
                if save_per_epoch:
                    saver.save(sess, save_path=self.config['save_path'], global_step=epoch)
                    logging.info("Training checkpoint has been saved.")

    def test(self, testing_data, feats):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        lengths = np.array([len(sent) for sent in testing_data], dtype=np.int32)
        preds = []
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, save_path=self.config['load_path'])
            logging.info("Loaded model from %s" % self.config['load_path'])
            for i in range(len(testing_data)):
                if i % 50 == 0:
                    logging.info("Predicting testing sentence %d." % i)
                features = feats[[i]]
                length = lengths[[i]]
                masks = np.zeros(shape=(1, self.config['max_len']), dtype=np.int32)
                masks[0, 0:length[0]] = 1
                feed_dict = {
                    self.curword: features[:, :, 0],
                    self.lastword: features[:, :, 1],
                    self.nextword: features[:, :, 2],
                    self.predicate: features[:, :, 3],
                    self.curpostag: features[:, :, 4],
                    self.lastpostag: features[:, :, 5],
                    self.nextpostag: features[:, :, 6],
                    self.distance: features[:, :, 7],
                    self.seq_length: length,
                    self.mask: masks,
                }
                outputs = sess.run(self.outputs, feed_dict=feed_dict)[0]
                best_valid_labelseq = self.find_best(outputs, lengths[i], features[0, :, 7])
                preds.append(best_valid_labelseq)
        return preds

    def find_best(self, scores, length, dist):
        scores = scores.T  # (n_label, max_len)
        record = np.full(shape=(67, length), fill_value=-np.Inf)
        path = np.full(shape=(67, length), fill_value=-1, dtype=np.int32)
        pred = []
        for i in range(length):
            if i == 0:
                if dist[i] != 0:
                    record[0:17, i] = scores[0:17, i]
                    record[50:66, i] = scores[50:66, i]
                else:
                    record[66, i] = scores[66, i]
            else:
                if dist[i] != 0 and dist[i - 1] != 0:
                    for j in range(66):
                        max_score = -np.Inf
                        argmax_prev = -1
                        for k in range(66):
                            if self.transition_rules[k, j] and record[k, i - 1] + scores[j, i] > max_score:
                                max_score = record[k, i - 1] + scores[j, i]
                                argmax_prev = k
                        record[j, i] = max_score
                        path[j, i] = argmax_prev
                elif dist[i] != 0 and dist[i - 1] == 0:
                    record[0:17, i] = scores[0:17, i] + record[66, i-1]
                    path[0:17, i] = 66
                    record[50:66, i] = scores[50:66, i] + record[66, i-1]
                    path[50:66, i] = 66
                else:
                    max_score = -np.Inf
                    argmax_prev = -1
                    for k in range(66):
                        if self.transition_rules[k, 66] and record[k, i - 1] + scores[66, i] > max_score:
                            max_score = record[k, i - 1] + scores[66, i]
                            argmax_prev = k
                    record[66, i] = max_score
                    path[66, i] = argmax_prev
                if i == length - 1:
                    record[0:17, i] = -np.Inf
                    path[0:17, i] = -1
                    record[34:50, i] = -np.Inf
                    path[34:50, i] = -1
        move = np.argmax(record[:, -1])
        pred.append(move)
        while len(pred) < length:
            assert record[move, length - len(pred)] != -np.Inf and path[move, length - len(pred)] != -1
            move = path[move, length - len(pred)]
            pred.append(move)
        pred.reverse()
        return pred
