# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import logging
import pickle


class LSTMSRLer:
    def __init__(self, configure, idx2label, label2idx):
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
        if self.config['use_dist_embedding']:
            total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim'] + self.config['distance_dim']
        else:
            total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim']

        self.W = tf.get_variable("W_3", shape=(2 * self.config['RNN_dim'], self.config['n_label']),
                                  initializer=tf.random_normal_initializer())
        self.b = tf.get_variable("b_3", shape=(self.config['n_label'],),
                                  initializer=tf.random_normal_initializer())

        # initialize RNN cell
        self.fw_lstmcells = []
        self.bw_lstmcells = []

        for _i in range(self.config['num_of_rnn_layers']):
            self.fw_lstmcells.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim']))
            self.bw_lstmcells.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim']))

        # initialize transition rule
        self.make_transition_rule()

        self.idx2label = idx2label
        self.label2idx = label2idx

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
        if self.config['use_dist_embedding']:
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
        if self.config['use_dist_embedding']:
            dist_emb = tf.nn.embedding_lookup(self.distance_embedding, self.distance)
            embedding = tf.concat(
                [curword_emb, lastword_emb, nextword_emb, predicate_emb, curpos_emb, lastpos_emb, nextpos_emb, dist_emb],
                axis=2)
        else:
            embedding = tf.concat(
                [curword_emb, lastword_emb, nextword_emb, predicate_emb, curpos_emb, lastpos_emb, nextpos_emb],
                axis=2)

        # first fully connected layer
        if self.config['use_dist_embedding']:
            total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim'] + self.config['distance_dim']
        else:
            total_embed_dim = 4 * self.config['embedding_dim'] + 3 * self.config['postag_dim']

        # recurrent layer
        if self.config['num_of_rnn_layers'] > 1:
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.fw_lstmcells, self.bw_lstmcells, embedding,
                                                                           sequence_length=self.seq_length,
                                                                           dtype="float32")
            hidden_rnn = outputs[:, :, -2 * self.config['RNN_dim']:]
        else:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_lstmcells[0], self.bw_lstmcells[0], embedding,
                                                                        self.seq_length,
                                                                        dtype="float32")
            hidden_rnn = tf.concat(outputs, axis=2)

        # output layer
        logits = tf.matmul(tf.reshape(hidden_rnn, (-1, 2 * self.config['RNN_dim'])), self.W) + self.b  # (batch_size * max_len, n_label)
        if self.config['use_crf']:
            if training:
                inputs = tf.reshape(logits, shape=(-1, self.config['max_len'], self.config['n_label']))
                label_index = tf.argmax(self.label, axis=2, output_type="int32")
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs, label_index, self.seq_length)
                self.loss = -tf.reduce_sum(log_likelihood)
                tf.summary.scalar('loss', self.loss)
                self.train_op = self.getOptimizer(self.config['optimizer'], self.config['lrate']).minimize(self.loss)
                self.merge = tf.summary.merge_all() # record loss
            else:
                self.outputs = tf.reshape(logits, (-1, self.config['max_len'], self.config['n_label']))
        else:
            if training:
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.cast(tf.reshape(self.label, shape=(-1, self.config['n_label'])), "float32"), logits=logits)
                loss = tf.reshape(loss, (-1, self.config['max_len']))
                loss = loss * tf.cast(self.mask, "float32")
                self.loss = tf.reduce_sum(loss)
                tf.summary.scalar('loss', self.loss)
                # initialize training op
                self.train_op = self.getOptimizer(self.config['optimizer'], self.config['lrate']).minimize(self.loss)
                self.merge = tf.summary.merge_all() # record loss
            else:
                self.outputs = tf.reshape(logits, (-1, self.config['max_len'], self.config['n_label']))

    @staticmethod
    def getOptimizer(name, lrate):
        assert name in ['sgd', 'adam', 'momentum', 'adagrad']
        if name == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=lrate)
        elif name == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lrate)
        elif name == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=lrate, momentum=0.8)
        elif name == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=lrate)

    def train(self, training_data, feats, labels, val_data, val_feats, val_labels, save_per_epoch):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=self.config['num_spoch'])
        lengths = np.array([len(sent) for sent in training_data], dtype=np.int32)
        val_lengths = np.array([len(sent) for sent in val_data], dtype=np.int32)
        with tf.Session() as sess:
            train_filewriter = tf.summary.FileWriter(logdir=self.config['log_dir'], graph=sess.graph)
            valid_filewriter = tf.summary.FileWriter(logdir=self.config['log_dir'], graph=sess.graph)
            sess.run(init)
            rng = np.random.RandomState(seed=1701214021)
            for epoch in range(self.config['num_spoch']):
                logging.info("Epoch %d started:" % epoch)
                sum_loss = 0.
                ids = np.arange(len(training_data))
                rng.shuffle(ids)
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
                        self.seq_length: length,
                        self.label: onehot_label,
                        self.mask: masks,
                        # self.rel_mask: rel_masks,
                    }
                    if self.config['use_dist_embedding']:
                        feed_dict[self.distance] = features[:, :, 7]
                    if self.config['use_crf']:
                        summary, trans_params, iter_loss, _ = sess.run([self.merge, self.transition_params, self.loss, self.train_op], feed_dict=feed_dict)
                    else:
                        summary, iter_loss, _ = sess.run([self.merge, self.loss, self.train_op], feed_dict=feed_dict)
                    sum_loss += iter_loss
                    if iter % 10 == 0:
                        logging.info("Iter %d, training loss: %f" % (
                            iter, sum_loss * 1. / (1 + iter) / self.config['batch_size']))
                        train_filewriter.add_summary(summary=summary, global_step=epoch * batch_num + iter)
                logging.info(
                    "Epoch %d, training loss: %f" % (epoch, sum_loss * 1. / batch_num / self.config['batch_size']))

                # calc validation loss
                sum_val_loss = 0.
                ids = np.arange(len(val_data))
                val_batch_num = len(val_data) // self.config['batch_size']
                for iter in range(val_batch_num):
                    data_id = ids[iter * self.config['batch_size']:(iter + 1) * self.config['batch_size']]
                    features = val_feats[data_id]
                    length = val_lengths[data_id]
                    label = val_labels[data_id]
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
                        self.seq_length: length,
                        self.label: onehot_label,
                        self.mask: masks,
                        # self.rel_mask: rel_masks,
                    }
                    if self.config['use_dist_embedding']:
                        feed_dict[self.distance] = features[:, :, 7]
                    summary, iter_loss = sess.run([self.merge, self.loss], feed_dict=feed_dict)
                    sum_val_loss += iter_loss
                logging.info(
                    "Epoch %d, validation loss: %f" % (epoch, sum_val_loss * 1. / val_batch_num / self.config['batch_size']))

                if save_per_epoch:
                    saver.save(sess, save_path=self.config['save_path'], global_step=epoch)
                    if self.config['use_crf']:
                        fout = open(self.config['save_path'] + ("-%d.trans.pkl" % epoch), "wb")
                        pickle.dump(trans_params, fout)
                        fout.close()
                    logging.info("Training checkpoint has been saved.")
            train_filewriter.close()
            valid_filewriter.close()

    def test(self, testing_data, feats):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        lengths = np.array([len(sent) for sent in testing_data], dtype=np.int32)
        preds = []
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, save_path=self.config['load_path'])
            if self.config['use_crf']:
                fin = open(self.config['load_path'] + ".trans.pkl", "rb")
                trans_param = pickle.load(fin)
                fin.close()
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
                    self.seq_length: length,
                    self.mask: masks,
                }
                if self.config['use_dist_embedding']:
                    feed_dict[self.distance] = features[:, :, 7]
                outputs = sess.run(self.outputs, feed_dict=feed_dict)[0]
                if self.config['use_crf']:
                    sequence, _ = tf.contrib.crf.viterbi_decode(outputs[:lengths[i]], trans_param)
                    sequence = self.modify(sequence)
                else:
                    sequence = self.find_best(outputs, lengths[i], features[0, :, 7])
                    # sequence = self.modify(np.argmax(outputs[:lengths[i], :], axis=1))
                preds.append(sequence)
        return preds

    def modify(self, sequence):
        lastname = ""
        for i in range(len(sequence)):
            tagid = sequence[i]
            tag = self.idx2label[tagid]
            if tag[0] == "B":
                lastname = tag[2:]
                continue
            if tag[0] == "I" or tag[0] == "E":
                if tag[2:] != lastname:
                    lastname = tag[2:]
                    sequence[i] = self.label2idx["B-"+lastname]
        return sequence

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
                    record[0:17, i] = scores[0:17, i] + record[66, i - 1]
                    path[0:17, i] = 66
                    record[50:66, i] = scores[50:66, i] + record[66, i - 1]
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
