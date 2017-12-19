# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import numpy as np


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
        self.W3 = tf.get_variable("W_2", shape=(self.config['fc_hidden2_dim'], 66),
                                  initializer=tf.random_normal_initializer())
        self.b3 = tf.get_variable("b_2", shape=(66,),
                                  initializer=tf.random_normal_initializer())

        # initialize RNN cell
        self.fw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim'])
        self.bw_lstmcell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config['RNN_dim'])

        # initialize transition rule
        self.make_transition_rule()

    def make_transition_rule(self):
        # make a matrix to indicate whether a transition is legal
        trans = np.ones(shape=(66, 66), dtype="float32")
        # B flag can only be followed with I or E with same name
        trans[:17, :] = 0
        trans[:17, 17:34] = np.diag(np.ones(17))
        trans[:11, 34:45] = np.diag(np.ones(11))
        trans[12:17, 45:50] = np.diag(np.ones(5))

        # E, O, S flag can be followed with any labels except E and I
        trans[17:34, 17:50] = 0
        trans[50:66, 17:50] = 0

        # I flag can only be followed with I or E with same name
        trans[34:50, :] = 0
        trans[34:50, 34:50] = np.diag(np.ones(16))
        trans[34:45, 17:28] = np.diag(np.ones(11))
        trans[45:50, 29:34] = np.diag(np.ones(5))

        self.transition_rules = tf.constant(trans, dtype="float32")

    def compile(self, mode):
        if mode == 'train':
            self._compile_train()
        elif mode == 'test':
            self._compile_test()

    def _compile_train(self):
        # define placeholder
        self.curword = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.lastword = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.nextword = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.predicate = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.curpostag = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.lastpostag = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.nextpostag = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))
        self.distance = tf.placeholder(dtype="int32", shape=(self.config['batch_size'], self.config['max_len']))

        self.seq_length = tf.placeholder(dtype="int32", shape=(self.config['batch_size'],))

        self.label = tf.placeholder(dtype="int32",
                                    shape=(self.config['batch_size'], self.config['max_len'],
                                           66))  # one-hot encode, treat rel as 'O'
        self.mask = tf.placeholder(dtype="int32",
                                   shape=(self.config['batch_size'], self.config['max_len']))  # 0 for padding words
        self.rel_mask = tf.placeholder(dtype="int32", shape=(
            self.config['batch_size'], self.config['max_len']))  # 1 for rel, 0 for others

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
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_lstmcell, self.bw_lstmcell, hidden_1, self.seq_length)
        hidden_rnn = tf.concat(outputs, axis=2)

        # second fully connected layer
        hidden_2 = tf.matmul(tf.reshape(hidden_rnn, (-1, 2 * self.config['RNN_dim'])), self.W2) + self.b2
        hidden_2 = tf.tanh(hidden_2)

        # output layer
        logits = tf.matmul(hidden_2, self.W3) + self.b3
        logits = tf.reshape(logits, (-1, self.config['max_len'], 66))

        # calculate loss function
        scores = tf.reduce_sum(tf.cast(self.label, "float32") * logits, axis=2) * tf.cast(self.mask, "float32") * (
            1.0 - tf.cast(self.rel_mask, "float32"))
        scores = tf.reduce_sum(scores, axis=1)

        tiled_trans_rules = tf.tile(tf.expand_dims(self.transition_rules, axis=0), [self.config['batch_size'], 1, 1])
        '''
        1. use scan function to calculate "logexpsum" item
        2. skip considering predicate word when calculating probability of a label sequence
        Are the two methods effective? Anyway they are complex...
        '''
        def _step(sumexp_scores, params):
            # fetch score at current step, padding mask and rel word mask
            single_step_scores, mask, rel_mask = params
            mask = tf.tile(tf.expand_dims(tf.cast(mask, "float32"), axis=1), [1, 66])
            rel_mask = tf.tile(tf.expand_dims(tf.cast(rel_mask, "float32"), axis=1), [1, 66])

            # calculate sum exp scores at this step, considering special case where current word is rel word
            exp_single_step_scores = tf.exp(single_step_scores) * (1.0 - rel_mask) + tf.ones(
                (self.config['batch_size'], 66)) * rel_mask
            expand_sumexp_scores = tf.tile(tf.expand_dims(sumexp_scores, axis=2), [1, 1, 66])
            expand_exp_single_step_scores = tf.tile(tf.expand_dims(exp_single_step_scores, axis=1), [1, 66, 1])
            new_sumexp_scores = expand_sumexp_scores * expand_exp_single_step_scores * tiled_trans_rules
            new_sumexp_scores = tf.reduce_sum(tf.transpose(new_sumexp_scores, [0, 2, 1]), axis=2)
            new_sumexp_scores = new_sumexp_scores * (1 - rel_mask) + (tf.concat(
                [tf.zeros(self.config['batch_size'], 50), new_sumexp_scores[:, 50],
                 tf.zeros(self.config['batch_size'], 15)], axis=1)) * rel_mask
            new_sumexp_scores = sumexp_scores * (1 - mask) + new_sumexp_scores * mask
            return new_sumexp_scores

        scores = tf.transpose(scores, [1, 0, 2])  # (max_len, batch_size, num_labels)
        masks = tf.transpose(self.mask, [1, 0])  # (max_len, batch_size)
        rel_masks = tf.transpose(self.rel_mask, [1, 0])  # (max_len, batch_size)
        init_rel_mask = tf.tile(tf.expand_dims(tf.cast(rel_masks[0], "float32"), axis=1), [1, 66])
        init_sumexp_score = tf.exp(scores[0]) * (1.0 - init_rel_mask) + \
                            tf.ones((self.config['batch_size'], 66)) \
                            * init_rel_mask  # exp score at first step, consider special case where first word is rel word
        all_sumexp_scores = tf.scan(_step, elems=(scores[1:, ], masks[1:, ], rel_masks[1:, ]),
                                    initializer=init_sumexp_score)
        logsumexp_scores = tf.log(tf.reduce_sum(all_sumexp_scores[-1], axis=1))

        self.loss = tf.reduce_sum(scores - logsumexp_scores, axis=0)

        # initialize training op
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.config['lrate']).minimize(self.loss)

    def _compile_test(self):
        # TODO: build graph for prediction
        pass


    def train(self, training_data, feats, labels):
        lengths = np.array([len(sent) for sent in training_data], dtype=np.int32)
        # TODO: prepare numpy array for feed_dict

        # TODO: run training session

