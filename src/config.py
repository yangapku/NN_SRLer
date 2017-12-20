# -*- coding: utf-8 -*-
from __future__ import print_function, division

configure = {
    # if doing training, set 'do_train' to True and 'do_predict' to False
    'do_train': True,
    'batch_size': 50,
    'num_spoch': 5,
    'lrate': 0.001,
    'save_path': None,

    # if doing predicting, set 'do_train' to False and 'do_predict' to True
    'do_predict': False,
    'load_path': None,

    # shared settings in training and predicting
    'vocab_size': 16000, # 0 for <eos>, 1 for <bos> and 2 for <unk>
    'embedding_dim': 50,
    'postag_dim': 20,
    'distance_dim': 20,
    'fc_hidden1_dim': 200,
    'RNN_dim': 100,
    'num_of_rnn_layers': 1,
    'fc_hidden2_dim': 100,
    'max_len': 250,
}