# -*- coding: utf-8 -*-
from __future__ import print_function, division

configure = {
    # if doing training, set 'do_train' to True and 'do_predict' to False
    'do_train': True,
    'training_data_path': '../dat/trainIn.txt',
    'batch_size': 50,
    'num_spoch': 15,
    'lrate': 0.01,
    'save_path': '../models/' + '171220_ver02/171220_ver02',

    # if doing predicting, set 'do_train' to False and 'do_predict' to True
    'do_predict': False,
    'testing_data_path': '../dat/devIn.txt',
    'testing_sourcefile_path': '../dat/cpbdev.txt',
    'load_path': '../models/' + '171220_ver01-9',
    'output_path': '../outputs/dev_171220_ver01_9.txt',

    # shared settings in training and predicting
    'vocab_size': 16000, # 0 for <eos>, 1 for <bos> and 2 for <unk>
    'n_label': 67,
    'embedding_dim': 50,
    'postag_dim': 20,
    'distance_dim': 20,
    'fc_hidden1_dim': 200,
    'RNN_dim': 100,
    'num_of_rnn_layers': 1,
    'fc_hidden2_dim': 100,
    'max_len': 250,
}