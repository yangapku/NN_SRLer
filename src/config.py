# -*- coding: utf-8 -*-
from __future__ import print_function, division

configure = {
    # if doing training, set 'do_train' to True and 'do_predict' to False
    'do_train': True,
    'training_data_path': '../dat/trainIn.txt',
    'validate_data_path': '../dat/devIn.txt',
    'batch_size': 50,
    'num_spoch': 10,
    'optimizer': 'sgd',
    'lrate': 0.001,
    'save_path': '../models/' + '171225_ver04/171225_ver05',
    'log_dir': '../log/' + '171225_ver05/',

    # if doing predicting, set 'do_train' to False and 'do_predict' to True
    'do_predict': False,
    'testing_data_path': '../dat/devIn.txt',
    'testing_sourcefile_path': '../dat/cpbdev.txt',
    'load_path': '../models/' + '171222_ver04/171222_ver04-9',
    'output_path': '../outputs/test_171222_ver04_9.txt',

    # shared settings in training and predicting
    'vocab_size': 17000, # 0 for <eos>, 1 for <bos> and 2 for <unk>
    'use_crf': True,
    'use_dist_embedding': True,
    'n_label': 67,
    'embedding_dim': 50,
    'postag_dim': 20,
    'distance_dim': 20,
    'RNN_dim': 100,
    'num_of_rnn_layers': 1,
    'max_len': 150,
}
