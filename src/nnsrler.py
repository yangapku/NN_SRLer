# -*- coding: utf-8 -*-
from __future__ import division, print_function
from data_utils import *
from config import configure
from model import *
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    # read in dictionaries
    dicts = load_dictionaries(configure) # word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label

    # initialize LSTM SRL model
    model = LSTMSRLer(configure)

    # do training
    if configure['do_train']:
        # build computation graph for training
        model.compile('train')

        # load data
        raw_train_data, train_feats, train_label = train_data_loader(dicts, configure)

        # run training process
        model.train(raw_train_data, train_feats, train_label)