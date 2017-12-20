# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np


def load_dictionaries(config):
    # load wordidx
    fp = open("../dat/wordidx", "r")
    word2idx, idx2word = dict(), dict()
    words = [line.strip() for line in fp.readlines()]
    if config['vocab_size'] is not None:
        words = words[:config['vocab_size']]
    for idx, word in enumerate(words):
        word2idx[word] = idx
        idx2word[idx] = word
    fp.close()

    # load labelidx
    fp = open("../dat/postagidx", "r")
    postag2idx, idx2postag = dict(), dict()
    for idx, postag in enumerate([line.strip() for line in fp.readlines()]):
        postag2idx[postag] = idx
        idx2postag[idx] = postag
    fp.close()

    # load labelidx
    fp = open("../dat/labelidx", "r")
    label2idx, idx2label = dict(), dict()
    for idx, label in enumerate([line.strip() for line in fp.readlines()]):
        label2idx[label] = idx
        idx2label[idx] = label
    fp.close()

    return (word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label)


def train_data_loader(dicts, config):
    word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label = dicts

    # read in preprocessed training data
    fp = open(config["training_data_path"], "r")
    records = [line.strip() for line in fp.readlines()]
    train_data = []
    sent_data = []
    for record in records:
        if record == '':
            train_data.append(sent_data)
            sent_data = []
            continue
        feats = record.split("\t")
        feats[7] = int(feats[7]) # preprocess distance feature
        assert len(feats) == 9  # a valid training record should have 9 attributes
        sent_data.append(feats)
    fp.close()

    # transform data into padded numpy array
    training_X = np.zeros(shape=(len(train_data), config['max_len'], 8), dtype=np.int32)
    training_y = np.zeros(shape=(len(train_data), config['max_len']), dtype=np.int32)
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            curword, lastword, nextword, pos, lastpos, nextpos, relword, dist, label = train_data[i][j]
            training_X[i, j, 0] = word2idx[curword] if curword in word2idx else 2
            training_X[i, j, 1] = word2idx[lastword] if lastword in word2idx else 2
            training_X[i, j, 2] = word2idx[nextword] if nextword in word2idx else 2
            training_X[i, j, 3] = word2idx[relword] if relword in word2idx else 2
            training_X[i, j, 4] = postag2idx[pos]
            training_X[i, j, 5] = postag2idx[lastpos]
            training_X[i, j, 6] = postag2idx[nextpos]
            training_X[i, j, 7] = dist
            training_y[i, j] = label2idx[label]
    return train_data, training_X, training_y

def test_data_loader(dicts, config):
    word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label = dicts

    # read in preprocessed training data
    fp = open(config['testing_data_path'], "r")
    records = [line.strip() for line in fp.readlines()]
    test_data = []
    sent_data = []
    for record in records:
        if record == '':
            test_data.append(sent_data)
            sent_data = []
            continue
        feats = record.split("\t")[:-1]
        feats[7] = int(feats[7]) # preprocess distance feature
        assert len(feats) == 8  # a valid training record should have 9 attributes
        sent_data.append(feats)
    fp.close()

    # transform data into padded numpy array
    testing_X = np.zeros(shape=(len(test_data), config['max_len'], 8), dtype=np.int32)
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            curword, lastword, nextword, pos, lastpos, nextpos, relword, dist = test_data[i][j]
            testing_X[i, j, 0] = word2idx[curword] if curword in word2idx else 2
            testing_X[i, j, 1] = word2idx[lastword] if lastword in word2idx else 2
            testing_X[i, j, 2] = word2idx[nextword] if nextword in word2idx else 2
            testing_X[i, j, 3] = word2idx[relword] if relword in word2idx else 2
            testing_X[i, j, 4] = postag2idx[pos]
            testing_X[i, j, 5] = postag2idx[lastpos]
            testing_X[i, j, 6] = postag2idx[nextpos]
            testing_X[i, j, 7] = dist
    return test_data, testing_X

def write_outputs(preds, config, idx2label):
    fout = open(config['output_path'], "w")
    fp = open(config['testing_sourcefile_path'], "r")
    data = fp.readlines()
    data = [line.strip().split(" ") for line in data]
    for i in range(len(data)):
        line = ''
        for j in range(len(data[i])):
            line += "/".join([data[i][j].split("/")[0], data[i][j].split("/")[1], idx2label[preds[i][j]]]) + " "
        fout.write(line+"\n")
    fout.close()
    fp.close()