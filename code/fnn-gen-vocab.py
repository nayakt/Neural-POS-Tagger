import sys
import os
import gzip
import json
import pickle
from collections import OrderedDict
import re
import codecs
from operator import itemgetter
import nltk
from multiprocessing import Process
from nltk.parse.stanford import StanfordParser
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import numpy as np
random_seed = 1023
import random
from collections import namedtuple


def build_vocab(data, embed_file):
    vocab = OrderedDict()
    suf2_vocab = list()
    suf3_vocab = list()
    pref2_vocab = list()
    pref3_vocab = list()
    for d in data:
        for word in d.Words:
            word = word.strip().lower()
            if len(word) > 0:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

            if len(word) >= 2:
                suf2 = word[-2] + word[-1]
                pref2 = word[0] + word[1]
                if suf2 not in suf2_vocab:
                    suf2_vocab.append(suf2)
                if pref2 not in pref2_vocab:
                    pref2_vocab.append(pref2)

            if len(word) >= 3:
                suf3 = word[-3] + word[-2] + word[-1]
                pref3 = word[0] + word[1] + word[2]
                if suf3 not in suf3_vocab:
                    suf3_vocab.append(suf3)
                if pref3 not in pref3_vocab:
                    pref3_vocab.append(pref3)

    print('word vocab length:', len(vocab))

    # print(len(suf2_vocab))
    # print(len(suf3_vocab))
    # print(len(pref2_vocab))
    # print(len(pref3_vocab))

    suf2_dict = OrderedDict()
    suf2_dict['<PAD>'] = 0
    suf2_dict['<UNK>'] = 1
    idx = 2
    for suf2 in suf2_vocab:
        suf2_dict[suf2] = idx
        idx += 1

    suf3_dict = OrderedDict()
    suf3_dict['<PAD>'] = 0
    suf3_dict['<UNK>'] = 1
    idx = 2
    for suf3 in suf3_vocab:
        suf3_dict[suf3] = idx
        idx += 1

    pref2_dict = OrderedDict()
    pref2_dict['<PAD>'] = 0
    pref2_dict['<UNK>'] = 1
    idx = 2
    for pref2 in pref2_vocab:
        pref2_dict[pref2] = idx
        idx += 1

    pref3_dict = OrderedDict()
    pref3_dict['<PAD>'] = 0
    pref3_dict['<UNK>'] = 1
    idx = 2
    for pref3 in pref3_vocab:
        pref3_dict[pref3] = idx
        idx += 1

    # print(len(suf2_dict))
    # print(len(suf3_dict))
    # print(len(pref2_dict))
    # print(len(pref3_dict))

    embed_mat = OrderedDict()
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            vec = [np.float32(val) for val in parts[1:]]
            embed_mat[word] = vec

    embed_vocab = OrderedDict()
    embed_matrix = list()
    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))
    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    embed_vocab['<S>'] = 2
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    embed_vocab['<END>'] = 3
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    word_idx = 4
    for word in vocab:
        if word in embed_mat:
            embed_matrix.append(embed_mat[word])
            embed_vocab[word] = word_idx
            word_idx += 1
        else:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1
    # print(len(embed_vocab))
    # print(len(embed_matrix))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32), suf2_dict, suf3_dict, pref2_dict, pref3_dict


def get_training_data(file_name):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    samples = []
    for line in lines:
        labels = list()
        words = list()
        line = line.strip()
        parts = line.split()
        for part in parts:
            tag = part.split('/')[-1]
            word = '/'.join(part.split('/')[0:-1])
            words.append(word)
            labels.append(tag)
        sample = Sample(Len=len(words), Words=words, Labels=labels)
        samples.append(sample)
    return samples


def get_tags(file_name):
    tags = []
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    for line in lines:
        parts = line.split()
        for part in parts:
            tag = part.split('/')[-1]
            if tag not in tags:
                tags.append(tag)
    pos_map = OrderedDict()
    reverse_pos_map = OrderedDict()
    pos_map['<PAD>'] = 0
    reverse_pos_map[0] = '<PAD>'
    pos_map['<S>'] = 1
    reverse_pos_map[1] = '<S>'
    pos_map['<END>'] = 2
    reverse_pos_map[2] = '<END>'
    idx = 3
    for pos in tags:
        pos_map[pos] = idx
        reverse_pos_map[idx] = pos
        idx += 1
    return pos_map, reverse_pos_map


if __name__ == "__main__":
    train_file = sys.argv[1]
    embedding_file = sys.argv[2]
    word_embed_dim = int(sys.argv[3])
    vocab_file = sys.argv[4]
    Sample = namedtuple("Sample", "Len Words Labels")

    pos_map, reverse_pos_map = get_tags(train_file)
    train_data = get_training_data(train_file)

    a, b, c, d, e, f = build_vocab(train_data, embedding_file)

    vocab_data = [a, b, c, d, e, f, pos_map, reverse_pos_map]
    output = open(vocab_file, 'wb')
    pickle.dump(vocab_data, output)
    output.close()



















