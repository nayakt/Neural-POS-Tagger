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
    char_vocab = OrderedDict()
    char_vocab['<PAD>'] = 0
    char_vocab['<UNK>'] = 1
    char_idx = 2
    vocab = OrderedDict()
    for d in data:
        for word in d.Words:
            word = word.strip()
            if len(word) > 0:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
                for c in word:
                    if c not in char_vocab:
                        char_vocab[c] = char_idx
                        char_idx += 1
    print('word vocab size:', len(vocab))
    print('char vocab size:', len(char_vocab))

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
    word_idx = 2
    for word in vocab:
        if word in embed_mat:
            embed_matrix.append(embed_mat[word])
            embed_vocab[word] = word_idx
            word_idx += 1
        elif word.lower() in embed_mat:
            embed_matrix.append(embed_mat[word.lower()])
            embed_vocab[word] = word_idx
            word_idx += 1
        else:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1
    return char_vocab, embed_vocab, np.array(embed_matrix, dtype=np.float32)


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
    pos_map['<s>'] = 1
    reverse_pos_map[1] = '<s>'
    pos_map['<end>'] = 2
    reverse_pos_map[2] = '<end>'
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

    char_vocab, word_vocab, word_embed_matrix = build_vocab(train_data, embedding_file)

    vocab_data = [char_vocab, word_vocab, word_embed_matrix, pos_map, reverse_pos_map]
    output = open(vocab_file, 'wb')
    pickle.dump(vocab_data, output)
    output.close()



















