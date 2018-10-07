import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

random_seed = 1023
import numpy as np
np.random.seed(random_seed)
import random
random.seed(random_seed)

from collections import namedtuple
import math
import pickle
import datetime
from tqdm import tqdm

import torch
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
        else:
            print(msg[i], ' ', end='')


def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        char_vocab, embed_vocab, embed_matrix, pos_map, rev_pos_map = pickle.load(f)
    return char_vocab, embed_vocab, embed_matrix, pos_map, rev_pos_map


def get_training_data(file_name):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    samples = []
    for line in lines:
        if len(line.strip().split()) > max_sent_len or len(line.strip().split()) < 2:
            continue
        labels = list()
        words = list()
        for part in line.strip().split():
            part = part.strip()
            tag = part.split('/')[-1]
            word = '/'.join(part.split('/')[0:-1])
            words.append(word)
            labels.append(tag)
        samples.append(Sample(Len=len(words), Words=words,
                              PrevWords=['<S>'] + words[:-1], PrevPrevWords=['<S>', '<S>'] + words[:-2],
                              NextWords=words[1:] + ['<END>'], NextNextWords=words[2:] + ['<END>', '<END>'],
                              PrevPOS=['<S>'] + labels[0:-1],
                              PrevPrevPOS=['<S>', '<S>'] + labels[0:-2], Labels=labels))
    return samples


def get_test_data(file_name):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    samples = []
    for line in lines:
        labels = list()
        words = line.strip().split()
        samples.append(Sample(Len=len(words), Words=words,
                              PrevWords=['<S>'] + words[:-1], PrevPrevWords=['<S>', '<S>'] + words[:-2],
                              NextWords=words[1:] + ['<END>'], NextNextWords=words[2:] + ['<END>', '<END>'],
                              PrevPOS=labels,
                              PrevPrevPOS=labels, Labels=labels))
    return samples


def get_max_len(sample_batch):
    max_len = len(sample_batch[0].Words)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].Words) > max_len:
            max_len = len(sample_batch[idx].Words)
    return max_len


def get_pos_index_seq(tags, max_len):
    tag_seq = list()
    for tag in tags:
        tag_seq.append(pos_map[tag])
    pad_len = max_len - len(tags)
    for i in range(0, pad_len):
        tag_seq.append(pos_map['<PAD>'])
    return tag_seq


def get_words_index_seq(words, max_len):
    words_seq = list()
    for word in words:
        word = word.lower()
        if word in word_vocab:
            words_seq.append(word_vocab[word])
        else:
            words_seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        words_seq.append(word_vocab['<PAD>'])
    return words_seq


def get_suf2_index_seq(words, max_len):
    suf2_seq = list()
    for word in words:
        if word in ['<S>', '<END>']:
            suf2_seq.append(suf2_dict['<PAD>'])
        else:
            word = word.lower()
            if len(word) >= 2:
                suf2 = word[-2] + word[-1]
                if suf2 in suf2_dict:
                    suf2_seq.append(suf2_dict[suf2])
                else:
                    suf2_seq.append(suf2_dict['<PAD>'])
            else:
                suf2_seq.append(suf2_dict['<PAD>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        suf2_seq.append(suf2_dict['<PAD>'])
    return suf2_seq


def get_suf3_index_seq(words, max_len):
    suf3_seq = list()
    for word in words:
        if word in ['<S>', '<END>']:
            suf3_seq.append(suf3_dict['<PAD>'])
        else:
            word = word.lower()
            if len(word) >= 3:
                suf3 = word[-3] + word[-2] + word[-1]
                if suf3 in suf3_dict:
                    suf3_seq.append(suf3_dict[suf3])
                else:
                    suf3_seq.append(suf3_dict['<PAD>'])
            else:
                suf3_seq.append(suf3_dict['<PAD>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        suf3_seq.append(suf3_dict['<PAD>'])
    return suf3_seq


def get_pref2_index_seq(words, max_len):
    pref2_seq = list()
    for word in words:
        if word in ['<S>', '<END>']:
            pref2_seq.append(pref2_dict['<PAD>'])
        else:
            word = word.lower()
            if len(word) >= 2:
                pref2 = word[0] + word[1]
                if pref2 in pref2_dict:
                    pref2_seq.append(pref2_dict[pref2])
                else:
                    pref2_seq.append(pref2_dict['<PAD>'])
            else:
                pref2_seq.append(pref2_dict['<PAD>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        pref2_seq.append(pref2_dict['<PAD>'])
    return pref2_seq


def get_pref3_index_seq(words, max_len):
    pref3_seq = list()
    for word in words:
        if word in ['<S>', '<END>']:
            pref3_seq.append(pref3_dict['<PAD>'])
        else:
            word = word.lower()
            if len(word) >= 3:
                pref3 = word[0] + word[1] + word[2]
                if pref3 in pref3_dict:
                    pref3_seq.append(pref3_dict[pref3])
                else:
                    pref3_seq.append(pref3_dict['<PAD>'])
            else:
                pref3_seq.append(pref3_dict['<PAD>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        pref3_seq.append(pref3_dict['<PAD>'])
    return pref3_seq


def get_feat(word):
    feat = []
    if word[0] == word[0].upper():
        feat.append(1)
    else:
        feat.append(0)
    if re.match('[0-9]+', word) is not None:
        feat.append(1)
    else:
        feat.append(0)
    if '-' in word:
        feat.append(1)
    else:
        feat.append(0)
    return feat


def get_feature_vec(words, max_len):
    feat_seq = list()
    for word in words:
        if word in ['<S>', '<END>']:
            feat_seq.append([0, 0, 0])
        else:
            feat_seq.append(get_feat(word))
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        feat_seq.append([0, 0, 0])
    return feat_seq


def get_accuracy(samples, preds):
    total = 0
    correct = 0
    for i in range(0, len(samples)):
        s = samples[i]
        total += len(s.Words)
        for j in range(0, len(s.Words)):
            pred_cls_idx = np.argmax(preds[i][j])
            gold_tag = s.Labels[j]
            if pred_cls_idx == pos_map[gold_tag]:
                correct += 1
    return correct/total


def write_pred_file(preds, data, file_name):
    writer = open(file_name, 'w')
    for i in range(0, len(data)):
        s = data[i]
        word_tag_pairs = []
        for j in range(0, len(s.Words)):
            pred_cls_idx = np.argmax(preds[i][j])
            pred_tag = reverse_pos_map[pred_cls_idx]
            word_tag_pairs.append(s.Words[j] + '/' + pred_tag)
        writer.write(' '.join(word_tag_pairs) + '\n')
    writer.close()


def shuffle_data(data):
    # return random.shuffle(data)
    custom_print('shuffling data......')
    data.sort(key=lambda x: x.Len)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_training_input(samples, batch_idx, cur_batch_size, training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_start = batch_idx * cur_batch_size
    batch_end = min(len(samples), batch_start + cur_batch_size)
    cur_batch = samples[batch_start:batch_end]
    batch_max_len = get_max_len(cur_batch)

    words_seq_list = list()
    words_suf2_seq_list = list()
    words_suf3_seq_list = list()
    words_pref2_seq_list = list()
    words_pref3_seq_list = list()
    words_feat_seq_list = list()

    prev_words_seq_list = list()
    prev_words_suf2_seq_list = list()
    prev_words_suf3_seq_list = list()
    prev_words_pref2_seq_list = list()
    prev_words_pref3_seq_list = list()
    prev_words_feat_seq_list = list()

    prev_prev_words_seq_list = list()
    prev_prev_words_suf2_seq_list = list()
    prev_prev_words_suf3_seq_list = list()
    prev_prev_words_pref2_seq_list = list()
    prev_prev_words_pref3_seq_list = list()
    prev_prev_words_feat_seq_list = list()

    next_words_seq_list = list()
    next_words_suf2_seq_list = list()
    next_words_suf3_seq_list = list()
    next_words_pref2_seq_list = list()
    next_words_pref3_seq_list = list()
    next_words_feat_seq_list = list()

    next_next_words_seq_list = list()
    next_next_words_suf2_seq_list = list()
    next_next_words_suf3_seq_list = list()
    next_next_words_pref2_seq_list = list()
    next_next_words_pref3_seq_list = list()
    next_next_words_feat_seq_list = list()

    prev_pos_seq_list = list()
    prev_prev_pos_seq_list = list()

    labels_list = list()
    for sample in cur_batch:
        words_seq_list.append(get_words_index_seq(sample.Words, batch_max_len))
        prev_words_seq_list.append(get_words_index_seq(sample.PrevWords, batch_max_len))
        prev_prev_words_seq_list.append(get_words_index_seq(sample.PrevPrevWords, batch_max_len))
        next_words_seq_list.append(get_words_index_seq(sample.NextWords, batch_max_len))
        next_next_words_seq_list.append(get_words_index_seq(sample.NextNextWords, batch_max_len))

        words_suf2_seq_list.append(get_suf2_index_seq(sample.Words, batch_max_len))
        prev_words_suf2_seq_list.append(get_suf2_index_seq(sample.PrevWords, batch_max_len))
        prev_prev_words_suf2_seq_list.append(get_suf2_index_seq(sample.PrevPrevWords, batch_max_len))
        next_words_suf2_seq_list.append(get_suf2_index_seq(sample.NextWords, batch_max_len))
        next_next_words_suf2_seq_list.append(get_suf2_index_seq(sample.NextNextWords, batch_max_len))

        words_suf3_seq_list.append(get_suf3_index_seq(sample.Words, batch_max_len))
        prev_words_suf3_seq_list.append(get_suf3_index_seq(sample.PrevWords, batch_max_len))
        prev_prev_words_suf3_seq_list.append(get_suf3_index_seq(sample.PrevPrevWords, batch_max_len))
        next_words_suf3_seq_list.append(get_suf3_index_seq(sample.NextWords, batch_max_len))
        next_next_words_suf3_seq_list.append(get_suf3_index_seq(sample.NextNextWords, batch_max_len))

        words_pref2_seq_list.append(get_pref2_index_seq(sample.Words, batch_max_len))
        prev_words_pref2_seq_list.append(get_pref2_index_seq(sample.PrevWords, batch_max_len))
        prev_prev_words_pref2_seq_list.append(get_pref2_index_seq(sample.PrevPrevWords, batch_max_len))
        next_words_pref2_seq_list.append(get_pref2_index_seq(sample.NextWords, batch_max_len))
        next_next_words_pref2_seq_list.append(get_pref2_index_seq(sample.NextNextWords, batch_max_len))

        words_pref3_seq_list.append(get_pref3_index_seq(sample.Words, batch_max_len))
        prev_words_pref3_seq_list.append(get_pref3_index_seq(sample.PrevWords, batch_max_len))
        prev_prev_words_pref3_seq_list.append(get_pref3_index_seq(sample.PrevPrevWords, batch_max_len))
        next_words_pref3_seq_list.append(get_pref3_index_seq(sample.NextWords, batch_max_len))
        next_next_words_pref3_seq_list.append(get_pref3_index_seq(sample.NextNextWords, batch_max_len))

        words_feat_seq_list.append(get_feature_vec(sample.Words, batch_max_len))
        prev_words_feat_seq_list.append(get_feature_vec(sample.PrevWords, batch_max_len))
        prev_prev_words_feat_seq_list.append(get_feature_vec(sample.PrevPrevWords, batch_max_len))
        next_words_feat_seq_list.append(get_feature_vec(sample.NextWords, batch_max_len))
        next_next_words_feat_seq_list.append(get_feature_vec(sample.NextNextWords, batch_max_len))

        if training:
            prev_pos_seq_list.append(get_pos_index_seq(sample.PrevPOS, batch_max_len))
            prev_prev_pos_seq_list.append(get_pos_index_seq(sample.PrevPrevPOS, batch_max_len))
            labels_list += get_pos_index_seq(sample.Labels, batch_max_len)
        else:
            prev_pos_seq_list.append(pos_map['<S>'])
            prev_prev_pos_seq_list.append(pos_map['<S>'])
    return batch_end - batch_start, batch_max_len, {'words': np.array(words_seq_list),
                                                    'words_suf2': np.array(words_suf2_seq_list),
                                                    'words_suf3': np.array(words_suf3_seq_list),
                                                    'words_pref2': np.array(words_pref2_seq_list),
                                                    'words_pref3': np.array(words_pref3_seq_list),
                                                    'words_feat': np.array(words_feat_seq_list),
                                                    'prev_words': np.array(prev_words_seq_list),
                                                    'prev_words_suf2': np.array(prev_words_suf2_seq_list),
                                                    'prev_words_suf3': np.array(prev_words_suf3_seq_list),
                                                    'prev_words_pref2': np.array(prev_words_pref2_seq_list),
                                                    'prev_words_pref3': np.array(prev_words_pref3_seq_list),
                                                    'prev_words_feat': np.array(prev_words_feat_seq_list),
                                                    'prev_prev_words': np.array(prev_prev_words_seq_list),
                                                    'prev_prev_words_suf2': np.array(prev_prev_words_suf2_seq_list),
                                                    'prev_prev_words_suf3': np.array(prev_prev_words_suf3_seq_list),
                                                    'prev_prev_words_pref2': np.array(prev_prev_words_pref2_seq_list),
                                                    'prev_prev_words_pref3': np.array(prev_prev_words_pref3_seq_list),
                                                    'prev_prev_words_feat': np.array(prev_prev_words_feat_seq_list),
                                                    'next_words': np.array(next_words_seq_list),
                                                    'next_words_suf2': np.array(next_words_suf2_seq_list),
                                                    'next_words_suf3': np.array(next_words_suf3_seq_list),
                                                    'next_words_pref2': np.array(next_words_pref2_seq_list),
                                                    'next_words_pref3': np.array(next_words_pref3_seq_list),
                                                    'next_words_feat': np.array(next_words_feat_seq_list),
                                                    'next_next_words': np.array(next_words_seq_list),
                                                    'next_next_words_suf2': np.array(next_words_suf2_seq_list),
                                                    'next_next_words_suf3': np.array(next_words_suf3_seq_list),
                                                    'next_next_words_pref2': np.array(next_words_pref2_seq_list),
                                                    'next_next_words_pref3': np.array(next_words_pref3_seq_list),
                                                    'next_next_words_feat': np.array(next_words_feat_seq_list),
                                                    'prev_pos': np.array(prev_pos_seq_list),
                                                    'prev_prev_pos': np.array(prev_prev_pos_seq_list)}, \
           {'labels': np.array(labels_list)}


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.class_count = len(pos_map)
        self.input_dim = 5 * (3 + word_embed_dim + 4 * other_embed_dim) + 2 * other_embed_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.pos_embeddings = nn.Embedding(len(pos_map), other_embed_dim, padding_idx=0)
        self.suf2_embeddings = nn.Embedding(len(suf2_dict), other_embed_dim, padding_idx=0)
        self.suf3_embeddings = nn.Embedding(len(suf3_dict), other_embed_dim, padding_idx=0)
        self.pref2_embeddings = nn.Embedding(len(pref2_dict), other_embed_dim, padding_idx=0)
        self.pref3_embeddings = nn.Embedding(len(pref3_dict), other_embed_dim, padding_idx=0)

        self.linear = nn.Linear(self.input_dim, self.class_count)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, words, words_suf2, words_suf3, words_pref2, words_pref3, words_feat,
                            prev_words, prev_words_suf2, prev_words_suf3, prev_words_pref2, prev_words_pref3,
                            prev_words_feat,
                            prev_prev_words, prev_prev_words_suf2, prev_prev_words_suf3, prev_prev_words_pref2,
                            prev_prev_words_pref3, prev_prev_words_feat,
                            next_words, next_words_suf2, next_words_suf3, next_words_pref2, next_words_pref3,
                            next_words_feat,
                            next_next_words, next_next_words_suf2, next_next_words_suf3, next_next_words_pref2,
                            next_next_words_pref3, next_next_words_feat,
                            prev_pos=None, prev_prev_pos=None, training=False):
        words_embed = self.word_embeddings(words)
        words_suf2_embed = self.suf2_embeddings(words_suf2)
        words_suf3_embed = self.suf3_embeddings(words_suf3)
        words_pref2_embed = self.pref2_embeddings(words_pref2)
        words_pref3_embed = self.pref3_embeddings(words_pref3)
        words = torch.cat((words_embed, words_suf2_embed, words_suf3_embed, words_pref2_embed, words_pref3_embed,
                           words_feat), -1)

        prev_words_embed = self.word_embeddings(prev_words)
        prev_words_suf2_embed = self.suf2_embeddings(prev_words_suf2)
        prev_words_suf3_embed = self.suf3_embeddings(prev_words_suf3)
        prev_words_pref2_embed = self.pref2_embeddings(prev_words_pref2)
        prev_words_pref3_embed = self.pref3_embeddings(prev_words_pref3)
        prev_words = torch.cat((prev_words_embed, prev_words_suf2_embed, prev_words_suf3_embed, prev_words_pref2_embed,
                                prev_words_pref3_embed, prev_words_feat), -1)

        prev_prev_words_embed = self.word_embeddings(prev_prev_words)
        prev_prev_words_suf2_embed = self.suf2_embeddings(prev_prev_words_suf2)
        prev_prev_words_suf3_embed = self.suf3_embeddings(prev_prev_words_suf3)
        prev_prev_words_pref2_embed = self.pref2_embeddings(prev_prev_words_pref2)
        prev_prev_words_pref3_embed = self.pref3_embeddings(prev_prev_words_pref3)
        prev_prev_words = torch.cat((prev_prev_words_embed, prev_prev_words_suf2_embed, prev_prev_words_suf3_embed,
                                     prev_prev_words_pref2_embed, prev_prev_words_pref3_embed, prev_prev_words_feat),
                                    -1)

        next_words_embed = self.word_embeddings(next_words)
        next_words_suf2_embed = self.suf2_embeddings(next_words_suf2)
        next_words_suf3_embed = self.suf3_embeddings(next_words_suf3)
        next_words_pref2_embed = self.pref2_embeddings(next_words_pref2)
        next_words_pref3_embed = self.pref3_embeddings(next_words_pref3)
        next_words = torch.cat((next_words_embed, next_words_suf2_embed, next_words_suf3_embed, next_words_pref2_embed,
                                next_words_pref3_embed, next_words_feat), -1)

        next_next_words_embed = self.word_embeddings(next_next_words)
        next_next_words_suf2_embed = self.suf2_embeddings(next_next_words_suf2)
        next_next_words_suf3_embed = self.suf3_embeddings(next_next_words_suf3)
        next_next_words_pref2_embed = self.pref2_embeddings(next_next_words_pref2)
        next_next_words_pref3_embed = self.pref3_embeddings(next_next_words_pref3)
        next_next_words = torch.cat((next_next_words_embed, next_next_words_suf2_embed, next_next_words_suf3_embed,
                                     next_next_words_pref2_embed, next_next_words_pref3_embed, next_next_words_feat),
                                    -1)
        if training:
            prev_pos_embed = self.pos_embeddings(prev_pos)
            prev_prev_pos_embed = self.pos_embeddings(prev_prev_pos)
            final_feature = torch.cat((words, prev_words, prev_prev_words, next_words, next_next_words, prev_pos_embed,
                                       prev_prev_pos_embed), -1)
            seq_labels = self.linear(final_feature)
            seq_labels = seq_labels.view(-1, self.class_count)
            probs = self.logsoftmax(seq_labels)
        else:
            prev_pos_embed = torch.unsqueeze(self.pos_embeddings(prev_pos), 1)
            prev_prev_pos_embed = torch.unsqueeze(self.pos_embeddings(prev_prev_pos), 1)
            x = torch.cat((words, prev_words, prev_prev_words, next_words, next_next_words), -1)
            t = x.size()[1]
            cur_x = torch.unsqueeze(x[:, 0, :], 1)
            seq_labels = self.linear(torch.cat((cur_x, prev_pos_embed, prev_prev_pos_embed),
                                               -1)).view(-1, self.class_count)
            seq_labels = F.softmax(seq_labels)
            cur_tags = torch.max(seq_labels, -1)[1]
            probs = seq_labels.view(-1, 1, self.class_count)

            if t > 1:
                prev_prev_pos_embed = prev_pos_embed
                prev_pos_embed = torch.unsqueeze(self.pos_embeddings(cur_tags), 1)
                cur_x = torch.unsqueeze(x[:, 1, :], 1)
                seq_labels = self.linear(torch.cat((cur_x, prev_pos_embed, prev_prev_pos_embed),
                                                   -1)).view(-1, self.class_count)
                seq_labels = F.softmax(seq_labels)
                cur_tags = torch.max(seq_labels, -1)[1]
                cur_probs = seq_labels.view(-1, 1, self.class_count)
                probs = torch.cat((probs, cur_probs), 1)

            if t > 2:
                prev_prev_pos_embed = prev_pos_embed
                prev_pos_embed = torch.unsqueeze(self.pos_embeddings(cur_tags), 1)
                for j in range(2, t):
                    cur_x = torch.unsqueeze(x[:, j, :], 1)
                    seq_labels = self.linear(torch.cat((cur_x, prev_pos_embed, prev_prev_pos_embed),
                                                       -1)).view(-1, self.class_count)
                    seq_labels = F.softmax(seq_labels)
                    cur_tags = torch.max(seq_labels, -1)[1]
                    cur_probs = seq_labels.view(-1, 1, self.class_count)
                    probs = torch.cat((probs, cur_probs), 1)
                    prev_prev_pos_embed = prev_pos_embed
                    prev_pos_embed = torch.unsqueeze(self.pos_embeddings(cur_tags), 1)
        return probs


def get_model():
    return BaselineModel()


def predict(samples, model):
    pred_batch_size = 32
    batch_count = int(math.ceil(len(samples) / pred_batch_size))
    preds = list()
    torch.cuda.manual_seed(random_seed)
    model.eval()
    for batch_idx in range(0, batch_count):
        cur_batch_len, cur_seq_len, cur_samples_input, cur_samples_target \
            = get_training_input(samples, batch_idx, pred_batch_size)

        words = torch.from_numpy(cur_samples_input['words'].astype('long'))
        words_suf2 = torch.from_numpy(cur_samples_input['words_suf2'].astype('long'))
        words_suf3 = torch.from_numpy(cur_samples_input['words_suf3'].astype('long'))
        words_pref2 = torch.from_numpy(cur_samples_input['words_pref2'].astype('long'))
        words_pref3 = torch.from_numpy(cur_samples_input['words_pref3'].astype('long'))
        words_feat = torch.from_numpy(cur_samples_input['words_feat'].astype('float32'))

        prev_words = torch.from_numpy(cur_samples_input['prev_words'].astype('long'))
        prev_words_suf2 = torch.from_numpy(cur_samples_input['prev_words_suf2'].astype('long'))
        prev_words_suf3 = torch.from_numpy(cur_samples_input['prev_words_suf3'].astype('long'))
        prev_words_pref2 = torch.from_numpy(cur_samples_input['prev_words_pref2'].astype('long'))
        prev_words_pref3 = torch.from_numpy(cur_samples_input['prev_words_pref3'].astype('long'))
        prev_words_feat = torch.from_numpy(cur_samples_input['prev_words_feat'].astype('float32'))

        prev_prev_words = torch.from_numpy(cur_samples_input['prev_prev_words'].astype('long'))
        prev_prev_words_suf2 = torch.from_numpy(cur_samples_input['prev_prev_words_suf2'].astype('long'))
        prev_prev_words_suf3 = torch.from_numpy(cur_samples_input['prev_prev_words_suf3'].astype('long'))
        prev_prev_words_pref2 = torch.from_numpy(cur_samples_input['prev_prev_words_pref2'].astype('long'))
        prev_prev_words_pref3 = torch.from_numpy(cur_samples_input['prev_prev_words_pref3'].astype('long'))
        prev_prev_words_feat = torch.from_numpy(cur_samples_input['prev_prev_words_feat'].astype('float32'))

        next_words = torch.from_numpy(cur_samples_input['next_words'].astype('long'))
        next_words_suf2 = torch.from_numpy(cur_samples_input['next_words_suf2'].astype('long'))
        next_words_suf3 = torch.from_numpy(cur_samples_input['next_words_suf3'].astype('long'))
        next_words_pref2 = torch.from_numpy(cur_samples_input['next_words_pref2'].astype('long'))
        next_words_pref3 = torch.from_numpy(cur_samples_input['next_words_pref3'].astype('long'))
        next_words_feat = torch.from_numpy(cur_samples_input['next_words_feat'].astype('float32'))

        next_next_words = torch.from_numpy(cur_samples_input['next_next_words'].astype('long'))
        next_next_words_suf2 = torch.from_numpy(cur_samples_input['next_next_words_suf2'].astype('long'))
        next_next_words_suf3 = torch.from_numpy(cur_samples_input['next_next_words_suf3'].astype('long'))
        next_next_words_pref2 = torch.from_numpy(cur_samples_input['next_next_words_pref2'].astype('long'))
        next_next_words_pref3 = torch.from_numpy(cur_samples_input['next_next_words_pref3'].astype('long'))
        next_next_words_feat = torch.from_numpy(cur_samples_input['next_next_words_feat'].astype('float32'))

        prev_pos = torch.from_numpy(cur_samples_input['prev_pos'].astype('long'))
        prev_prev_pos = torch.from_numpy(cur_samples_input['prev_prev_pos'].astype('long'))

        if torch.cuda.is_available():
            words = words.cuda()
            words_suf2 = words_suf2.cuda()
            words_suf3 = words_suf3.cuda()
            words_pref2 = words_pref2.cuda()
            words_pref3 = words_pref3.cuda()
            words_feat = words_feat.cuda()

            prev_words = prev_words.cuda()
            prev_words_suf2 = prev_words_suf2.cuda()
            prev_words_suf3 = prev_words_suf3.cuda()
            prev_words_pref2 = prev_words_pref2.cuda()
            prev_words_pref3 = prev_words_pref3.cuda()
            prev_words_feat = prev_words_feat.cuda()

            prev_prev_words = prev_prev_words.cuda()
            prev_prev_words_suf2 = prev_prev_words_suf2.cuda()
            prev_prev_words_suf3 = prev_prev_words_suf3.cuda()
            prev_prev_words_pref2 = prev_prev_words_pref2.cuda()
            prev_prev_words_pref3 = prev_prev_words_pref3.cuda()
            prev_prev_words_feat = prev_prev_words_feat.cuda()

            next_words = next_words.cuda()
            next_words_suf2 = next_words_suf2.cuda()
            next_words_suf3 = next_words_suf3.cuda()
            next_words_pref2 = next_words_pref2.cuda()
            next_words_pref3 = next_words_pref3.cuda()
            next_words_feat = next_words_feat.cuda()

            next_next_words = next_next_words.cuda()
            next_next_words_suf2 = next_next_words_suf2.cuda()
            next_next_words_suf3 = next_next_words_suf3.cuda()
            next_next_words_pref2 = next_next_words_pref2.cuda()
            next_next_words_pref3 = next_next_words_pref3.cuda()
            next_next_words_feat = next_next_words_feat.cuda()

            prev_pos = prev_pos.cuda()
            prev_prev_pos = prev_prev_pos.cuda()

        words = autograd.Variable(words)
        words_suf2 = autograd.Variable(words_suf2)
        words_suf3 = autograd.Variable(words_suf3)
        words_pref2 = autograd.Variable(words_pref2)
        words_pref3 = autograd.Variable(words_pref3)
        words_feat = autograd.Variable(words_feat)

        prev_words = autograd.Variable(prev_words)
        prev_words_suf2 = autograd.Variable(prev_words_suf2)
        prev_words_suf3 = autograd.Variable(prev_words_suf3)
        prev_words_pref2 = autograd.Variable(prev_words_pref2)
        prev_words_pref3 = autograd.Variable(prev_words_pref3)
        prev_words_feat = autograd.Variable(prev_words_feat)

        prev_prev_words = autograd.Variable(prev_prev_words)
        prev_prev_words_suf2 = autograd.Variable(prev_prev_words_suf2)
        prev_prev_words_suf3 = autograd.Variable(prev_prev_words_suf3)
        prev_prev_words_pref2 = autograd.Variable(prev_prev_words_pref2)
        prev_prev_words_pref3 = autograd.Variable(prev_prev_words_pref3)
        prev_prev_words_feat = autograd.Variable(prev_prev_words_feat)

        next_words = autograd.Variable(next_words)
        next_words_suf2 = autograd.Variable(next_words_suf2)
        next_words_suf3 = autograd.Variable(next_words_suf3)
        next_words_pref2 = autograd.Variable(next_words_pref2)
        next_words_pref3 = autograd.Variable(next_words_pref3)
        next_words_feat = autograd.Variable(next_words_feat)

        next_next_words = autograd.Variable(next_next_words)
        next_next_words_suf2 = autograd.Variable(next_next_words_suf2)
        next_next_words_suf3 = autograd.Variable(next_next_words_suf3)
        next_next_words_pref2 = autograd.Variable(next_next_words_pref2)
        next_next_words_pref3 = autograd.Variable(next_next_words_pref3)
        next_next_words_feat = autograd.Variable(next_next_words_feat)

        prev_pos = autograd.Variable(prev_pos)
        prev_prev_pos = autograd.Variable(prev_prev_pos)

        outputs = model(words, words_suf2, words_suf3, words_pref2, words_pref3, words_feat,
                        prev_words, prev_words_suf2, prev_words_suf3, prev_words_pref2, prev_words_pref3,
                        prev_words_feat,
                        prev_prev_words, prev_prev_words_suf2, prev_prev_words_suf3, prev_prev_words_pref2,
                        prev_prev_words_pref3, prev_prev_words_feat,
                        next_words, next_words_suf2, next_words_suf3, next_words_pref2, next_words_pref3,
                        next_words_feat,
                        next_next_words, next_next_words_suf2, next_next_words_suf3, next_next_words_pref2,
                        next_next_words_pref3, next_next_words_feat,
                        prev_pos, prev_prev_pos)
        preds += list(outputs.data.cpu().numpy())
    return preds


def torch_train(train_samples, dev_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    model = get_model()
    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()
    loss_func = nn.NLLLoss(ignore_index=pos_map['<PAD>'])
    optimizer = optim.Adam(model.parameters())
    custom_print(optimizer)
    best_dev_acc = -1.0
    for epoch_idx in range(0, num_epoch):
        model.train()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1
        np.random.seed(cur_seed)
        torch.cuda.manual_seed(cur_seed)
        random.seed(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss = 0.0
        for batch_idx in tqdm(range(0, batch_count)):
            cur_batch_len, cur_seq_len, cur_samples_input, cur_samples_target\
                = get_training_input(cur_shuffled_train_data, batch_idx, batch_size, training=True)

            words = torch.from_numpy(cur_samples_input['words'].astype('long'))
            words_suf2 = torch.from_numpy(cur_samples_input['words_suf2'].astype('long'))
            words_suf3 = torch.from_numpy(cur_samples_input['words_suf3'].astype('long'))
            words_pref2 = torch.from_numpy(cur_samples_input['words_pref2'].astype('long'))
            words_pref3 = torch.from_numpy(cur_samples_input['words_pref3'].astype('long'))
            words_feat = torch.from_numpy(cur_samples_input['words_feat'].astype('float32'))

            prev_words = torch.from_numpy(cur_samples_input['prev_words'].astype('long'))
            prev_words_suf2 = torch.from_numpy(cur_samples_input['prev_words_suf2'].astype('long'))
            prev_words_suf3 = torch.from_numpy(cur_samples_input['prev_words_suf3'].astype('long'))
            prev_words_pref2 = torch.from_numpy(cur_samples_input['prev_words_pref2'].astype('long'))
            prev_words_pref3 = torch.from_numpy(cur_samples_input['prev_words_pref3'].astype('long'))
            prev_words_feat = torch.from_numpy(cur_samples_input['prev_words_feat'].astype('float32'))

            prev_prev_words = torch.from_numpy(cur_samples_input['prev_prev_words'].astype('long'))
            prev_prev_words_suf2 = torch.from_numpy(cur_samples_input['prev_prev_words_suf2'].astype('long'))
            prev_prev_words_suf3 = torch.from_numpy(cur_samples_input['prev_prev_words_suf3'].astype('long'))
            prev_prev_words_pref2 = torch.from_numpy(cur_samples_input['prev_prev_words_pref2'].astype('long'))
            prev_prev_words_pref3 = torch.from_numpy(cur_samples_input['prev_prev_words_pref3'].astype('long'))
            prev_prev_words_feat = torch.from_numpy(cur_samples_input['prev_prev_words_feat'].astype('float32'))

            next_words = torch.from_numpy(cur_samples_input['next_words'].astype('long'))
            next_words_suf2 = torch.from_numpy(cur_samples_input['next_words_suf2'].astype('long'))
            next_words_suf3 = torch.from_numpy(cur_samples_input['next_words_suf3'].astype('long'))
            next_words_pref2 = torch.from_numpy(cur_samples_input['next_words_pref2'].astype('long'))
            next_words_pref3 = torch.from_numpy(cur_samples_input['next_words_pref3'].astype('long'))
            next_words_feat = torch.from_numpy(cur_samples_input['next_words_feat'].astype('float32'))

            next_next_words = torch.from_numpy(cur_samples_input['next_next_words'].astype('long'))
            next_next_words_suf2 = torch.from_numpy(cur_samples_input['next_next_words_suf2'].astype('long'))
            next_next_words_suf3 = torch.from_numpy(cur_samples_input['next_next_words_suf3'].astype('long'))
            next_next_words_pref2 = torch.from_numpy(cur_samples_input['next_next_words_pref2'].astype('long'))
            next_next_words_pref3 = torch.from_numpy(cur_samples_input['next_next_words_pref3'].astype('long'))
            next_next_words_feat = torch.from_numpy(cur_samples_input['next_next_words_feat'].astype('float32'))

            prev_pos = torch.from_numpy(cur_samples_input['prev_pos'].astype('long'))
            prev_prev_pos = torch.from_numpy(cur_samples_input['prev_prev_pos'].astype('long'))

            cur_target = torch.from_numpy(cur_samples_target['labels'].astype('long'))

            if torch.cuda.is_available():
                words = words.cuda()
                words_suf2 = words_suf2.cuda()
                words_suf3 = words_suf3.cuda()
                words_pref2 = words_pref2.cuda()
                words_pref3 = words_pref3.cuda()
                words_feat = words_feat.cuda()

                prev_words = prev_words.cuda()
                prev_words_suf2 = prev_words_suf2.cuda()
                prev_words_suf3 = prev_words_suf3.cuda()
                prev_words_pref2 = prev_words_pref2.cuda()
                prev_words_pref3 = prev_words_pref3.cuda()
                prev_words_feat = prev_words_feat.cuda()

                prev_prev_words = prev_prev_words.cuda()
                prev_prev_words_suf2 = prev_prev_words_suf2.cuda()
                prev_prev_words_suf3 = prev_prev_words_suf3.cuda()
                prev_prev_words_pref2 = prev_prev_words_pref2.cuda()
                prev_prev_words_pref3 = prev_prev_words_pref3.cuda()
                prev_prev_words_feat = prev_prev_words_feat.cuda()

                next_words = next_words.cuda()
                next_words_suf2 = next_words_suf2.cuda()
                next_words_suf3 = next_words_suf3.cuda()
                next_words_pref2 = next_words_pref2.cuda()
                next_words_pref3 = next_words_pref3.cuda()
                next_words_feat = next_words_feat.cuda()

                next_next_words = next_next_words.cuda()
                next_next_words_suf2 = next_next_words_suf2.cuda()
                next_next_words_suf3 = next_next_words_suf3.cuda()
                next_next_words_pref2 = next_next_words_pref2.cuda()
                next_next_words_pref3 = next_next_words_pref3.cuda()
                next_next_words_feat = next_next_words_feat.cuda()

                prev_pos = prev_pos.cuda()
                prev_prev_pos = prev_prev_pos.cuda()

                cur_target = cur_target.cuda()

            words = autograd.Variable(words)
            words_suf2 = autograd.Variable(words_suf2)
            words_suf3 = autograd.Variable(words_suf3)
            words_pref2 = autograd.Variable(words_pref2)
            words_pref3 = autograd.Variable(words_pref3)
            words_feat = autograd.Variable(words_feat)

            prev_words = autograd.Variable(prev_words)
            prev_words_suf2 = autograd.Variable(prev_words_suf2)
            prev_words_suf3 = autograd.Variable(prev_words_suf3)
            prev_words_pref2 = autograd.Variable(prev_words_pref2)
            prev_words_pref3 = autograd.Variable(prev_words_pref3)
            prev_words_feat = autograd.Variable(prev_words_feat)

            prev_prev_words = autograd.Variable(prev_prev_words)
            prev_prev_words_suf2 = autograd.Variable(prev_prev_words_suf2)
            prev_prev_words_suf3 = autograd.Variable(prev_prev_words_suf3)
            prev_prev_words_pref2 = autograd.Variable(prev_prev_words_pref2)
            prev_prev_words_pref3 = autograd.Variable(prev_prev_words_pref3)
            prev_prev_words_feat = autograd.Variable(prev_prev_words_feat)

            next_words = autograd.Variable(next_words)
            next_words_suf2 = autograd.Variable(next_words_suf2)
            next_words_suf3 = autograd.Variable(next_words_suf3)
            next_words_pref2 = autograd.Variable(next_words_pref2)
            next_words_pref3 = autograd.Variable(next_words_pref3)
            next_words_feat = autograd.Variable(next_words_feat)

            next_next_words = autograd.Variable(next_next_words)
            next_next_words_suf2 = autograd.Variable(next_next_words_suf2)
            next_next_words_suf3 = autograd.Variable(next_next_words_suf3)
            next_next_words_pref2 = autograd.Variable(next_next_words_pref2)
            next_next_words_pref3 = autograd.Variable(next_next_words_pref3)
            next_next_words_feat = autograd.Variable(next_next_words_feat)

            prev_pos = autograd.Variable(prev_pos)
            prev_prev_pos = autograd.Variable(prev_prev_pos)

            cur_target = autograd.Variable(cur_target)

            model.zero_grad()
            outputs = model(words, words_suf2, words_suf3, words_pref2, words_pref3, words_feat,
                            prev_words, prev_words_suf2, prev_words_suf3, prev_words_pref2, prev_words_pref3,
                            prev_words_feat,
                            prev_prev_words, prev_prev_words_suf2, prev_prev_words_suf3, prev_prev_words_pref2,
                            prev_prev_words_pref3, prev_prev_words_feat,
                            next_words, next_words_suf2, next_words_suf3, next_words_pref2, next_words_pref3,
                            next_words_feat,
                            next_next_words, next_next_words_suf2, next_next_words_suf3, next_next_words_pref2,
                            next_next_words_pref3, next_next_words_feat,
                            prev_pos, prev_prev_pos, training=True)

            loss = loss_func(outputs, cur_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.data[0]

        train_loss /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training Loss:', train_loss)
        custom_print('Training Time:', end_time - start_time)

        # custom_print('\nDev Results\n')
        torch.cuda.manual_seed(random_seed)
        dev_preds = predict(dev_samples, model)
        dev_acc = get_accuracy(dev_samples, dev_preds)
        custom_print('Dev Acc.:', dev_acc)
        if dev_acc >= best_dev_acc:
            custom_print('Model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)
        custom_print('\n')


if __name__ == "__main__":
    job_name = sys.argv[1]
    vocab_file = sys.argv[2]
    word_embed_dim = int(sys.argv[3])

    batch_size = 32
    num_epoch = 10

    other_embed_dim = 25
    max_sent_len = 50

    Sample = namedtuple("Sample", "Len Words PrevWords PrevPrevWords NextWords NextNextWords"
                                  " PrevPOS PrevPrevPOS Labels")
    custom_print("loading word vectors......")
    with open(vocab_file, 'rb') as f:
        word_vocab, word_embed_matrix, suf2_dict, suf3_dict, pref2_dict, pref3_dict, pos_map, reverse_pos_map = \
            pickle.load(f)
    if job_name == 'train':
        train_file = sys.argv[4]
        model_file = sys.argv[5]
        custom_print(sys.argv)
        custom_print('loading data......')
        data = get_training_data(train_file)
        cut_point = int(len(data) * 0.1)
        dev_data = data[:cut_point]
        train_data = data[cut_point:]
        custom_print('Training data size:', len(train_data))
        custom_print('Dev data size:', len(dev_data))
        custom_print("training started......")
        torch_train(train_data, dev_data, model_file)
    if job_name == 'test':
        model_file = sys.argv[4]
        test_file = sys.argv[5]
        out_file = sys.argv[6]
        test_data = get_test_data(test_file)
        best_model = get_model()
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load(model_file))
        torch.cuda.manual_seed(random_seed)
        test_preds = predict(test_data, best_model)
        write_pred_file(test_preds, test_data, out_file)
        custom_print('finished.....')

























































