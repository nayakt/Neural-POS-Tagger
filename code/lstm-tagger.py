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
        samples.append(Sample(Len=len(words), Words=words, Labels=labels))
    return samples


def get_test_data(file_name):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    samples = []
    for line in lines:
        labels = list()
        words = line.strip().split()
        sample = Sample(Len=len(words), Words=words, Labels=labels)
        samples.append(sample)
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
        if word in word_vocab:
            words_seq.append(word_vocab[word])
        else:
            words_seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        words_seq.append(word_vocab['<PAD>'])
    return words_seq


def get_char_seq(words, max_len):
    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])
    return char_seq


def get_feat(word):
    # 0 for padding, 1 for numbers, 2 for all lower, 3 for all upper, 4 for capitalized initial, 5 for others
    if word == word.lower():
        return 2
    if word == word.upper():
        return 3
    if word[0] == word[0].upper():
        return 4
    return 5


def get_featute_vec(words, max_len):
    feat_seq = list()
    for word in words:
        feat_seq.append(get_feat(word))
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        feat_seq.append(0)
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
    labels_list = list()
    char_seq_list = list()
    for sample in cur_batch:
        words_seq_list.append(get_words_index_seq(sample.Words, batch_max_len))
        char_seq_list.append(get_char_seq(sample.Words, batch_max_len))
        if training:
            labels_list += get_pos_index_seq(sample.Labels, batch_max_len)
    return batch_end - batch_start, batch_max_len, {'words_seq_input': np.array(words_seq_list),
                                                    'char_seq_input': np.array(char_seq_list)}, \
           {'labels': np.array(labels_list)}


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.drop_rate = 0.0
        self.class_count = len(reverse_pos_map)
        self.input_dim = word_embed_dim + char_embed_dim
        self.hidden_dim = self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.char_embeddings = nn.Embedding(len(char_vocab), char_embed_dim, padding_idx=0)

        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.conv_non_linear = nn.ReLU()
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 1, batch_first=True, dropout=self.drop_rate,
                            bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_dim, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, words_seq, char_seq, training=False):
        batch_len = words_seq.size()[0]
        seq_len = words_seq.size()[1]
        words_embed = self.word_embeddings(words_seq)
        words_embed = self.dropout(words_embed)

        char_embeds = self.char_embeddings(char_seq)
        char_embeds = self.dropout(char_embeds)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_conv = self.conv_non_linear(self.dropout(self.conv1d(char_embeds)))
        char_feature = self.max_pool(char_conv)
        char_feature = char_feature.permute(0, 2, 1)

        sent_input = torch.cat((words_embed, char_feature), 2)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(2, batch_len, self.hidden_dim)))
        h0 = h0.cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(2, batch_len, self.hidden_dim)))
        c0 = c0.cuda()
        sent_output, hc = self.lstm(sent_input, (h0, c0))
        seq_labels = self.linear(sent_output).view(-1, self.class_count)
        if training:
            probs = self.logsoftmax(seq_labels)
            probs = probs.view(batch_len * seq_len, self.class_count)
        else:
            probs = F.softmax(seq_labels).view(-1, seq_len, self.class_count)
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
        words_seq = torch.from_numpy(cur_samples_input['words_seq_input'].astype('long'))
        char_seq = torch.from_numpy(cur_samples_input['char_seq_input'].astype('long'))
        if torch.cuda.is_available():
            words_seq = words_seq.cuda()
            char_seq = char_seq.cuda()
        words_seq = autograd.Variable(words_seq)
        char_seq = autograd.Variable(char_seq)
        outputs = model(words_seq, char_seq)
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

            words_seq = torch.from_numpy(cur_samples_input['words_seq_input'].astype('long'))
            char_seq = torch.from_numpy(cur_samples_input['char_seq_input'].astype('long'))
            cur_target = torch.from_numpy(cur_samples_target['labels'].astype('long'))

            if torch.cuda.is_available():
                words_seq = words_seq.cuda()
                char_seq = char_seq.cuda()
                cur_target = cur_target.cuda()

            words_seq = autograd.Variable(words_seq)
            char_seq = autograd.Variable(char_seq)
            cur_target = autograd.Variable(cur_target)

            model.zero_grad()
            outputs = model(words_seq, char_seq, training=True)

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
    max_word_len = 10
    conv_filter_size = 3
    char_embed_dim = 50
    char_feature_size = 50
    max_sent_len = 50

    Sample = namedtuple("Sample", "Len Words Labels")
    custom_print("loading word vectors......")
    char_vocab, word_vocab, word_embed_matrix, pos_map, reverse_pos_map = load_vocab(vocab_file)

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

























































