# Neural-POS-Tagger
This repository contains two neural part-of-speech tagging models for English. One model is based on LSTM network and other one is based on feed-forward neural network.

GloVe Embeddings
----------------

Download the pre-trained word embeddings

Dataset
-------

Dataset is extracted from penn tree bank files

Requirements
------------

Python 3.5
Pytorch 0.2

LSTM Based Tagger
-----------------

1) Create the vocab.pkl file

python3.5 lstm-gen-vocab.py <training_file_path> <glove_file_path> <embedding_dimension> <vocab_file_path>

2) Train

CUDA_VISIBLE_DEVICES="0" python3.5 lstm-tagger.py train <vocab_file_path> <embedding_dimension> <training_file_path> <model_file_path>

3) Test

CUDA_VISIBLE_DEVICES="0" python3.5 lstm-tagger.py test <vocab_file_path> <embedding_dimension> <model_file_path> <test_file_path> <output_file_path>

4) Evaluation

python3.5 eval.py <output_file_path> <reference_file_path>

Feature Based Tagger
--------------------

1) Create the vocab.pkl file

python3.5 fnn-gen-vocab.py <training_file_path> <glove_file_path> <embedding_dimension> <vocab_file_path>

2) Train

CUDA_VISIBLE_DEVICES="0" python3.5 fnn-tagger.py train <vocab_file_path> <embedding_dimension> <training_file_path> <model_file_path>

3) Test

CUDA_VISIBLE_DEVICES="0" python3.5 fnn-tagger.py test <vocab_file_path> <embedding_dimension> <model_file_path> <test_file_path> <output_file_path>

4) Evaluation

python3.5 eval.py <output_file_path> <reference_file_path>

Notes
-----

model_file_path should have .h5py extension.

vocab_file_path should have .pkl extension
