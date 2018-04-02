# -*- coding:utf-8 -*-
from utils.dataset import *

train_set, valid_set, vocab = load_dataset(train_path='../data/valid.tsv',
                                           valid_path='../data/valid.tsv',
                                           vocab_path='../data/vocab.txt',
                                           vocab_size=10000,
                                           max_oov_size=500,
                                           src_max_len=1000,
                                           tgt_max_len=50)
print(len(vocab))

data_iter = torchtext.data.BucketIterator(
    dataset=train_set.data, batch_size=3,
    sort=False, sort_within_batch=True, device=-1,
    sort_key=lambda x: len(getattr(x, train_set.source_field)), repeat=False)

for batch_input in data_iter.__iter__():
    src_input, src_length = getattr(batch_input, train_set.source_field)
    src_oov_input, src_length = getattr(batch_input, train_set.source_oov_field)

    print(src_input, src_oov_input)
    break


