# -*- coding:utf-8 -*-
import os
import collections
import logging
import torchtext
from .vocab import *

SRC_FILED_NAME = 'src'
SRC_OOV_FIELD_NAME = 'src_oov'
TGT_FIELD_NAME = 'tgt'
TGT_OOV_FIELD_NAME = 'tgt_oov'


class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, pad_idx, unk_idx, sos_idx, eos_idx, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['sequential'] = True
        kwargs['batch_first'] = True
        kwargs['include_lengths'] = True
        kwargs['pad_token'] = pad_idx
        kwargs['unk_token'] = unk_idx
        kwargs['preprocessing'] = lambda seq: [int(i) for i in seq]

        super(SourceField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, pad_idx, unk_idx, sos_idx, eos_idx, **kwargs):
        kwargs['use_vocab'] = False
        kwargs['sequential'] = True
        kwargs['batch_first'] = True
        kwargs['include_lengths'] = False
        kwargs['pad_token'] = pad_idx
        kwargs['unk_token'] = unk_idx
        kwargs['preprocessing'] = lambda seq: [sos_idx] + [int(i) for i in seq] + [eos_idx]

        super(TargetField, self).__init__(**kwargs)


class PointerTextDataset(object):
    def __init__(self, vocab, csv_path, src_max_len, tgt_max_len):
        self._vocab = vocab
        src_field = SourceField(pad_idx=vocab.pad_idx, unk_idx=vocab.unk_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        tgt_field = TargetField(pad_idx=vocab.pad_idx, unk_idx=vocab.unk_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)

        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self._data_set = torchtext.data.TabularDataset(
            path=csv_path, format='tsv',
            fields=[(SRC_FILED_NAME, src_field),
                    (SRC_OOV_FIELD_NAME, src_field),
                    (TGT_FIELD_NAME, tgt_field),
                    (TGT_OOV_FIELD_NAME, tgt_field)],
            filter_pred=self._len_filter)
        pass

    def _len_filter(self, example):
        src = getattr(example, SRC_FILED_NAME)
        tgt = getattr(example, TGT_FIELD_NAME)
        return len(src) <= self.src_max_len and len(tgt) <= self.tgt_max_len

    @property
    def data(self):
        return self._data_set

    @property
    def vocab(self):
        return self._vocab


def _generate_pointer_text(vocab, input_file, saved_file):
    src_sents, tgt_sents = [], []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            ss = line.strip().split('\t')
            src_sents.append(ss[0].split())
            tgt_sents.append(ss[1].split())

    assert len(src_sents) == len(tgt_sents)

    with open(saved_file, 'w', encoding='utf-8') as fw:
        for src_words, tgt_words in zip(src_sents, tgt_sents):
            src_ids, src_oov_ids, oov_dict = vocab.copy_covert2dix(src_words)
            tgt_ids, tgt_oov_ids, _ = vocab.copy_covert2dix(tgt_words, oov_dict)
            s1 = ' '.join([str(i) for i in src_ids])
            s2 = ' '.join([str(i) for i in src_oov_ids])
            s3 = ' '.join([str(i) for i in tgt_ids])
            s4 = ' '.join([str(i) for i in tgt_oov_ids])
            fw.write('\t'.join([s1, s2, s3, s4]) + '\n')


def load_dataset(train_path, valid_path, vocab_path,
                 vocab_size, max_oov_size, src_max_len, tgt_max_len):
    logging.info("loading vocab from %s, vocab size: %d, max oov size: %d" %
                 (vocab_path, vocab_size, max_oov_size))
    vocab = load_vocabulary(vocab_path, vocab_size, max_oov_size)
    print(vocab.pad_idx, vocab.unk_idx, vocab.sos_idx, vocab.eos_idx)

    train_index_path = train_path.replace(".tsv", ".idx.tsv")
    valid_index_path = valid_path.replace(".tsv", ".idx.tsv")

    logging.info('generating pointer text data from %s to %s ' % (train_path, train_index_path))
    _generate_pointer_text(vocab, train_path, train_index_path)

    logging.info('generating pointer text data from %s to %s ' % (valid_path, valid_index_path))
    _generate_pointer_text(vocab, valid_path, valid_index_path)

    train_set = PointerTextDataset(vocab=vocab, csv_path=train_index_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len)
    valid_set = PointerTextDataset(vocab=vocab, csv_path=valid_index_path, src_max_len=src_max_len, tgt_max_len=tgt_max_len)

    return train_set, valid_set, vocab

