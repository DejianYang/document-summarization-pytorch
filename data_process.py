# -*- coding:utf-8 -*-
import os
import json
import jieba
import numpy as np
from utils.vocab import build_vocab

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return [line.strip() for line in fr]


def tokenize(sentence):
    words = jieba.cut(sentence.replace(' ', '').strip())
    seg_words = []
    entity = []
    flag = False
    for w in words:
        if w == '《' or w == '【':
            seg_words.append(w)
            entity = []
            flag = True
        elif w == '》' or w == '】' or w == '<' or w == '>':
            seg_words.append(''.join(entity))
            entity = []
            seg_words.append(w)
            if w == '》' or w == '】':
                flag = False
        else:

            if flag:
                entity.append(w)
            else:
                seg_words.append(w)

    seg_str = ' '.join(seg_words)
    seg_str = seg_str.replace('< Paragraph >', '<Paragraph>')
    seg_str = seg_str.replace('< P ar agr a ph  >', '<Paragraph>')
    seg_str = seg_str.replace('< P ar agr a ph >', '<Paragraph>')
    return seg_str


def _load_raw_data(raw_path):
    samples = []
    with open(raw_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            obj = json.loads(line.strip('\r\n'))
            article = obj['article'].strip().replace('\n', '')
            summary = obj['summarization'].strip().replace('\n', '')
            samples.append((article, summary))
    return samples


def _filter_valid_samples(train_samples, valid_samples):
    train_sum_dict = {}
    for article, summary in train_samples:
        train_sum_dict[summary] = article

    new_valid_samples = []
    ignore_count = 0
    for article, summary in valid_samples:
        if summary in train_sum_dict:
            ignore_count += 1
            continue
        new_valid_samples.append((article, summary))

    print('ignore %d valid samples' % ignore_count)
    return train_samples, new_valid_samples


def _tokenize_and_save_data(samples, saved_dir, prefix):
    art_saved_path = os.path.join(saved_dir, '%s.art' % prefix)
    sum_saved_path = os.path.join(saved_dir, '%s.sum' % prefix)

    with open(art_saved_path, 'w', encoding='utf-8') as fw1, \
            open(sum_saved_path, 'w', encoding='utf-8') as fw2:
        for (article, summary) in samples:
            fw1.write(tokenize(article) + '\n')
            fw2.write(tokenize(summary) + '\n')


def _get_length_summary(data_dir, prefix):
    article_lines = load_lines(os.path.join(data_dir, '%s.art' % prefix))
    article_lengths = [len(a.split()) for a in article_lines]
    sorted_article_lengths = sorted(article_lengths)
    print('95\% length', sorted_article_lengths[int(len(sorted_article_lengths) * .95)])
    print('98\% length', sorted_article_lengths[int(len(sorted_article_lengths) * .98)])
    print('mean length', np.mean(sorted_article_lengths))
    print('max length', np.max(sorted_article_lengths))

    sum_lines = load_lines(os.path.join(data_dir, '%s.sum' % prefix))
    sum_lengths = [len(a.split()) for a in sum_lines]
    sorted_sum_lengths = sorted(sum_lengths)
    print('95\% length', sorted_sum_lengths[int(len(sorted_sum_lengths) * .95)])
    print('98\% length', sorted_sum_lengths[int(len(sorted_sum_lengths) * .98)])
    print('mean length', np.mean(sorted_sum_lengths))
    print('max length', np.max(sorted_sum_lengths))


def load_and_process_raw_data(train_raw_path, valid_raw_path, saved_dir):
    train_samples = _load_raw_data(train_raw_path)
    valid_samples = _load_raw_data(valid_raw_path)

    print('train samples', len(train_samples))
    print('valid samples', len(valid_samples))

    train_samples, valid_samples = _filter_valid_samples(train_samples, valid_samples)

    # load user defined dict
    jieba.add_word("<Paragraph>")
    jieba.load_userdict("../../DATA/baike.word.clean.txt")

    _tokenize_and_save_data(train_samples, saved_dir, 'train')
    _tokenize_and_save_data(valid_samples, saved_dir, 'valid')

    _get_length_summary(saved_dir, 'train')
    _get_length_summary(saved_dir, 'valid')
    pass


def merge_into_tsv_file(src_file, tgt_file, saved_file):
    src_lines = [line.strip() for line in open(src_file, 'r', encoding='utf-8')]
    tgt_lines = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8')]
    assert len(src_lines) == len(tgt_lines)
    with open(saved_file, 'w', encoding='utf-8') as fw:
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_line = src_line.replace('\t', ',')
            tgt_line = tgt_line.replace('\t', ',')
            fw.write('%s\t%s\n' % (src_line, tgt_line))


if __name__ == '__main__':
    train_path = "../../DATA/TTNewsCorpus_NLPCC2017/toutiao4nlpcc/train_with_summ.txt"
    valid_path = "../../DATA/TTNewsCorpus_NLPCC2017/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt"
    saved_data_dir = './data/'
    # load_and_process_raw_data(train_raw_path=train_path, valid_raw_path=valid_path, saved_dir=saved_data_dir)

    merge_into_tsv_file('./data/train.art', './data/train.sum', './data/train.tsv')
    merge_into_tsv_file('./data/valid.art', './data/valid.sum', './data/valid.tsv')

    print('... build vocab')
    build_vocab('./data/train.art', './data/vocab.txt')
    pass
