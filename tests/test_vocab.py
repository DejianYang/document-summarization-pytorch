# -*- coding:utf-8 -*-
import random
from utils.vocab import *
import torchtext


def generate_dataset(path, num_samples, max_length, max_idx):
    # generate data file
    with open(path, 'w', encoding='utf-8') as fout:
        for _ in range(num_samples):
            length = random.randint(1, max_length)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, max_idx)))
            fout.write("\t".join([" ".join(seq), " ".join(reversed(seq))]))
            fout.write('\n')


generate_dataset('./data/toy.txt', 1000, 50, 70)

build_vocab('./data/toy.txt', './data/toy.vocab.txt')

vocab = load_vocabulary('./data/toy.vocab.txt', 50, 20)
print('vocab size', len(vocab), vocab.vocab_size)
print(vocab.pad_idx, vocab.unk_idx, vocab.sos_idx, vocab.eos_idx)

src_sents, tgt_sents = [], []
with open('./data/toy.txt', 'r', encoding='utf-8') as fr:
    for line in fr:
        ss = line.strip().split('\t')
        src_sents.append(ss[0].split())
        tgt_sents.append(ss[1].split())

with open('./data/toy.tsv', 'w', encoding='utf-8') as fw:
    for src_words, tgt_words in zip(src_sents, tgt_sents):
        src_ids, src_oov_ids, oov_dict = vocab.copy_covert2dix(src_words)
        tgt_ids, tgt_oov_ids, _ = vocab.copy_covert2dix(tgt_words, oov_dict)
        s1 = ' '.join([str(i) for i in src_ids])
        s2 = ' '.join([str(i) for i in src_oov_ids])
        s3 = ' '.join([str(i) for i in tgt_ids])
        s4 = ' '.join([str(i) for i in tgt_oov_ids])
        fw.write('\t'.join([s1, s2, s3, s4])+'\n')

SourceFiled = torchtext.data.Field(use_vocab=False,
                                   sequential=True,
                                   include_lengths=True,
                                   batch_first=True,
                                   pad_token=vocab.pad_idx,
                                   unk_token=vocab.unk_idx,
                                   preprocessing=lambda seq: [int(i) for i in seq])
TargetFiled = torchtext.data.Field(use_vocab=False,
                                   sequential=True,
                                   batch_first=True,
                                   unk_token=vocab.unk_idx,
                                   pad_token=vocab.pad_idx,
                                   preprocessing=lambda seq: [vocab.sos_idx]+[int(i) for i in seq]+[vocab.eos_idx])

data_set = torchtext.data.TabularDataset(
    path='./data/toy.tsv', format='tsv',
    fields=[("src", SourceFiled),
            ('src_oov', SourceFiled),
            ("tgt", TargetFiled),
            ('tgt_oov', TargetFiled)])

data_iterator = torchtext.data.BucketIterator(
    dataset=data_set, batch_size=3,
    sort=False, sort_within_batch=True, device=-1,
    sort_key=lambda x: len(x.src), repeat=False)


for batch_input in data_iterator.__iter__():
    print(batch_input)
    src_input, l1 = getattr(batch_input, 'src')
    src_oov_input, l2 = getattr(batch_input, 'src_oov')

    tgt_input = getattr(batch_input, 'tgt')
    tgt_oov_input = getattr(batch_input, 'tgt_oov')
    print('-'*100)
    print(src_input.size(), tgt_input.size())
    print(src_input, src_oov_input)
    print(tgt_input, tgt_oov_input)
    print(l1, l2)
    break
    pass
