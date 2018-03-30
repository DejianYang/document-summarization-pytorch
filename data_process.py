# -*- coding:utf-8 -*-


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
    merge_into_tsv_file('./data/train.art', './data/train.sum', './data/train.tsv')
    merge_into_tsv_file('./data/valid.art', './data/valid.sum', './data/valid.tsv')
