# -*- coding:utf-8 -*-
import json
import os
from pyrouge import Rouge155


def load_chn_vocab(vocab_file):
    char2idx = {}
    with open(vocab_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            ss = line[:-1].split()
            if len(ss) != 2:
                print('waring ', line)
                continue
            char2idx[ss[0]] = int(ss[1])
    return char2idx


def _clean_summary_sentence(sentence):
    sentence = sentence.replace("<unk>", "")
    sentence = sentence.replace("<pad>", "")
    sentence = sentence.replace("<sos>", "")
    sentence = sentence.replace("<eos>", "")
    sentence = sentence.strip().replace(" ", "")
    return sentence


def _index_chn_char(vocab, sentence):
    idx_words = []
    if len(sentence) == 0:
        return ' '.join(['<unk>', '<pad>', '<sos>', '<eos>'])
    for chn_word in sentence:
        if chn_word in vocab:
            idx_words.append(str(vocab[chn_word]))
        else:
            # print(chn_word)
            idx_words.append(chn_word)

    return ' '.join(idx_words)


def prepare_data_for_rouge(file, saved_dir, prefix, vocab):
    saved_lines = []
    with open(file, 'r', encoding='utf-8') as fr:
        for line in fr:
            clean_sentence = _clean_summary_sentence(line.strip())
            saved_lines.append(_index_chn_char(vocab, clean_sentence))
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    for i, _idx_line in enumerate(saved_lines):
        with open(os.path.join(saved_dir, prefix + '.%05d.txt' % i), 'w', encoding='utf-8') as fw:
            fw.write(_idx_line)
    pass


def _evaluate_rouge(gold_dir, pred_dir, gold_prefix, pred_prefix):
    r = Rouge155()
    r.system_dir = gold_dir
    r.model_dir = pred_dir
    r.system_filename_pattern = gold_prefix + '.(\d+).txt'
    r.model_filename_pattern = pred_prefix + '.#ID#.txt'
    output = r.convert_and_evaluate()
    print('---------------------ROUGE----------------------------')
    print(output)
    output_dict = r.output_to_dict(output)
    return output_dict


def evaluate_rouge(vocab_file, gold_file, pred_file, gold_rouge_dir, gold_rouge_prefix,
                   pred_rouge_dir, pred_rouge_prefix, rouge_result_saved_path=None):
    vocab = load_chn_vocab(vocab_file)
    print('...prepare gold rouge file')
    prepare_data_for_rouge(gold_file, gold_rouge_dir, gold_rouge_prefix, vocab)
    print('...prepare predict rouge file')
    prepare_data_for_rouge(pred_file, pred_rouge_dir, pred_rouge_prefix, vocab)
    rouge_result = _evaluate_rouge(gold_rouge_dir, pred_rouge_dir, gold_rouge_prefix, pred_rouge_prefix)
    if rouge_result_saved_path is not None:
        json.dump(rouge_result, open(rouge_result_saved_path, 'w', encoding='utf-8'))
    return rouge_result


if __name__ == '__main__':
    # evaluate_rouge(vocab_file="../data/char_dict.txt",
    #                gold_file="../data/valid.sum", pred_file="../data/valid.pred.sum",
    #                gold_rouge_dir='../data/gold/', gold_rouge_prefix="valid",
    #                pred_rouge_dir="../data/pred/", pred_rouge_prefix="valid.pred",
    #                rouge_result_saved_path="../data/rouge.valid.pred.json")
    evaluate_rouge(vocab_file="../data/char_dict.txt",
                   gold_file="../data/valid.sum", pred_file="../data/valid.pred2.sum",
                   gold_rouge_dir='../data/gold/', gold_rouge_prefix="valid",
                   pred_rouge_dir="../data/pred2/", pred_rouge_prefix="valid.pred",
                   rouge_result_saved_path="../data/rouge.valid.pred3.json")
    pass
