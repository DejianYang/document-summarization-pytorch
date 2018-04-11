import torch
from torch.autograd import Variable


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.utils.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_ids, src_oov_ids, oov_dict = self.src_vocab.copy_convert2idx(src_seq)

        src_id_seq = Variable(torch.LongTensor(src_ids), volatile=True).view(1, -1)
        src_oov_seq = Variable(torch.LongTensor(src_oov_ids), volatile=True).view(1, -1)

        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
            src_oov_seq = src_oov_seq.cuda()

        softmax_list, _, other = self.model((src_id_seq, src_oov_seq), [len(src_seq)])
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = self.tgt_vocab.convert2words(tgt_id_seq)
        return tgt_seq

    def predict_file(self, src_input_file, tgt_output_file):
        print('predict the file from %s' % src_input_file)
        src_lines = [line.strip().split() for line in open(src_input_file, 'r', encoding='utf-8')]
        with open(tgt_output_file, 'w', encoding='utf-8') as fw:
            for i, src_inp_seq in enumerate(src_lines):
                tgt_pred_seq = self.predict(src_inp_seq)
                fw.write(' '.join(tgt_pred_seq)+'\n')
                if (i+1) % 100 == 0:
                    print('...predict %d lines' % (i+1))
        print('predict over!')
