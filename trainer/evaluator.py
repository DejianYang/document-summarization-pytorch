from __future__ import print_function, division

import torch
import torchtext
from models import NLLLoss
from utils.fields import *
from utils.dataset import *


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data.data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        tgt_vocab = data.vocab
        pad = tgt_vocab.pad_idx

        for batch_input in batch_iterator:
            src_inputs, src_lengths = getattr(batch_input, SRC_FILED_NAME)
            src_oov_inputs, _ = getattr(batch_input, SRC_OOV_FIELD_NAME)
            tgt_inputs = getattr(batch_input, TGT_FIELD_NAME)
            tgt_oov_inputs = getattr(batch_input, TGT_OOV_FIELD_NAME)

            loss = self.loss
            # Forward propagation
            decoder_outputs, decoder_hidden, other = model((src_inputs, src_oov_inputs),
                                                           src_lengths.tolist(), tgt_inputs)

            # Evaluation
            seqlist = other['sequence']
            for step, step_output in enumerate(decoder_outputs):
                target = tgt_oov_inputs[:, step + 1]
                loss.eval_batch(step_output.contiguous().view(tgt_oov_inputs.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].contiguous().view(-1).eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
