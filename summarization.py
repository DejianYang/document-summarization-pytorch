import os
import argparse
import logging

import torch
import torchtext

from trainer import SupervisedTrainer, Evaluator, Predictor
from models import EncoderRNN, DecoderRNN, TopKDecoder, Seq2Seq, Perplexity, Optimizer
from utils.fields import *
from utils.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    default='./data/valid.tsv', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    default='./data/valid.tsv', help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./data/summarization/',
                    help='Path to experiment directory. If load_checkpoint is True, '
                         'then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info', help='Logging level.')

opt = parser.parse_args()

src_max_len = 2000
tgt_max_len = 50


def len_filter(example):
    return len(example.src) <= src_max_len and len(example.tgt) <= src_max_len


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
log_file = os.path.join(opt.expt_dir, 'log.txt')
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Prepare dataset
src = SourceField()
tgt = TargetField()
train_set = torchtext.data.TabularDataset(
    path=opt.train_path, format='tsv',
    fields=[(SEQ2SEQ_SOURCE_FILED_NAME, src), (SEQ2SEQ_TARGET_FILED_NAME, tgt)],
    filter_pred=len_filter
)
valid_set = torchtext.data.TabularDataset(
    path=opt.dev_path, format='tsv',
    fields=[(SEQ2SEQ_SOURCE_FILED_NAME, src), (SEQ2SEQ_TARGET_FILED_NAME, tgt)],
    filter_pred=len_filter
)
print('training samples', len(train_set.examples))
print('valid samples', len(valid_set.examples))

src.build_vocab(train_set, max_size=30000)
tgt.build_vocab(train_set, max_size=30000)
input_vocab = src.vocab
output_vocab = tgt.vocab
# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)

print('use cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    loss.cuda()

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 256
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), src_max_len, hidden_size,
                             bidirectional=bidirectional,
                             rnn_cell='lstm',
                             variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), tgt_max_len, hidden_size * 2,
                             dropout_p=0.2, use_attention=True,
                             bidirectional=bidirectional,
                             rnn_cell='lstm',
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2Seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    trainer = SupervisedTrainer(loss=loss, batch_size=32,
                                checkpoint_every=1000,
                                print_every=100, expt_dir=opt.expt_dir)
    trainer.train(seq2seq, train_set,
                  num_epochs=5, dev_data=valid_set,
                  optimizer=optimizer,
                  teacher_forcing_ratio=0.5,
                  resume=opt.resume)

evaluator = Evaluator(loss=loss, batch_size=32)
dev_loss, accuracy = evaluator.evaluate(seq2seq, valid_set)

beam_search = Seq2Seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 5))
predictor = Predictor(beam_search, input_vocab, output_vocab)
predictor.predict_file('./data/valid.art', opt.expt_dir + '/valid.pred.sum')
