# -*- coding:utf-8 -*-
import torch
import json
import argparse
from models import Perplexity, Optimizer, EncoderRNN, CopyDecoder, DecoderRNN, Seq2Seq
from utils.dataset import *
from trainer import SupervisedTrainer, Evaluator, Predictor


def load_config_and_setup_log(path):
    config = json.load(open(path, 'r', encoding='utf-8'))

    # setup log print format
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_file = os.path.join(config['log_dir'], 'log.txt')
    logging.basicConfig(format=log_format, level=logging.INFO, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return config


def train(args):
    config = load_config_and_setup_log(args.config)
    logging.info(config)

    train_set, valid_set, vocab = load_dataset(train_path=config['train_path'],
                                               valid_path=config['valid_path'],
                                               vocab_path=config['vocab_path'],
                                               vocab_size=config['vocab_size'],
                                               max_oov_size=config['max_oov_size'],
                                               src_max_len=config['src_max_len'],
                                               tgt_max_len=config['tgt_max_len'])

    logging.info('train samples: %d' % len(list(train_set.data.examples)))
    logging.info('valid samples: %d' % len(list(valid_set.data.examples)))

    logging.info('building loss')

    # Prepare loss
    weight = torch.ones(len(vocab))
    loss = Perplexity(weight, vocab.pad_idx)

    if torch.cuda.is_available():
        loss.cuda()

    hidden_size = config['hidden_size']
    bidirectional = config['bidirectional']
    encoder = EncoderRNN(len(vocab), config['src_max_len'], hidden_size,
                         bidirectional=bidirectional,
                         rnn_cell=config['rnn_cell'],
                         variable_lengths=True)
    decoder = DecoderRNN(len(vocab), config['tgt_max_len'], hidden_size * 2,
                         dropout_p=config['dropout_prob'], use_attention=True,
                         bidirectional=bidirectional,
                         rnn_cell=config['rnn_cell'],
                         eos_id=vocab.eos_idx, sos_id=vocab.sos_idx)
    seq2seq = Seq2Seq(encoder, decoder)
    print(seq2seq)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-config['init_w'], config['init_w'])

    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=config['max_grad_norm'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, config['checkout_every'], gamma=0.9)
    optimizer.set_scheduler(scheduler)

    # train
    trainer = SupervisedTrainer(loss=loss, batch_size=config['batch_size'],
                                checkpoint_every=config['checkpoint_every'],
                                print_every=config['display_every'], expt_dir=config['log_dir'])
    trainer.train(seq2seq, train_set.data,
                  num_epochs=config['num_epochs'], dev_data=valid_set.data,
                  optimizer=optimizer,
                  teacher_forcing_ratio=config['teacher_forcing_ratio'],
                  resume=None)

    pass


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Single Document Summarization')

    # data
    parser.add_argument("--config", type=str, default="./configs/sum.json", help="config path")

    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
    pass
