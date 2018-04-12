# -*- coding:utf-8 -*-
import torch
import json
import argparse
import logging
from models import Perplexity, Optimizer, EncoderRNN, CopyDecoder, TopKDecoder, Seq2Seq
from utils.dataset import *
from trainer import SupervisedTrainer2, Evaluator, Predictor


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'


def load_config(path):
    config = json.load(open(path, 'r', encoding='utf-8'))
    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])
    log_file = os.path.join(config['log_dir'], 'log.txt')
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    config_saved_path = os.path.join(config['log_dir'], 'config.json')
    logging.info('save current config into {}'.format(config_saved_path))
    json.dump(config, open(config_saved_path, 'w', encoding='utf-8'))

    return config


def train(args):
    config = load_config(args.config)

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
    weight = torch.ones(len(vocab)+vocab.oov_size)
    loss = Perplexity(weight, vocab.pad_idx)

    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None

    if not args.resume:
        hidden_size = config['hidden_size']
        bidirectional = config['bidirectional']
        encoder = EncoderRNN(len(vocab), config['src_max_len'], hidden_size,
                             bidirectional=bidirectional,
                             rnn_cell=config['rnn_cell'],
                             variable_lengths=True)
        decoder = CopyDecoder(len(vocab), vocab.oov_size,
                              config['tgt_max_len'], hidden_size * 2,
                              dropout_p=config['dropout_prob'], use_attention=True,
                              bidirectional=bidirectional,
                              rnn_cell=config['rnn_cell'],
                              eos_id=vocab.eos_idx, sos_id=vocab.sos_idx, use_pointer=True)
        seq2seq = Seq2Seq(encoder, decoder)
        print(seq2seq)

        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-config['init_w'], config['init_w'])

        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=config['max_grad_norm'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, 1000, 0.9)
        optimizer.set_scheduler(scheduler)

    # train
    trainer = SupervisedTrainer2(loss=loss, batch_size=config['batch_size'],
                                 checkpoint_every=config['checkpoint_every'],
                                 print_every=config['display_every'], expt_dir=config['log_dir'])
    seq2seq = trainer.train(seq2seq, train_set,
                            num_epochs=config['num_epochs'], dev_data=valid_set,
                            optimizer=optimizer,
                            teacher_forcing_ratio=config['teacher_forcing_ratio'],
                            resume=args.resume)

    evaluator = Evaluator(loss=loss, batch_size=config['batch_size'])
    dev_loss, accuracy = evaluator.evaluate(seq2seq, valid_set)
    logging.info("Dev Loss: %f; Dev Accuracy: %f" % (dev_loss, accuracy))

    beam_search = Seq2Seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, config['beam_size']))
    predictor = Predictor(beam_search, vocab, vocab)
    predictor.predict_file('./data/valid.art', config['log_dir'] + '/valid.pred.e%d.sum' % config['num_epochs'])

    pass


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Single Document Summarization')

    # data
    parser.add_argument("--config", type=str, default="./configs/sum.test.json", help="config path")
    parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
    pass
