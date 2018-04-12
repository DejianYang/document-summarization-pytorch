# -*- coding:utf-8 -*-
import os
import argparse
import torch
import logging
import json
from utils.dataset import load_dataset_from_file
from utils.checkpoint import *
from models import Perplexity, Optimizer, EncoderRNN, CopyDecoder, TopKDecoder, Seq2Seq
from trainer import SupervisedTrainer2, Evaluator, Predictor

LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)


def infer(args):
    config_path = os.path.join(args.output, 'config.json')
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    assert args.load_checkpoint is not None
    logging.info("loading checkpoint from {}".format(
        os.path.join(config['log_dir'], Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)))
    checkpoint_path = os.path.join(config['log_dir'], Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)

    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    # print model
    logging.info(seq2seq)

    # load valid set
    valid_set = load_dataset_from_file(data_path=config['valid_path'],
                                       vocab=input_vocab,
                                       src_max_len=config['src_max_len'],
                                       tgt_max_len=config['tgt_max_len'])

    # Prepare loss
    weight = torch.ones(len(output_vocab) + output_vocab.oov_size)
    loss = Perplexity(weight, output_vocab.pad_idx)

    if torch.cuda.is_available():
        loss.cuda()
    # evaluator = Evaluator(loss=loss, batch_size=config['batch_size'])

    # dev_loss, accuracy = evaluator.evaluate(seq2seq, valid_set)
    # logging.info("Dev Loss: %f; Dev Accuracy: %f" % (dev_loss, accuracy))

    beam_search = Seq2Seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, config['beam_size']))
    predictor = Predictor(beam_search, input_vocab, output_vocab)
    predictor.predict_file('./data/valid.art', config['log_dir'] + '/valid.pred.{}.sum'.format(args.load_checkpoint))


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Single Document Summarization')
    # data
    parser.add_argument("--output", type=str, default="./data/sum_test", help="config path")

    # checkout point path
    parser.add_argument("--load_checkpoint", type=str, default="2018_04_12_12_29_02",
                        help="checkout point path")

    return parser.parse_args()


if __name__ == '__main__':
    infer(parse_args())
