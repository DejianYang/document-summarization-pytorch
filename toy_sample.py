import os
import argparse
import torch
import torchtext
import random
import logging
from utils.fields import SourceField, TargetField
from models import Perplexity
from models import Optimizer
from models import CopyDecoder, EncoderRNN, Seq2Seq, TopKDecoder
from trainer import SupervisedTrainer, Predictor
from utils.checkpoint import Checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./data/toy/log',
                    help='Path to experiment directory. If load_checkpoint is True, '
                         'then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    # default="2018_03_31_20_13_17",
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()


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


LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

generate_dataset('./data/toy/train.tsv', 2000, 50, 100)
generate_dataset('./data/toy/valid.tsv', 200, 50, 100)

src = SourceField()
tgt = TargetField()


if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    topk_decoder = TopKDecoder(seq2seq.decoder, k=5)
    beam_search_model = Seq2Seq(seq2seq.encoder, topk_decoder)
    predictor = Predictor(beam_search_model, input_vocab, output_vocab)

    # while True:
    seq_str = "1 3 5 7 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "2 4 6 7 7 5 3 1 21 21 30"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "1 2 2 3 3 3"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "20 30 10 5 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "1 2 3 4 5 6 7 8 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))


else:
    train = torchtext.data.TabularDataset(
        path="./data/toy/train.tsv", format='tsv',
        fields=[('src', src), ('tgt', tgt)])

    dev = torchtext.data.TabularDataset(
        path="./data/toy/valid.tsv", format='tsv',
        fields=[('src', src), ('tgt', tgt)])

    src.build_vocab(train, max_size=100)
    tgt.build_vocab(train, max_size=100)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    # Initialize model
    hidden_size = 128
    max_len = 50
    bidirectional = True
    encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                         rnn_cell='lstm', bidirectional=bidirectional, variable_lengths=True)
    decoder = CopyDecoder(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                          rnn_cell='lstm', dropout_p=0.2, use_attention=True,
                          bidirectional=bidirectional, eos_id=tgt.eos_id, sos_id=tgt.sos_id)

    seq2seq = Seq2Seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    # train
    trainer = SupervisedTrainer(loss=loss, batch_size=32,
                                checkpoint_every=50,
                                print_every=10, expt_dir="./data/toy/log")

    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
    seq2seq = trainer.train(seq2seq, train,
                            num_epochs=15, dev_data=dev,
                            optimizer=optimizer,
                            teacher_forcing_ratio=0.5,
                            resume=False)
    topk_decoder = TopKDecoder(seq2seq.decoder, k=5)
    beam_search_model = Seq2Seq(seq2seq.encoder, topk_decoder)
    predictor = Predictor(beam_search_model, input_vocab, output_vocab)

    # while True:
    seq_str = "1 3 5 7 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "2 4 6 7 7 5 3 1 21 21 30"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "1 2 2 3 3 3"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "20 30 10 5 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

    seq_str = "1 2 3 4 5 6 7 8 9"  # aw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))

