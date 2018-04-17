# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from .attention import Attention, BahdanauAttention, BahdanauAttention2
from .baseRNN import BaseRNN
from .DecoderRNN import DecoderRNN


class CopyDecoder(BaseRNN):
    """
    Decoder for Pointer-Generator
    """
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, oov_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=True, use_pointer=True):

        assert use_attention is True
        super(CopyDecoder, self).__init__(vocab_size, max_len, hidden_size,
                                          input_dropout_p, dropout_p,
                                          n_layers, rnn_cell)

        self.vocab_size = vocab_size
        self.oov_size = oov_size

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size + self.oov_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.use_pointer = use_pointer
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.attention = BahdanauAttention2(self.hidden_size * 2, self.hidden_size, use_coverage=True)

        self.linear_in = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.gen_output_project = nn.Linear(hidden_size * 2, self.vocab_size)
        self.gen_prob_project = nn.Linear(3 * hidden_size, 1)

        pass

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _copy_step(self, input_oov_idx, emb_inp, context, h_titled, attn_dist):
        batch_size = input_oov_idx.size(0)
        seq_length = input_oov_idx.size(1)

        gen_logits = self.gen_output_project(torch.cat((h_titled, context), dim=1))
        gen_logits = F.softmax(gen_logits, dim=1)

        gen_probs = F.sigmoid(self.gen_prob_project(torch.cat((emb_inp, context, h_titled), dim=1)))
        gen_logits = gen_probs * gen_logits

        gen_extend_logits = Variable(gen_logits.data.new(batch_size, self.oov_size).zero_())
        gen_logits = torch.cat((gen_logits, gen_extend_logits), dim=1)

        copy_logits = Variable(gen_logits.data.new(batch_size, self.vocab_size + self.oov_size).zero_())
        copy_logits = copy_logits.scatter_add(1, input_oov_idx, attn_dist * (1.0 - gen_probs))

        log_final_logits = torch.log(gen_logits + copy_logits)
        # print(log_final_logits.size())
        return log_final_logits

    def forward_step(self, input_var, input_oov_idx, coverage, hidden, memory, memory_length):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        # lstm cell
        if isinstance(hidden, tuple):
            h, c = hidden
            ctx, attn_dist, coverage = self.attention.forward(query=torch.cat((h.squeeze(0), c.squeeze(0)), dim=1),
                                                              coverage=coverage,
                                                              memory=memory,
                                                              memory_length=memory_length)

            rnn_inp = self.linear_in(torch.cat([ctx, embedded], dim=1))
            _, hidden = self.rnn(rnn_inp.unsqueeze(1), hidden)

            logits = self._copy_step(input_oov_idx, embedded, ctx, hidden[0].squeeze(0), attn_dist)
            return logits, hidden, attn_dist, coverage

        else:
            rnn_inp = self.linear_in(torch.cat([hidden[0], embedded], dim=1))
            _, h = self.rnn(rnn_inp.unsqueeze(1), hidden)
            h = h.squeeze(0)
            ctx, attn_dist = self.attention.forward(h, coverage, memory, memory_length)
            coverage += attn_dist
            h_titled = self.linear_out(torch.cat((ctx, h), dim=1))

            logits = F.log_softmax(self.gen_output_project(h_titled), dim=1)
            return logits, h_titled.unsqueeze(0), attn_dist, coverage

    def forward(self, inputs, oov_inputs, encoder_hidden, encoder_outputs, encoder_lengths, teacher_forcing_ratio=1.0):
        input_vars, batch_size, max_length = self._validate_args(inputs,
                                                                 encoder_hidden,
                                                                 encoder_outputs,
                                                                 teacher_forcing_ratio=teacher_forcing_ratio)

        # print(input_vars, batch_size, max_length)
        decoder_hidden = self._init_state(encoder_hidden)
        batch_size, seq_len, enc_dim = encoder_outputs.size()

        coverage = Variable(encoder_outputs.data.new(batch_size, seq_len).zero_())

        ret_dict = dict()
        ret_dict[self.KEY_ATTN_SCORE] = []

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        # use_teacher_forcing = True  # if random.random() < teacher_forcing_ratio else False

        for step in range(max_length):
            decoder_input_var = input_vars[:, step]

            decoder_logit, decoder_hidden, step_attn_dist, coverage = \
                self.forward_step(input_var=decoder_input_var,
                                  input_oov_idx=oov_inputs,
                                  hidden=decoder_hidden,
                                  coverage=coverage,
                                  memory=encoder_outputs,
                                  memory_length=encoder_lengths)

            decoder_outputs += [decoder_logit]
            ret_dict[self.KEY_ATTN_SCORE] += [step_attn_dist]
            symbols = decoder_logit.topk(1)[1]

            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

        # decoder_outputs = torch.stack(decoder_outputs, 1)
        # update predict ids and lengths
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()
        # print(len(decoder_outputs), decoder_outputs[0].size())
        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            # if teacher_forcing_ratio > 0:
            #     raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = Variable(torch.LongTensor([self.sos_id] * batch_size),
                              volatile=True).view(batch_size, 1)
            inputs = self.to_cuda(inputs)

            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
