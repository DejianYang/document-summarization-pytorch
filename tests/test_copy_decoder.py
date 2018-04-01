import torch
import torch.nn as nn
from torch.autograd import Variable

from models import CopyDecoder
from models import EncoderRNN

input_var = Variable(torch.LongTensor([[1, 2, 3, 0], [3, 4, 0, 0]]))
input_length = torch.LongTensor([3, 2])

print(input_var, input_length)

encoder = EncoderRNN(100, 10, 4, rnn_cell='lstm', bidirectional=True)
decoder = CopyDecoder(100, 10, 8, 2, 3, rnn_cell='lstm', bidirectional=True)
print(encoder)
encoder_outputs, encoder_state = encoder(input_var, input_length)

print('encoder results')
print(encoder_outputs, encoder_state)

decoder_logits, _, ret_dict = decoder.forward(inputs=input_var,
                                              encoder_hidden=encoder_state,
                                              encoder_outputs=encoder_outputs,
                                              encoder_lengths=input_length)

print()
print("attention")
for attn_dist in ret_dict[decoder.KEY_ATTN_SCORE]:
    print(attn_dist)
