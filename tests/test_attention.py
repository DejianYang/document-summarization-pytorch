# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.attention import BahdanauAttention2

attn = BahdanauAttention2(dim=4)

memory = Variable(torch.rand(2, 3, 4))
lengths = [3, 2]
coverage = Variable(torch.zeros(2, 3))
print(memory, lengths, coverage)

hidden = Variable(torch.rand(2, 4))

ctx, attn_dist = attn.forward(hidden, memory=memory, coverage=coverage, memory_length=lengths)
print(ctx)
print(attn_dist)
coverage += attn_dist
print('-'*10)
print(coverage)

ctx, attn_dist = attn.forward(hidden, memory=memory, coverage=coverage, memory_length=lengths)
print(ctx)
print(attn_dist)