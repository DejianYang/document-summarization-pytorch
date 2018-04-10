# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

x = torch.rand(2, 5)
print(x)
output = torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
print(output)

x = torch.FloatTensor([9])
id = torch.LongTensor([[4, 3]])

output = torch.zeros(2, 8).scatter_add_(0, id, x)
print(output)
#
# embs = Variable(torch.rand(10, 5))
# idx = Variable(torch.LongTensor([[1, 3, 2], [3, 2, 1]]))
#
# print(embs)
# print(idx)
#
# seq_length = idx.size(1)
#
# for i in range(seq_length):
#     r = Variable(torch.rand(5))
#     print(embs[i, :], r)
#     embs[i, :] = embs[i, :] + r
#
#     print(embs[:, i])
#     break
#     pass
#
# memory = Variable(torch.rand(4, 3, 5))
# batch_size, seq_len, dim = memory.size()
#
# coverage = Variable(memory.data.new(batch_size, seq_len).zero_())
# print(memory)
# print(coverage)
#
# m = nn.Conv2d(1, 10, (1, 1), stride=1)
# inputs = Variable(torch.randn(4, 1, 1, 3))
# output = m(inputs)
# print(output)
# output2 = m(coverage.view_as(inputs))
# print(output2)
# pass
