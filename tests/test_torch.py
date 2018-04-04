# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable

embs = Variable(torch.rand(10, 5))
idx = Variable(torch.LongTensor([[1, 3, 2], [3, 2, 1]]))

print(embs)
print(idx)

seq_length = idx.size(1)

for i in range(seq_length):
    r = Variable(torch.rand(5))
    print(embs[i, :], r)
    embs[i, :] = embs[i, :] + r

    print(embs[:, i])
    break
    pass
