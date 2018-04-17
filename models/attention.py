import torch
import torch.nn as nn
import torch.nn.functional as F


def _sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    lengths = torch.LongTensor(lengths)
    if torch.cuda.is_available():
        lengths = lengths.cuda()
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return 1 - (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.dim = dim
        # self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, hidden, memory, memory_length=None):
        seq_length = memory.size(1)
        # (batch*1*dim)*(batch*dim*seq_l) -> batch*1*seq_l = batch*seq_l
        attn = torch.bmm(hidden.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)

        if memory_length is not None:
            mask = _sequence_mask(memory_length, seq_length)
            attn.data.masked_fill_(mask, -float('inf'))

        # batch * seq_l
        attn = F.softmax(attn, dim=1)
        # (batch, 1, seq_l) * (batch, seq_l, dim) -> (batch, 1, dim) -> (batch, dim)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)

        # # concat -> (batch, out_len, 2*dim)
        # cont = torch.cat((mix, hidden), dim=1)
        # # output -> (batch, out_len, dim)
        # attn_output = F.tanh(self.linear_out(combined))
        return context, attn


class BahdanauAttention2(nn.Module):
    """
    Bahdanau Attention with Coverage
    """
    def __init__(self, input_size, hidden_size, use_coverage=False):
        """
         Note attention size is equal to hidden size by default
        :param input_size(int): query input size
        :param hidden_size(int): rnn and attention hidden size(memory hidden size)
        :param use_coverage(bool, optional): use coverage
        """
        super(BahdanauAttention2, self).__init__()
        self.use_coverage = use_coverage
        self.input_size = input_size
        self.attn_size = hidden_size

        self.linear_wh = nn.Linear(input_size, self.attn_size)
        self.linear_wm = nn.Linear(hidden_size, self.attn_size, bias=False)
        if self.use_coverage:
            self.linear_wc = nn.Linear(1, self.attn_size, bias=False)
        self.v = nn.Linear(self.attn_size, 1, bias=False)

    def forward(self, query, memory, coverage=None, memory_length=None):
        """
         Bahdanau Attention with coverage
        :param query: target sequence hidden states, [batch_size, input_size]
        :param memory: source sequence hidden states, [batch_size, src_length, hidden_size]
        :param coverage: coverage vector of attention, [batch_size, src_length]
        :param memory_length: source sequence lengths to calculate the attention mask, [batch]
        :return: context vector and attention distributions, new attention coverage vector
        """
        assert query.dim() == 2
        assert memory.dim() == 3
        tgt_batch, tgt_dim = query.size()
        src_batch, src_len, src_dim = memory.size()
        assert tgt_batch == src_batch

        if self.use_coverage:
            assert coverage is not None, "coverage can not be none when use coverage"
            assert coverage.dim() == 2
            assert coverage.size() == (src_batch, src_len)

        wh = self.linear_wh(query).unsqueeze(1).expand(src_batch, src_len, self.attn_size)
        wm = self.linear_wm(memory.contiguous().view(-1, src_dim)).view(src_batch, src_len, self.attn_size)

        if self.use_coverage:
            wc = self.linear_wc(coverage.contiguous().view(-1, 1)).view(src_batch, src_len, self.attn_size)
            attn_dist = self.v(F.tanh(wh+wm+wc).view(-1, src_dim)).view(src_batch, src_len)
        else:
            attn_dist = self.v(F.tanh(wh + wm).view(-1, src_dim)).view(src_batch, src_len)

        # apply mask
        if memory_length is not None:
            mask = _sequence_mask(memory_length, src_len)
            attn_dist.data.masked_fill_(mask, -float('inf'))
        # attention distributions
        attn_dist = F.softmax(attn_dist, dim=1)
        #print("src_len", src_len, "lengths", memory_length)
        #print("attn_dist", attn_dist)
        context = torch.bmm(attn_dist.unsqueeze(1), memory).squeeze(1)

        if coverage is not None:
            coverage = coverage + attn_dist
            return context, attn_dist, coverage

        return context, attn_dist
