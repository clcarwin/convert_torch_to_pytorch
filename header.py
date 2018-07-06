import torch
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func, self.forward_prepare(input))


class Padding(nn.Module):
    # pad puts in [pad] amount of [value] over dimension [dim], starting at
    # index [index] in that dimension. If pad<0, index counts from the left.
    # If pad>0 index counts from the right.
    # When nInputDim is provided, inputs larger than that value will be considered batches
    # where the actual dim to be padded will be dimension dim + 1.
    def __init__(self, dim, pad, value, index, nInputDim):
        super(Padding, self).__init__()
        self.value = value
        # self.index = index
        self.dim = dim
        self.pad = pad
        self.nInputDim = nInputDim
        if index != 0:
            raise NotImplementedError("Padding: index != 0 not implemented")

    def forward(self, input):
        dim = self.dim
        if self.nInputDim != 0:
            dim += input.dim() - self.nInputDim
        pad_size = list(input.size())
        pad_size[dim] = self.pad
        padder = Variable(input.data.new(*pad_size).fill_(self.value))

        if self.pad < 0:
            padded = torch.cat((padder, input), dim)
        else:
            padded = torch.cat((input, padder), dim)
        return padded


class Dropout(nn.Dropout):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """
    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """
    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout2d, self).forward(input)


class StatefulMaxPool2d(nn.MaxPool2d):  # object keeps indices and input sizes

    def __init__(self, *args, **kwargs):
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, x):
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = x.size()
        if return_indices:
            return output, indices
        return output


class StatefulMaxUnpool2d(nn.Module):
    def __init__(self, pooling):
        super(StatefulMaxUnpool2d, self).__init__()
        self.pooling = pooling
        self.unpooling = nn.MaxUnpool2d(pooling.kernel_size, pooling.stride, pooling.padding)

    def forward(self, x):
        return self.unpooling.forward(x, self.pooling.indices, self.pooling.input_size)
