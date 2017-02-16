from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua

import numpy as np
import os
import math
import cv2

class LambdaLayer(nn.Module):
    def __init__(self, *args):
        super(LambdaLayer, self).__init__()
        self.lambda_func = None
        for m in args:
            if type(m).__name__ == 'function': self.lambda_func = m
            else: self.add_module(str(len(self._modules)), m)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        if 0==len(output):
            if type(input) is list: output = input
            else: output = [input]
        return output

class LambdaMap(LambdaLayer):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return map(self.lambda_func,self.forward_prepare(input))

class LambdaReduce(LambdaLayer):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func,self.forward_prepare(input))


def copy_param(m,n):
    if m.weight is not None: n.weight.data.copy_(m.weight)
    if m.bias is not None: n.bias.data.copy_(m.bias)
    if hasattr(n,'running_mean'): n.running_mean.copy_(m.running_mean)
    if hasattr(n,'running_var'): n.running_var.copy_(m.running_var)

def add_submodule(seq, *args):
    for n in args:
        seq.add_module(str(len(seq._modules)),n)

def lua_recursive(module,seq):
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name in ['TorchObject']:
            name = m._typename.replace('cudnn.','')
            m = m._obj

        if name in ['SpatialConvolution']:
            if not hasattr(m,'groups'): m.groups=1
            n = nn.Conv2d(m.nInputPlane,m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),1,m.groups,bias=(m.bias is not None))
            copy_param(m,n)
            add_submodule(seq,n)
        elif name in ['SpatialBatchNormalization']:
            n = nn.BatchNorm2d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m,n)
            add_submodule(seq,n)
        elif name in ['ReLU']:
            n = nn.ReLU()
            add_submodule(seq,n)
        elif name in ['SpatialMaxPooling']:
            n = nn.MaxPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(seq,n)
        elif name in ['SpatialAveragePooling']:
            n = nn.AvgPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(seq,n)
        elif name in ['SpatialUpSamplingNearest']:
            n = nn.UpsamplingNearest2d(scale_factor=m.scale_factor)
            add_submodule(seq,n)
        elif name in ['View']:
            n1 = LambdaMap(lambda x,s=m.size: x.view(s))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name in ['Linear']:
            # Linear in pytorch only accept 2D input
            n1 = LambdaMap(lambda x: x.view(1,-1) if 1==len(x.size()) else x )
            n2 = LambdaReduce(lambda x: x)
            n3 = nn.Linear(m.weight.size(1),m.weight.size(0),bias=(m.bias is not None))
            copy_param(m,n3)
            add_submodule(seq,n1,n2,n3)
        elif name in ['Dropout']:
            m.inplace = False
            n = nn.Dropout(m.p)
            add_submodule(seq,n)
        elif name in ['SoftMax']:
            n = nn.Softmax()
            add_submodule(seq,n)
        elif name in ['Identity']:
            n1 = LambdaMap(lambda x: x) # do nothing
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name in ['SpatialFullConvolution']:
            n = nn.ConvTranspose2d(m.nInputPlane,m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH))
            add_submodule(seq,n)
        elif name in ['SpatialReplicationPadding']:
            n = nn.ReplicationPad2d((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
            add_submodule(seq,n)
        elif name in ['SpatialReflectionPadding']:
            n = nn.ReflectionPad2d((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
            add_submodule(seq,n)
        elif name in ['Copy']:
            n1 = LambdaMap(lambda x: x) # do nothing
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name in ['Narrow']:
            n1 = LambdaMap(lambda x,a=(m.dimension,m.index,m.length): x.narrow(*a))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name in ['SpatialCrossMapLRN']:
            lrn = torch.legacy.nn.SpatialCrossMapLRN(m.size,m.alpha,m.beta,m.k)
            n1 = LambdaMap(lambda x,lrn=lrn: Variable(lrn.forward(x.data)))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name in ['Sequential']:
            n = nn.Sequential()
            lua_recursive(m,n)
            add_submodule(seq,n)
        elif name in ['ConcatTable']: # output is list
            n = LambdaMap(lambda x: x)
            lua_recursive(m,n)
            add_submodule(seq,n)
        elif name in ['CAddTable']: # input is list
            n = LambdaReduce(lambda x,y: x+y)
            add_submodule(seq,n)
        elif name in ['Concat']:
            dim = m.dimension
            n = LambdaReduce(lambda x,y,dim=dim: torch.cat((x,y),dim))
            lua_recursive(m,n)
            add_submodule(seq,n)
        elif name in ['TorchObject']:
            print('Not Implement',name,real._typename)
        else:
            print('Not Implement',name)


def lua_pytorch(module,t=0):
    s = ''
    for m in module.modules:
        s += '\t'*t
        name = type(m).__name__
        real = m
        if name in ['TorchObject']:
            name = m._typename.replace('cudnn.','')
            m = m._obj

        if name in ['SpatialConvolution']:
            if not hasattr(m,'groups'): m.groups=1
            s += 'nn.Conv2d({},{},{},{},{},{},{},bias={}),\n'.format(m.nInputPlane,
                m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),1,m.groups,m.bias is not None)
        elif name in ['SpatialBatchNormalization']:
            s += 'nn.BatchNorm2d({},{},{},{}),\n'.format(m.running_mean.size(0), m.eps, m.momentum, m.affine)
        elif name in ['ReLU']:
            s += 'nn.ReLU(),\n'
        elif name in ['SpatialMaxPooling']:
            s += 'nn.MaxPool2d({},{},{},ceil_mode={}),\n'.format((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),m.ceil_mode)
        elif name in ['SpatialAveragePooling']:
            s += 'nn.AvgPool2d({},{},{},ceil_mode={}),\n'.format((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),m.ceil_mode)
        elif name in ['SpatialUpSamplingNearest']:
            s += 'nn.UpsamplingNearest2d(scale_factor={}),\n'.format(m.scale_factor)
        elif name in ['View']:
            s += 'LambdaMap(lambda x,s={}: x.view(s)), # View\n'.format(m.size)
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'
        elif name in ['Linear']:
            s += 'LambdaMap(lambda x: x.view(1,-1) if 1==len(x.size()) else x ), # Linear hack\n'
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'
            s += '\t'*t + 'nn.Linear({},{},bias={}),\n'.format(m.weight.size(1),m.weight.size(0),(m.bias is not None))
        elif name in ['Dropout']:
            s += 'nn.Dropout({}),\n'.format(m.p)
        elif name in ['SoftMax']:
            s += 'nn.Softmax(),\n'
        elif name in ['Identity']:
            s += 'LambdaMap(lambda x: x), # Identity\n'
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'
        elif name in ['SpatialFullConvolution']:
            s += 'nn.ConvTranspose2d({},{},{},{},{}),\n'.format(m.nInputPlane,
                m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH))
        elif name in ['SpatialReplicationPadding']:
            s += 'nn.ReplicationPad2d({})'.format((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
        elif name in ['SpatialReflectionPadding']:
            s += 'nn.ReflectionPad2d({})'.format((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
        elif name in ['Copy']:
            s += 'LambdaMap(lambda x: x), # Copy\n'
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'
        elif name in ['Narrow']:
            s += 'LambdaMap(lambda x,a={}: x.narrow(*a)),\n'.format((m.dimension,m.index,m.length))
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'
        elif name in ['SpatialCrossMapLRN']:
            lrn = 'torch.legacy.nn.SpatialCrossMapLRN(*{})'.format((m.size,m.alpha,m.beta,m.k))
            s += 'LambdaMap(lambda x,lrn={}: Variable(lrn.forward(x.data))),\n'.format(lrn)
            s += '\t'*t + 'LambdaReduce(lambda x: x),\n'

        elif name in ['Sequential']:
            s += 'nn.Sequential(\n'
            s += lua_pytorch(m,t+1)
            s += '\t'*t + '),\n'
        elif name in ['ConcatTable']:
            s += 'LambdaMap(lambda x: x, # ConcatTable\n'
            s += lua_pytorch(m,t+1)
            s += '\t'*t + '),\n'
        elif name in ['CAddTable']:
            s += 'LambdaReduce(lambda x,y: x+y), # CAddTable\n'
        elif name in ['Concat']:
            dim = m.dimension
            s += 'LambdaReduce(lambda x,y,dim={}: torch.cat((x,y),dim), # Concat\n'.format(m.dimension)
            s += lua_pytorch(m,t+1)
            s += '\t'*t + '),\n'
        else:
            s += '# ' + name + ' Not Implement,\n'
    return s


def torch_to_pytorch(t7_filename,outputname=None):
    model = load_lua(t7_filename,unknown_classes=True)
    if type(model).__name__=='hashable_uniq_dict': model=model.model
    model.gradInput = None
    s = lua_pytorch(torch.legacy.nn.Sequential().add(model))
    header = '''
import torch
import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self, *args):
        super(LambdaLayer, self).__init__()
        self.lambda_func = None
        for m in args:
            if type(m).__name__ == 'function': self.lambda_func = m
            else: self.add_module(str(len(self._modules)), m)
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        if 0==len(output):
            if type(input) is list: output = input
            else: output = [input]
        return output

class LambdaMap(LambdaLayer):
    def forward(self, input):
        return map(self.lambda_func,self.forward_prepare(input))

class LambdaReduce(LambdaLayer):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
'''
    varname = t7_filename.replace('.t7','').replace('.','_').replace('-','_')
    s = '{}\n\n{} = {}'.format(header,varname,s[:-2])

    if outputname is None: outputname=varname
    with open(outputname+'.py', "w") as pyfile:
        pyfile.write(s)

    n = nn.Sequential()
    lua_recursive(model,n)
    torch.save(n.state_dict(),outputname+'.pth')


parser = argparse.ArgumentParser(description='Convert torch t7 model to pytorch')
parser.add_argument('--model','-m', type=str, required=True,
                    help='torch model file in t7 format')
parser.add_argument('--output', '-o', type=int, default=None,
                    help='output file name prefix, xxx.py xxx.pth')
args = parser.parse_args()

torch_to_pytorch(args.model,args.output)
