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

def lua_recursive_model(module,seq):
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.','')
            m = m._obj

        if name == 'SpatialConvolution':
            if not hasattr(m,'groups'): m.groups=1
            n = nn.Conv2d(m.nInputPlane,m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),1,m.groups,bias=(m.bias is not None))
            copy_param(m,n)
            add_submodule(seq,n)
        elif name == 'SpatialBatchNormalization':
            n = nn.BatchNorm2d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m,n)
            add_submodule(seq,n)
        elif name == 'ReLU':
            n = nn.ReLU()
            add_submodule(seq,n)
        elif name == 'SpatialMaxPooling':
            n = nn.MaxPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(seq,n)
        elif name == 'SpatialAveragePooling':
            n = nn.AvgPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(seq,n)
        elif name == 'SpatialUpSamplingNearest':
            n = nn.UpsamplingNearest2d(scale_factor=m.scale_factor)
            add_submodule(seq,n)
        elif name == 'View':
            n1 = LambdaMap(lambda x,s=m.size: x.view(s))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name == 'Linear':
            # Linear in pytorch only accept 2D input
            n1 = LambdaMap(lambda x: x.view(1,-1) if 1==len(x.size()) else x )
            n2 = LambdaReduce(lambda x: x)
            n3 = nn.Linear(m.weight.size(1),m.weight.size(0),bias=(m.bias is not None))
            copy_param(m,n3)
            add_submodule(seq,n1,n2,n3)
        elif name == 'Dropout':
            m.inplace = False
            n = nn.Dropout(m.p)
            add_submodule(seq,n)
        elif name == 'SoftMax':
            n = nn.Softmax()
            add_submodule(seq,n)
        elif name == 'Identity':
            n1 = LambdaMap(lambda x: x) # do nothing
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name == 'SpatialFullConvolution':
            n = nn.ConvTranspose2d(m.nInputPlane,m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH))
            add_submodule(seq,n)
        elif name == 'SpatialReplicationPadding':
            n = nn.ReplicationPad2d((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
            add_submodule(seq,n)
        elif name == 'SpatialReflectionPadding':
            n = nn.ReflectionPad2d((m.pad_l,m.pad_r,m.pad_t,m.pad_b))
            add_submodule(seq,n)
        elif name == 'Copy':
            n1 = LambdaMap(lambda x: x) # do nothing
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name == 'Narrow':
            n1 = LambdaMap(lambda x,a=(m.dimension,m.index,m.length): x.narrow(*a))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name == 'SpatialCrossMapLRN':
            lrn = torch.legacy.nn.SpatialCrossMapLRN(m.size,m.alpha,m.beta,m.k)
            n1 = LambdaMap(lambda x,lrn=lrn: Variable(lrn.forward(x.data)))
            n2 = LambdaReduce(lambda x: x)
            add_submodule(seq,n1,n2)
        elif name == 'Sequential':
            n = nn.Sequential()
            lua_recursive_model(m,n)
            add_submodule(seq,n)
        elif name == 'ConcatTable': # output is list
            n = LambdaMap(lambda x: x)
            lua_recursive_model(m,n)
            add_submodule(seq,n)
        elif name == 'CAddTable': # input is list
            n = LambdaReduce(lambda x,y: x+y)
            add_submodule(seq,n)
        elif name == 'Concat':
            dim = m.dimension
            n = LambdaReduce(lambda x,y,dim=dim: torch.cat((x,y),dim))
            lua_recursive_model(m,n)
            add_submodule(seq,n)
        elif name == 'TorchObject':
            print('Not Implement',name,real._typename)
        else:
            print('Not Implement',name)


def lua_recursive_source(module):
    s = []
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.','')
            m = m._obj

        if name == 'SpatialConvolution':
            if not hasattr(m,'groups'): m.groups=1
            s += ['nn.Conv2d({},{},{},{},{},{},{},bias={}),#Conv2d'.format(m.nInputPlane,
                m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),1,m.groups,m.bias is not None)]
        elif name == 'SpatialBatchNormalization':
            s += ['nn.BatchNorm2d({},{},{},{}),#BatchNorm2d'.format(m.running_mean.size(0), m.eps, m.momentum, m.affine)]
        elif name == 'ReLU':
            s += ['nn.ReLU()']
        elif name == 'SpatialMaxPooling':
            s += ['nn.MaxPool2d({},{},{},ceil_mode={}),#MaxPool2d'.format((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),m.ceil_mode)]
        elif name == 'SpatialAveragePooling':
            s += ['nn.AvgPool2d({},{},{},ceil_mode={}),#AvgPool2d'.format((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),m.ceil_mode)]
        elif name == 'SpatialUpSamplingNearest':
            s += ['nn.UpsamplingNearest2d(scale_factor={})'.format(m.scale_factor)]
        elif name == 'View':
            s += ['LambdaMap(lambda x,s={}: x.view(s)), # View'.format(m.size)]
            s += ['LambdaReduce(lambda x: x)']
        elif name == 'Linear':
            s += ['LambdaMap(lambda x: x.view(1,-1) if 1==len(x.size()) else x ), # Linear hack']
            s += ['LambdaReduce(lambda x: x)']
            s += ['nn.Linear({},{},bias={}),#Linear'.format(m.weight.size(1),m.weight.size(0),(m.bias is not None))]
        elif name == 'Dropout':
            s += ['nn.Dropout({})'.format(m.p)]
        elif name == 'SoftMax':
            s += ['nn.Softmax()']
        elif name == 'Identity':
            s += ['LambdaMap(lambda x: x), # Identity']
            s += ['LambdaReduce(lambda x: x)']
        elif name == 'SpatialFullConvolution':
            s += ['nn.ConvTranspose2d({},{},{},{},{})'.format(m.nInputPlane,
                m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH))]
        elif name == 'SpatialReplicationPadding':
            s += ['nn.ReplicationPad2d({})'.format((m.pad_l,m.pad_r,m.pad_t,m.pad_b))]
        elif name == 'SpatialReflectionPadding':
            s += ['nn.ReflectionPad2d({})'.format((m.pad_l,m.pad_r,m.pad_t,m.pad_b))]
        elif name == 'Copy':
            s += ['LambdaMap(lambda x: x), # Copy']
            s += ['LambdaReduce(lambda x: x)']
        elif name == 'Narrow':
            s += ['LambdaMap(lambda x,a={}: x.narrow(*a))'.format((m.dimension,m.index,m.length))]
            s += ['LambdaReduce(lambda x: x)']
        elif name == 'SpatialCrossMapLRN':
            lrn = 'torch.legacy.nn.SpatialCrossMapLRN(*{})'.format((m.size,m.alpha,m.beta,m.k))
            s += ['LambdaMap(lambda x,lrn={}: Variable(lrn.forward(x.data)))'.format(lrn)]
            s += ['LambdaReduce(lambda x: x)']

        elif name == 'Sequential':
            s += ['nn.Sequential( # Sequential']
            s += lua_recursive_source(m)
            s += [')']
        elif name == 'ConcatTable':
            s += ['LambdaMap(lambda x: x, # ConcatTable']
            s += lua_recursive_source(m)
            s += [')']
        elif name == 'CAddTable':
            s += ['LambdaReduce(lambda x,y: x+y), # CAddTable']
        elif name == 'Concat':
            dim = m.dimension
            s += ['LambdaReduce(lambda x,y,dim={}: torch.cat((x,y),dim), # Concat'.format(m.dimension)]
            s += lua_recursive_source(m)
            s += [')']
        else:
            s += '# ' + name + ' Not Implement,\n'
    s = map(lambda x: '\t{}'.format(x),s)
    return s

def simplify_source(s):
    s = map(lambda x: x.replace(',(1, 1),(0, 0),1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace('),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',1e-05,0.1,True),#BatchNorm2d',')'),s)
    s = map(lambda x: x.replace('),#BatchNorm2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),ceil_mode=False),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace(',ceil_mode=False),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace('),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),ceil_mode=False),#AvgPool2d',')'),s)
    s = map(lambda x: x.replace(',ceil_mode=False),#AvgPool2d',')'),s)
    s = map(lambda x: x.replace(',bias=True),#Linear',')'),s)
    s = map(lambda x: x.replace('),#Linear',')'),s)
    
    s = map(lambda x: '{},\n'.format(x),s)
    s = map(lambda x: x[1:],s)
    s = reduce(lambda x,y: x+y, s)
    return s

def torch_to_pytorch(t7_filename,outputname=None):
    model = load_lua(t7_filename,unknown_classes=True)
    if type(model).__name__=='hashable_uniq_dict': model=model.model
    model.gradInput = None
    slist = lua_recursive_source(torch.legacy.nn.Sequential().add(model))
    s = simplify_source(slist)
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
    lua_recursive_model(model,n)
    torch.save(n.state_dict(),outputname+'.pth')


parser = argparse.ArgumentParser(description='Convert torch t7 model to pytorch')
parser.add_argument('--model','-m', type=str, required=True,
                    help='torch model file in t7 format')
parser.add_argument('--output', '-o', type=int, default=None,
                    help='output file name prefix, xxx.py xxx.pth')
args = parser.parse_args()

torch_to_pytorch(args.model,args.output)
