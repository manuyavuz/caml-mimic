"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

from learn import models
import tcn
from constants import *
import datasets
import persistence
import numpy as np

# class MyDataParallel(nn.DataParallel):
#     def __getattr__(self, name):
#         module = self.__dict__.get('module')
#         if name == 'module':
#             return module
#         else:            
#             return getattr(module, name)

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    Y = len(dicts['ind2c'])
    if args.model == "rnn":
        model = models.VanillaRNN(Y, args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout, embedding=args.embedding)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout, embedding=args.embedding)
    elif args.model == "tcn":
        filter_size = int(args.filter_size)
        model = models.VanillaTCN(Y, args.embed_file, filter_size, args.num_filter_maps, args.tcn_layers, args.gpu, dicts, args.embed_size, 
                                   args.dropout, args.tcn_dilations, args.attention)
    if args.gpu:
        model = nn.DataParallel(model)
    if args.test_model:
        sd = torch.load(args.test_model)
        try:
            model.load_state_dict(sd)
        except RuntimeError as error:
            model = model.module
            model.load_state_dict(sd)
            model = nn.DataParallel(model)
    if args.gpu:
        model.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            for i in range(torch.cuda.device_count()):
                torch.tensor(1).cuda(i)
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers, args.tcn_layers, 
                  args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "rnn_dim", "cell_type", "rnn_layers", "tcn_layers", "lmbda", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

