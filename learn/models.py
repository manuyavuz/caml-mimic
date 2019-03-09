"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import numpy as np

from math import floor
import random
import sys
import time

from constants import *
from dataproc import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=128, embedding='default', tune_embeddings=False):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda
        self.embedding = embedding
        self.tune_embeddings = tune_embeddings

        #make embedding layer
        if self.embedding == 'elmo':
            options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
            weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"

            w2ind = dicts['w2ind']
            vocab = list(sorted(w2ind.keys(), key=w2ind.get))
            #add UNK and PAD
            vocab.insert(0, 'PAD')
            vocab.append('UNK')
            self.embed = Elmo(options_file, weight_file, 2, dropout=0, vocab_to_cache=vocab, requires_grad=False)

        elif self.embedding == 'bert':
            bert_folder = "/data/corpora/mimic/experiments/pytorch-pretrained-BERT/pubmed_pmc_470k"
            self.embed = BertModel.from_pretrained(bert_folder)
        else:
            if embed_file:
                print("loading pretrained embeddings...")
                W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

                self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
                if self.tune_embeddings:
                    for params in self.embed.parameters():
                        params.requires_grad = False
                self.embed.weight.data = W.clone()
            else:
                #add 2 to include UNK and PAD
                vocab_size = len(dicts['ind2w'])
                self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)
            

    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            loss = loss + diff
        return loss

    def get_embedding(self, input):
        if self.embedding == 'elmo':
            embedding = []
            chunk_size = 256
            for i in range(0, max(1,input.size(1)//chunk_size)):
                chunk_input = input[:, i*chunk_size:chunk_size+i*chunk_size]
                i_embedding = self.embed(chunk_input, chunk_input)['elmo_representations'][-1]
                embedding.append(i_embedding)
            return torch.cat(embedding, 1)
        elif self.embedding == 'bert':
            embedding = []
            chunk_size = 128
            for i in range(0, max(1,input.size(1)//chunk_size)):
                i_embedding, _ = self.embed(input[:, i*chunk_size:chunk_size+i*chunk_size], output_all_encoded_layers=False)
                embedding.append(i_embedding)
            return torch.cat(embedding, 1)
        else:
            return self.embed(input)

    def embed_descriptions(self, desc_data, gpu):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate over the batch
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                if gpu:
                    lt = Variable(torch.cuda.LongTensor(inst))
                else:
                    lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs


class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=64, dropout=0.5, embedding='default', tune_embeddings=False):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size, embedding=embedding, tune_embeddings=tune_embeddings)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)
        
        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)
        
    def forward(self, x, target, desc_data=None, get_attention=True):
        #get embeddings and apply dropout
        x = self.get_embedding(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if desc_data is not None:
            #run descriptions through description module
            b_batch = self.f(desc_data, self.gpu)
            #get l2 similarity loss
            diffs = self._compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target, diffs)
        return yhat, loss, alpha


class Transformer(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=64, dropout=0.5, embedding='default'):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size, embedding=embedding)
        #initialize conv layer as in 2.1
        # self.ff1 = nn.Linear(num_filter_maps, Y)
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        x = self.get_embedding(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)
        #linear output
        x = self.fc(x)
        #final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full

class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=64, dropout=0.5, embedding='default', tune_embeddings=False):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size, embedding=embedding, tune_embeddings=tune_embeddings) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        x = self.get_embedding(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)
        #linear output
        x = self.fc(x)
        #final sigmoid to get predictions
        yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full

class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=64, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = self.final(last_hidden)
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim/self.num_directions)).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim/self.num_directions)).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions)))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

        
        
####### manuyavuz ######

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilations=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels): 
            dilation_size = 2 ** i if dilations is None else dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class VanillaTCN(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, num_layers, gpu=True, dicts=None, embed_size=64, dropout=0.5, dilations=None, use_attention=False):
        super(VanillaTCN, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size)
        self.embed_drop = nn.Dropout(p=0.2)
        self.use_attention = use_attention
        num_channels = [num_filter_maps] * num_layers
        # dilations = [3] * num_layers
        self.tcn = TemporalConvNet(self.embed_size, num_channels, kernel_size=kernel_size, dropout=dropout, dilations=dilations)
#         xavier_uniform(self.tcn.weight)

        #context vectors for computing attention as in 2.2

        if self.use_attention:
            self.U = nn.Linear(num_filter_maps, Y)
            xavier_uniform(self.U.weight)
        self.linear = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.linear.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.get_embedding(x)
        # x = self.embed_drop(x)
        x = x.transpose(1, 2)

        tc = self.tcn(x)  # input should have dimension (N, C, L)

        if self.use_attention:
            #apply convolution and nonlinearity (tanh)
            x = F.tanh(tc.transpose(1,2))
            #apply attention
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
            #document representations are weighted sums using the attention. Can compute all at once as a matmul
            m = alpha.matmul(x)
            #final layer classification
            yhat = self.linear.weight.mul(m).sum(dim=2).add(self.linear.bias)
        else:
            x = F.max_pool1d(F.tanh(tc), kernel_size=tc.size()[2])
            x = x.squeeze(dim=2)
            yhat = self.linear(x)
            
        loss = self._get_loss(yhat, target)
#         return F.log_softmax(o, dim=1), loss
        return yhat, loss, None

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full


# class HierarchicalAttentionTCN(BaseModel):

#     def __init__(self, Y, embed_file, kernel_size, num_filter_maps, num_layers, gpu=True, dicts=None, embed_size=64, dropout=0.5, dilations=None):
#         super(HierarchicalAttentionTCN, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size)
#         self.embed_drop = nn.Dropout(p=0.2)
#         num_channels = [num_filter_maps] * num_layers
#         # dilations = [3] * num_layers
#         self.tcn = TemporalConvNet(self.embed_size, num_channels, kernel_size=kernel_size, dropout=dropout, dilations=dilations)
# #         xavier_uniform(self.tcn.weight)

#         #context vectors for computing attention as in 2.2
#         self.U = nn.ModuleList()
#         self.linear = nn.ModuleList()
#         for y in Y:
#             if self.use_attention:
#                 U = nn.Linear(num_filter_maps, y)
#                 xavier_uniform(U.weight)
#                 self.U.append(U)
#             linear = nn.Linear(num_filter_maps, y)
#             xavier_uniform(linear.weight)
#             self.linear.append(linear)

#     def forward(self, x, target, desc_data=None, get_attention=False):
#         #embed
#         x = self.get_embedding(x)
#         # x = self.embed_drop(x)
#         x = x.transpose(1, 2)

#         tc = self.tcn(x)  # input should have dimension (N, C, L)

#         if self.use_attention:
#             #apply convolution and nonlinearity (tanh)
#             x = F.tanh(tc.transpose(1,2))
#             for y in Y:
#                 # U = self.U[]
#                 #apply attention
#                 alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
#                 #document representations are weighted sums using the attention. Can compute all at once as a matmul
#                 m = alpha.matmul(x)
#                 #final layer classification
#                 yhat = self.linear.weight.mul(m).sum(dim=2).add(self.linear.bias)
#         else:
#             x = F.max_pool1d(F.tanh(tc), kernel_size=tc.size()[2])
#             x = x.squeeze(dim=2)
#             yhat = self.linear(x)
            
#         loss = self._get_loss(yhat, target)
# #         return F.log_softmax(o, dim=1), loss
#         return yhat, loss, None

#     def construct_attention(self, argmax, num_windows):
#         attn_batches = []
#         for argmax_i in argmax:
#             attns = []
#             for i in range(num_windows):
#                 #generate mask to select indices of conv features where max was i
#                 mask = (argmax_i == i).repeat(1,self.Y).t()
#                 #apply mask to every label's weight vector and take the sum to get the 'attention' score
#                 weights = self.fc.weight[mask].view(-1,self.Y)
#                 if len(weights.size()) > 0:
#                     window_attns = weights.sum(dim=0)
#                     attns.append(window_attns)
#                 else:
#                     #this window was never a max
#                     attns.append(Variable(torch.zeros(self.Y)).cuda())
#             #combine
#             attn = torch.stack(attns)
#             attn_batches.append(attn)
#         attn_full = torch.stack(attn_batches)
#         #put it in the right form for passing to interpret
#         attn_full = attn_full.transpose(1,2)
#         return attn_full

