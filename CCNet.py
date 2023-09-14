'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention,BertSelfOutput
from transformers import BertPreTrainedModel
from torch.nn import Softmax
import math
import copy
import pdb

class Inference(nn.Module):
    def __init__(self, config, num_layers=1):
        super().__init__()
        layer_list = []
        for i in range(num_layers):
            ccnet = CrissCrossAttention_layer111(config, 4)
            layer_list.append(ccnet)
        self.Layers = nn.ModuleList(layer_list)  

        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

    def forward(self, input,entity_pair_matrix, entity_pair_masks):
        b, n_e, _, d = input.size()
        node_index = (entity_pair_masks > 0)

        for i, layer_module in enumerate(self.Layers):
            input,entity_pair_matrix,hidden_states = layer_module(input, entity_pair_matrix, entity_pair_masks)  

        Input = self.dense(torch.cat((entity_pair_matrix, input), dim=-1))
        return Input

class CCA_net22(BertPreTrainedModel):
    def __init__(self, config,num_layers=1,num_attention_heads=12,return_intermediate=True):
        super().__init__(config)
        self.config = copy.deepcopy(config)
        self.config.hidden_size = 512
        self.return_intermediate = return_intermediate
        self.hidden_size = self.config.hidden_size

        self.relation_rep1 = nn.parameter.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.config.num_labels, self.config.hidden_size)).float())
        self.relation_layer = nn.Linear(self.config.hidden_size,self.config.num_labels)
        self.Online_inference = Inference(self.config,num_layers)

        self.Reduce_dense = nn.Sequential(
            nn.Linear(config.hidden_size, self.config.hidden_size,bias=False),
            nn.Dropout(config.hidden_dropout_prob),
            
        )

        self.MLP = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.BatchNorm1d(self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.hidden_size)
            )

    def forward(self, input,entity_pair_matrix, entity_pair_masks,entitys):
        '''
        :param input: [b,e_n,e_n,d]
        :param entity_pair_masks: [b,e_n,e_n]
        :param label_01:   [b,e_n,e_n,97]
        :param entitys: [b,n_e,d]
        :return:
        '''
        b, n_e, _, d = input.size()
        node_index = (entity_pair_masks > 0)

        input1 = self.Online_inference(input,entity_pair_matrix,entity_pair_masks)
        logits = self.relation_layer(input1)

        hidden = torch.zeros_like(input1)
        hidden[node_index] = self.MLP(input1[node_index]).float()
        with torch.no_grad():
            hidden1 = self.Online_inference(entity_pair_matrix,input, entity_pair_masks)
            hidden1.detach_()

        return logits,hidden,hidden1

def logit11(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states1 = hidden_states_new.unsqueeze(2).expand(b, n, n, n, d)  
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states1 = row_states.unsqueeze(1).expand(b, n, n, n, d)     
    
    key_states = torch.cat((col_states1[node_indx], row_states1[node_indx]), dim=-2)  

    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)  
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask1 = row_mask.unsqueeze(1).expand(b, n, n, n)     
        
        mask_self = 1 - torch.eye(n).to(attention_mask).unsqueeze(0).unsqueeze(1)  
        col_mask1 = col_mask1 * mask_self
        
        attention_mask = torch.cat((col_mask1[node_indx], row_mask1[node_indx]), dim=-1)

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states,attention_mask

def logit22(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()

    col_states1 = hidden_states_new.unsqueeze(2).expand(b, n, n, n, d)  
    col_states2 = hidden_states_new.unsqueeze(1).expand(b, n, n, n, d)  
    
    key_states = torch.cat((col_states1[node_indx], col_states2[node_indx]), dim=-2)  
    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)
        col_mask2 = attention_mask.unsqueeze(1).expand(b, n, n, n)
        
        attention_mask = torch.cat((col_mask1[node_indx], col_mask2[node_indx]), dim=-1)

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states, attention_mask

def logit33(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states2 = hidden_states_new.unsqueeze(1).expand(b, n, n, n, d)  
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states2 = row_states.unsqueeze(2).expand(b, n, n, n, d)  
    
    key_states = torch.cat((col_states2[node_indx], row_states2[node_indx]), dim=-2)  

    if attention_mask is not None:
        col_mask2 = attention_mask.unsqueeze(1).expand(b, n, n, n)
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask2 = row_mask.unsqueeze(2).expand(b, n, n, n)

        mask_self = 1 - torch.eye(n).to(attention_mask).unsqueeze(0).unsqueeze(1)  
        row_mask2 = row_mask2 * mask_self
        
        attention_mask = torch.cat((col_mask2[node_indx], row_mask2[node_indx]), dim=-1)

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states, attention_mask

def logit44(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states1 = row_states.unsqueeze(1).expand(b, n, n, n, d)  
    row_states2 = row_states.unsqueeze(2).expand(b, n, n, n, d)  
    
    key_states = torch.cat((row_states2[node_indx], row_states1[node_indx]), dim=-2)  

    if attention_mask is not None:
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask1 = row_mask.unsqueeze(1).expand(b, n, n, n)
        row_mask2 = row_mask.unsqueeze(2).expand(b, n, n, n)

        
        attention_mask = torch.cat((row_mask2[node_indx], row_mask1[node_indx]), dim=-1)
        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states, attention_mask


def logit111(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states1 = hidden_states_new.unsqueeze(2).expand(b, n, n, n, d)  
    
    key_states = col_states1[node_indx]

    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)  
        
        mask_self = 1 - torch.eye(n).to(attention_mask).unsqueeze(0).unsqueeze(1)  
        col_mask1 = col_mask1 * mask_self
        
        attention_mask = col_mask1[node_indx]

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states,attention_mask

def logit222(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states1 = row_states.unsqueeze(1).expand(b, n, n, n, d)     
    
    key_states = row_states1[node_indx]

    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)  
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask1 = row_mask.unsqueeze(1).expand(b, n, n, n)     
        
        row_mask1 = row_mask1.permute(0, 2, 1, 3).contiguous()
        mask_self = 1 - torch.eye(n).to(attention_mask).unsqueeze(0).unsqueeze(1)  
        row_mask1 = row_mask1 * mask_self

        row_mask1 = row_mask1.permute(0, 2, 1, 3).contiguous()

        attention_mask = row_mask1[node_indx]

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states,attention_mask

def logit333(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states2 = row_states.unsqueeze(2).expand(b, n, n, n, d)
    key_states = row_states2[node_indx]

    if attention_mask is not None:
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask2 = row_mask.unsqueeze(2).expand(b, n, n, n)
        
        attention_mask = row_mask2[node_indx]

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states, attention_mask

def logit444(hidden_states_new,node_indx,attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states2 = hidden_states_new.unsqueeze(1).expand(b, n, n, n, d)
    key_states = col_states2[node_indx] 

    if attention_mask is not None:
        col_mask2 = attention_mask.unsqueeze(1).expand(b, n, n, n)
        
        attention_mask = col_mask2[node_indx]

        attention_mask = (1.0 - attention_mask) * -10000.0  
        attention_mask = attention_mask.unsqueeze(1)  

    return key_states, attention_mask

class output_lagyer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor,node_indx):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor[node_indx])

        new_shape=input_tensor.size()[:-1]+(hidden_states.size(-1),)
        hidden_states_new = torch.zeros(new_shape).to(hidden_states)  
        hidden_states_new[node_indx]=hidden_states

        return hidden_states_new

class FFNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.act_fn = nn.ReLU(inplace=True)
        
        self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states,input_tensor,node_indx):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.LayerNorm2(hidden_states + input_tensor[node_indx])

        new_shape=input_tensor.size()[:-1]+(hidden_states.size(-1),)
        hidden_states_new = torch.zeros(new_shape).to(hidden_states)  
        hidden_states_new[node_indx]=hidden_states

        return hidden_states_new


class CCAttention01(nn.Module):
    def __init__(self, config,num_attention_heads):
        super().__init__()
        self.config = config
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.act_fn = nn.ReLU()

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.mid_output = output_lagyer(config)
        self.Output = FFNN(config)

        self.logit_function = {
            0: logit111,
            1: logit222,
            2: logit333,
            3: logit444,
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, x.size(-1)//self.num_attention_heads)
        x = x.view(*new_x_shape)  
        if(len(list(x.size()))==4):
            return x.permute(0, 2, 1, 3)
        if(len(list(x.size()))==5):
            return x.permute(0, 3, 1, 2, 4)
        if (len(list(x.size())) == 3):
            return x

    def forward(self, Input,hidden_states,attention_mask):
        '''
        :param hidden_states: [b,e_n,e_n,768]
        :param attention_mask: [b,e_n,e_n]
        :param output_attentions:
        :return:
        '''
        node_indx = (attention_mask >0)  
        
        new_hidden_states = Input[node_indx]
        new_hidden_states_Q = hidden_states[node_indx]

        key_layer = self.transpose_for_scores(self.key(new_hidden_states))  
        value_layer = self.transpose_for_scores(self.value(new_hidden_states))  
        query_layer = self.transpose_for_scores(self.query(new_hidden_states_Q))  

        new_hidden_states = torch.zeros(hidden_states.size()[:-1]+(key_layer.size(-1),)).to(key_layer)
        
        key_layers, value_layers, attention_masks = [], [], []
        for i in range(self.num_attention_heads):
            j=i%4
            new_hidden_states[node_indx] = key_layer[:, i]  
            
            key_state, attention_mask1 = self.logit_function[j](new_hidden_states, node_indx, attention_mask)
            key_state = torch.cat((key_layer[:, i].unsqueeze(1),key_state),dim=1) 

            new_hidden_states[node_indx] = value_layer[:, i]    
            value_state, _ = self.logit_function[j](new_hidden_states, node_indx)
            value_state = torch.cat((value_layer[:, i].unsqueeze(1), value_state), dim=1)  

            L = attention_mask1.size(0)
            attention_mask2 = torch.ones((L,1,1)).to(attention_mask1) 
            attention_mask1 = torch.cat((attention_mask2, attention_mask1), dim=-1)  

            key_layers.append(key_state)
            value_layers.append(value_state)
            attention_masks.append(attention_mask1)

        key_layer = torch.stack(key_layers, dim=1)  
        value_layer = torch.stack(value_layers, dim=1)  
        attention_mask = torch.cat(attention_masks, dim=1)  

        query_layer = query_layer.unsqueeze(2) 
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(key_layer.size(-1))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(2)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size(-1)*context_layer.size(-2),)
        context_layer = context_layer.view(*new_context_layer_shape)

        mid_output = self.mid_output(context_layer, hidden_states, node_indx)
        Output = self.Output(mid_output[node_indx], mid_output, node_indx)

        return Output


class CCAttention(nn.Module):
    def __init__(self, config,num_attention_heads):
        super().__init__()
        self.config = config
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.act_fn = nn.ReLU()

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        
        self.mid_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.logit_function = {
            0: logit111,
            1: logit222,
            2: logit333,
            3: logit444,
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, x.size(-1)//self.num_attention_heads)
        x = x.view(*new_x_shape)  
        if(len(list(x.size()))==4):
            return x.permute(0, 2, 1, 3)
        if(len(list(x.size()))==5):
            return x.permute(0, 3, 1, 2, 4)
        if (len(list(x.size())) == 3):
            return x

    def forward(self, Input,hidden_states,attention_mask):
        '''
        :param hidden_states: [b,e_n,e_n,768]
        :param attention_mask: [b,e_n,e_n]
        :param output_attentions:
        :return:
        '''
        node_indx = (attention_mask >0)  
        
        new_hidden_states = Input[node_indx]
        new_hidden_states_Q = hidden_states[node_indx]

        key_layer = self.transpose_for_scores(self.key(new_hidden_states))  
        value_layer = self.transpose_for_scores(self.value(new_hidden_states))  
        query_layer = self.transpose_for_scores(self.query(new_hidden_states_Q))  

        new_hidden_states = torch.zeros(hidden_states.size()[:-1]+(key_layer.size(-1),)).to(key_layer)
        
        key_layers, value_layers, attention_masks = [], [], []
        for i in range(self.num_attention_heads):
            j=i%4
            new_hidden_states[node_indx] = key_layer[:, i]  
            
            key_state, attention_mask1 = self.logit_function[j](new_hidden_states, node_indx, attention_mask)
            key_state = torch.cat((key_layer[:, i].unsqueeze(1),key_state),dim=1) 

            new_hidden_states[node_indx] = value_layer[:, i]    
            value_state, _ = self.logit_function[j](new_hidden_states, node_indx)
            value_state = torch.cat((value_layer[:, i].unsqueeze(1), value_state), dim=1)  

            L = attention_mask1.size(0)
            attention_mask2 = torch.ones((L,1,1)).to(attention_mask1) 
            attention_mask1 = torch.cat((attention_mask2, attention_mask1), dim=-1)  

            key_layers.append(key_state)
            value_layers.append(value_state)
            attention_masks.append(attention_mask1)

        key_layer = torch.stack(key_layers, dim=1)  
        value_layer = torch.stack(value_layers, dim=1)  
        attention_mask = torch.cat(attention_masks, dim=1)  

        query_layer = query_layer.unsqueeze(2) 
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(key_layer.size(-1))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(2)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size(-1)*context_layer.size(-2),)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.mid_output(context_layer)

        return context_layer

class CrissCrossAttention_layer111(nn.Module):
    def __init__(self, config,num_attention_heads):
        super().__init__()
        self.config = config
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.Self_Attention = CCAttention(config,num_attention_heads)
        self.Cross_Attention = CCAttention(config,num_attention_heads)

        self.gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Sigmoid()
        )

        self.mid_output = FFNN(config)
        self.mid_output1 = FFNN(config)

    def forward(self,Input,hidden_states,attention_mask):
        '''
        :param hidden_states: [b,e_n,e_n,768]
        :param attention_mask: [b,e_n,e_n]
        :param output_attentions:
        :return:
        '''
        node_indx = (attention_mask > 0)  
        hidden_states1 = self.combine(torch.cat((hidden_states,Input),dim=-1))

        hidden_states2 = self.Self_Attention(hidden_states1,hidden_states,attention_mask)
        Input1 = self.Cross_Attention(hidden_states1,Input,attention_mask)

        gate = self.gate(torch.cat((hidden_states2,Input1),dim=-1))
        hidden_states3 = gate * hidden_states2 + (1 - gate) * Input1

        hidden_states = self.mid_output(hidden_states3,hidden_states, node_indx)
        Input = self.mid_output1(hidden_states3, Input, node_indx)

        return Input,hidden_states,hidden_states1



