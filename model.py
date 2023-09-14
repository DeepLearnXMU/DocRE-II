import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss,compute_kl_loss
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
import math
import time
import copy
import pdb
import torch.nn.functional as F
from CCNet import CCA_net22
from BERT import BertModel
import ujson as json
import random

def contrast_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class DocREModel(nn.Module):
    def __init__(self, args,config,emb_size=768, block_size=64, num_labels=-1,base_train=True):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.num_labels=num_labels
        self.train_base = config.train_base

        config.output_hidden_states = True 
        self.encoder = BertModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        self.loss_fnt = ATLoss()
        self.hidden_size = config.hidden_size

        self.head_pair = nn.Sequential(
            nn.Linear(config.hidden_size*3, emb_size),
            nn.Tanh(),
        )
        self.tail_pair = nn.Sequential(
            nn.Linear(config.hidden_size*3, emb_size),
            nn.Tanh(),
        )
        self.entity_pair_extractor = nn.Sequential(
            nn.Linear(emb_size * block_size, emb_size),
            nn.ReLU(inplace=True),
        )
        self.Relation_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(self.config.hidden_size, self.config.num_labels,bias=False),
        )
        if (not self.train_base):
            self.CCA = CCA_net22(self.config, num_layers=config.decoder_layers)

        self.emb_size = emb_size
        self.block_size = block_size
        self.Step =0
        self.Total =0
        

    def model_initial(self):
        for model in [self.Relation_classifier,self.head_pair,self.tail_pair]:
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)

        for model in [self.entity_pair_extractor]:
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def Encode(self, input_ids, attention_mask):
        sequence_output, attention,hidden_states= process_long_input(self.encoder, input_ids, attention_mask)
        return sequence_output, attention,hidden_states

    def get_mention_rep(self, sequence_output, attention, attention_mask, entity_pos, entity_type, Sentence_index):
        mention = []
        mention_positon = []
        mention_entity_type = []
        mention_entity_id = []
        mention_sentence_id = []

        entity2mention = []
        Entity_attention = []  
        Mention_attention = []  
        Sentence = []

        batch_size, doc_len, _ = sequence_output.size()
        Max_met_num = -1
        Max_sent_num = -1
        for i in range(batch_size):  
            mention.append([sequence_output[i][0]])  
            mention_entity_type.append([7])  
            mention_entity_id.append([48])  
            mention_sentence_id.append([29])  
            mention_indx = 1

            entity2mention.append([])
            entity_atts = []  
            mention_atts = []  
            Sentence.append([])
            for j, e in enumerate(entity_pos[i]):  
                e_att = []  
                entity2mention[-1].append([])
                for start, end, sentence_id in e:
                    mention[-1].append((sequence_output[i][start + 1] + sequence_output[i][end]) / 2.0)
                    e_att.append((attention[i, :, start + 1] + attention[i, :, end])/2.0)
                    mention_entity_type[-1].append(entity_type[i][j])
                    mention_entity_id[-1].append(j + 1)  
                    mention_sentence_id[-1].append(sentence_id + 1)  
                    entity2mention[-1][-1].append(mention_indx)
                    mention_indx += 1
                mention_atts.extend(e_att)  
                
                if len(e_att) > 1:  
                    e_att = torch.stack(e_att, dim=0).mean(0)  
                else:
                    e_att = e_att[0]
                entity_atts.append(e_att)  

            entity_atts = torch.stack(entity_atts, dim=0)  
            Entity_attention.append(entity_atts)  
            mention_atts = torch.stack(mention_atts, dim=0)  
            Mention_attention.append(mention_atts)  

            if (Max_met_num < mention_indx):
                Max_met_num = mention_indx
            
            for j, (start_sent, end_sent) in enumerate(Sentence_index[i]):
                Sentence[-1].append(torch.mean(sequence_output[i][start_sent:end_sent], dim=0))  
            if len(Sentence_index[i]) > Max_sent_num:
                Max_sent_num = len(Sentence_index[i])

        for i in range(batch_size):  
            origin_len = len(mention[i])
            extence = Max_met_num - origin_len
            for j in range(extence):
                mention[i].append(torch.zeros(768).to(sequence_output.device))
            mention[i] = torch.stack(mention[i], dim=0)  

        mention = torch.stack(mention, dim=0)  

        mention_feature = {
            "mention": mention,
            "entity2mention": entity2mention,  
            "Entity_attention": Entity_attention,  
        }
        return mention_feature

    def create_sample(self,Sample_pro,label_01=None):
        
        Pos_mask = label_01[:, :, :, 1:].sum(-1)
        Pos_mask = (Pos_mask>0).float()    
        P_n = Pos_mask.sum()      
        NUM = 5 - 2 * min(1.0, self.Step / self.Total)
        N_n = int((P_n * 5).item())

        mask = label_01[:, :, :, 0]  
        Neg_mask = torch.zeros_like(mask)
        Shape = mask.size()
        mask = mask.view(1, -1)
        Neg_mask = Neg_mask.view(1, -1)
        index = torch.multinomial(mask, N_n)   
        Neg_mask[0, index] = 1
        Neg_mask = Neg_mask.view(Shape)   

        train_mask = Pos_mask + Neg_mask
        N_mask = Neg_mask - F.dropout(Neg_mask, p=0.2) * 0.8
        train_mask1 = Pos_mask + N_mask

        r = random.uniform(0.2, 0.5)  
        s_n = int(r * P_n+1)      

        mask_p = torch.zeros_like(Pos_mask)
        Shape = mask_p.size()
        mask_p = mask_p.view(1, -1)

        pro_t = (Sample_pro * Pos_mask).view(1, -1)  
        index = torch.multinomial(pro_t, s_n)
        mask_p[0, index] = 1
        mask_p = mask_p.view(Shape)

        
        r = random.uniform(0.05, 0.15)  
        s_n = int(N_n * r + 1)

        mask_n = torch.zeros_like(Neg_mask)
        Shape = mask_n.size()
        mask_n = mask_n.view(1, -1)

        pro_t = (Sample_pro * Neg_mask).view(1, -1)  
        index = torch.multinomial(pro_t, s_n)
        mask_n[0, index] = 1
        mask_n = mask_n.view(Shape)

        Mask_label = mask_p + mask_n

        Label = 1 - label_01  
        Shape = Label.size()
        Label = Label.view(-1, 97)

        index = torch.multinomial(Label, 1)  
        row = torch.tensor([[i] for i in range(index.size(0))]).to(index)

        Label = Label * 0  
        Label[row, index] = 1  
        Label = Label.view(Shape)

        train_mask1 = (train_mask1 + Mask_label > 0).float()
        mask = Mask_label.unsqueeze(-1)
        Label = label_01 * (1 - mask) + Label * mask

        return Label,train_mask,train_mask1,Mask_label


    def Entity_level_predict( self,Entity_feature):
        entity_pair_matrix=Entity_feature["entity_pair_matrix"]
        entity_pair_masks=Entity_feature["entity_pair_masks"]
        hss=Entity_feature["hss"]
        tss=Entity_feature["tss"]
        hts_tensor=Entity_feature["hts_tensor"]
        rss=Entity_feature["rss"]
        label_01 = Entity_feature["label_01"]
        entitys = Entity_feature["entitys"]
        E_num = Entity_feature["E_num"]

        loss, loss1, Error = 0, 0, 0
        Index = (entity_pair_masks > 0)

        logit = self.Relation_classifier(entity_pair_matrix)
        if(self.train_base):
            label = logit * (Index.float()).unsqueeze(-1)  
            label[Index] = self.loss_fnt.get_label(label[Index], num_labels=self.num_labels)
            Error = ((label[Index] != label[Index]).sum(-1) > 0).sum()
            if label_01 is not None:
                logits = logit[Index]
                labels = label_01[Index]
                loss += self.loss_fnt(logits.float(), labels)
                loss1 = loss
        else:
            entity_pair_matrix1 = self.CCA.Reduce_dense(entity_pair_matrix)
            if label_01 is not None:
                Sample_pro = entity_pair_masks

                Label, train_mask, train_mask1, Mask_label = self.create_sample(Sample_pro, label_01)
                Index_local = (train_mask > 0)
                logits = logit[Index_local]
                labels = label_01[Index_local]
                loss1 += self.loss_fnt(logits.float(), labels) / 10.0

                Label, train_mask,train_mask1,Mask_label = self.create_sample(Sample_pro,label_01)
                Input = torch.matmul(Label, self.CCA.relation_rep1)
                logit0, hidden, hidden1 = self.CCA(Input, entity_pair_matrix1, entity_pair_masks, entitys)

                Index_local = (train_mask > 0)
                logits = logit0[Index_local]
                labels = label_01[Index_local]
                loss += self.loss_fnt(logits.float(), labels) / 10.0

                Label, train_mask,train_mask1,Mask_label = self.create_sample(Sample_pro,label_01)
                Input = torch.matmul(Label, self.CCA.relation_rep1)
                logit1, hidden2, hidden3 = self.CCA(Input, entity_pair_matrix1, entity_pair_masks, entitys)

                Index_local = (train_mask1 > 0)
                logits = logit1[Index_local]
                labels = label_01[Index_local]
                loss += self.loss_fnt(logits.float(), labels) / 10.0

                contract_loss1 = contrast_loss_fn(hidden[Index], hidden3[Index].detach())
                contract_loss2 = contrast_loss_fn(hidden2[Index], hidden1[Index].detach())
                contract_loss = (contract_loss1 + contract_loss2).mean()
                loss1 += contract_loss
            else:
                label = logit * (Index.float()).unsqueeze(-1)  
                label[Index] = self.loss_fnt.get_label(label[Index], num_labels=self.num_labels)

                label_01 = label
                logit1 = logit.clone()
                for i in range(3):
                    Input = torch.matmul(label_01, self.CCA.relation_rep1).float()  
                    logit_t, _, _ = self.CCA(Input, entity_pair_matrix1, entity_pair_masks, entitys)
                    logit1 += logit_t
                    label_01 = logit1 * (Index.float()).unsqueeze(-1)  
                    label_01[Index] = self.loss_fnt.get_label(label_01[Index], num_labels=self.num_labels)

                Error = ((label[Index] != label_01[Index]).sum(-1) > 0).sum()

                logit = logit1

        logit1 = []
        for i, ht in enumerate(hts_tensor):
            logit1.append(logit[i][ht[:, 0], ht[:, 1], :])
        logit1 = torch.cat(logit1, 0)  

        return logit1,loss,loss1,Error

    
    def get_entity_pair(self, mention_feature, encoder_out, hts, Sentence_index, encoder_output, encoder_mask,
                            labels=None, evidence_num=None):
        sequence_output = mention_feature["mention"]
        Entity_attention = mention_feature["Entity_attention"]  
        entity2mention = mention_feature["entity2mention"]

        hss, tss, rss, Entity_pairs, hts_tensor, entitys = [], [], [], [], [],[]
        batch_size = sequence_output.size()[0]
        Pad = torch.zeros((1, self.config.hidden_size)).to(sequence_output.device)  
        for i in range(batch_size):
            entity_embs = []
            for j, e in enumerate(entity2mention[i]):
                e_index = torch.LongTensor(e).to(sequence_output.device)
                e_emb = torch.logsumexp(sequence_output[i].index_select(0, e_index), dim=0)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)       
            entitys.append(entity_embs)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hts_tensor.append(ht_i)

            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            doc_rep = sequence_output[i][0][None, :].expand(hs.size()[0], 768)

            h_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 0])
            t_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 1])
            
            
            ht_att = (h_att * t_att).sum(1)  
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  
            
            rs = contract("ld,rl->rd", encoder_out[i], ht_att)  

            hs_pair = self.head_pair(torch.cat([hs, rs,doc_rep], dim=1))    
            ts_pair = self.tail_pair(torch.cat([ts, rs,doc_rep], dim=1))    
            
            b1 = hs_pair.view(-1, self.hidden_size // self.block_size, self.block_size)
            b2 = ts_pair.view(-1, self.hidden_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.hidden_size* self.block_size)
            entity_pair = self.entity_pair_extractor(bl)  

            hss.append(hs)  
            tss.append(ts)  
            rss.append(rs)

            entity_pair = torch.cat([Pad, entity_pair], dim=0)  
            Entity_pairs.append(entity_pair)  

        Max_entity_num = max([len(x) for x in entity2mention])
        entity_pair_index = torch.zeros((batch_size, Max_entity_num, Max_entity_num)).long().to(sequence_output.device)
        label_01 = torch.zeros((batch_size, Max_entity_num, Max_entity_num, 97)).float()  
        E_num = torch.zeros((batch_size, Max_entity_num, Max_entity_num)).float()
        for i, ht_i in enumerate(hts_tensor):
            index = torch.arange(ht_i.size()[0]).to(sequence_output.device) + 1  
            entity_pair_index[i][ht_i[:, 0], ht_i[:, 1]] = index
            if labels is not None:
                label = torch.tensor(labels[i]).float()  
                label_01[i][ht_i[:, 0], ht_i[:, 1]] = label
                e_num = torch.tensor(evidence_num[i]).float()  
                E_num[i][ht_i[:, 0], ht_i[:, 1]] = e_num

        entity_pair_masks = (entity_pair_index != 0).float()
        label_01 = label_01.to(sequence_output.device)
        E_num = E_num.to(sequence_output.device)

        entity_pair_matrix = []
        for i in range(batch_size):
            entity_pair_matrix.append(Entity_pairs[i][entity_pair_index[i]])
            pad_l = Max_entity_num - entitys[i].size(0)
            entitys[i] = torch.cat([entitys[i],] + [Pad] * pad_l, dim=0)

        entity_pair_matrix = torch.stack(entity_pair_matrix, dim=0)

        hss = torch.cat(hss, dim=0)  
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        entitys = torch.stack(entitys)

        if labels is None:
            label_01 = None
            E_num = None

        Entity_feature = {
            "entity_pair_matrix": entity_pair_matrix,
            "entity_pair_masks": entity_pair_masks,
            "hss": hss,
            "tss": tss,
            "rss": rss,
            "hts_tensor": hts_tensor,
            "label_01": label_01,
            "entitys":entitys,
            "E_num":E_num,
        }
        return Entity_feature

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                entity_type=None,
                sample_index=None,
                Sentence_index=None,
                evidence_num=None
                ):

        sequence_output, attention,hidden_states = self.Encode(input_ids, attention_mask)
        sequence_output = (hidden_states[-1] + hidden_states[-2] + hidden_states[-3]) / 3.0

        mention_feature=self.get_mention_rep(sequence_output,attention,attention_mask,entity_pos,entity_type,Sentence_index)

        Entity_feature=self.get_entity_pair(
            mention_feature,
            sequence_output,
            hts,
            Sentence_index,
            sequence_output,
            attention_mask,
            labels=labels,
            evidence_num=evidence_num
        )

        logits,loss,loss1,Error = self.Entity_level_predict(Entity_feature)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

        if labels is not None:
            output = ((loss+loss1)/2,loss1) + output
        else:
            output = output + (Error,)

        return output
