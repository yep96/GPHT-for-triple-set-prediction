import logging
from math import sqrt
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict as ddict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import BatchType, TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class KGEModel(nn.Module, ABC):
    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args, steps):
        logging.info('test_step')
        model.eval()

        bigData = True
        scoreSum = 0

        nents = len(data_reader.entity_dict)
        nrels = len(data_reader.relation_dict)
        ents = torch.LongTensor(range(nents))
        scores = {} if bigData else []

        thrs = [args.thr]
        length = nents * nents * nrels
        if args.thr < 0:
            thrs = [10, 3, 1, 0.3, 0.1, 0.05, 0.001]
        minThr = min(thrs) # 0.3

        def idx2hrt(ind):
            return (ind//nents//nrels, ind//nents % nrels, ind % nents)

        with torch.no_grad():
            for head in tqdm(range(nents), desc="-Head", ncols=60):
                for rel in tqdm(range(nrels), desc="    Relation", ncols=60, leave=False):
                    positive_sample = torch.LongTensor([[head, rel, -1]]).cuda()
                    negative_sample = ents.cuda().clone().unsqueeze(0)

                    score = model((positive_sample, negative_sample), BatchType.TAIL_BATCH)
                    if bigData:
                        score.exp_()
                        scoreSum += score.sum().item()
                    else:
                        scores.extend(score.clone().detach().cpu())

        if bigData:
            scores = {}
            with torch.no_grad():
                for head in tqdm(range(nents), desc="+Head", ncols=60):
                    for rel in tqdm(range(nrels), desc="    Relation", ncols=60, leave=False):
                        positive_sample = torch.LongTensor([[head, rel, -1]]).cuda()
                        negative_sample = ents.cuda().clone().unsqueeze(0)
                        score = model((positive_sample, negative_sample), BatchType.TAIL_BATCH)[0]
                        score = score.exp() * length / scoreSum

                        scores.update({hrt: sc  for idx in torch.where(score >= minThr)[0] if scores.get(
                            (hrt := head*nrels*nents + rel*nents + idx.item()), -1e9) < (sc := score[idx].item())})

            scores = sorted(scores.items(), key=lambda x: -x[1])
        else:
            scores = torch.cat(scores, dim=0).softmax(dim=0).unsqueeze(0) * length
            raise "torch.cat may change indices +-1"
            values, indices = [x.cpu().tolist() for x in scores.sort(dim=1, descending=True)]
            scores = torch.cat(scores.sort(dim=1, descending=True), dim=0).T.tolist()

        bestF1 = -1
        for owl in [True, False]:
            for test in ['VALID', 'TEST']:
                truthTriples = set(data_reader.train_data + data_reader.valid_data + (data_reader.test_data if test == 'TEST' else []))
                testTriples = set(data_reader.test_data if test == 'TEST' else data_reader.valid_data)
                if owl:
                    htPairs = {(h,t) for h,_,t in truthTriples}
                metrics = {}
                trueCnt, addedTrueCnt, idx = 0, 0, 0
                truthRank, truthRevRank, Len = 0, 0, len(scores)
                for thr in sorted(thrs, reverse=True):
                    while idx < Len and scores[idx][1] >= thr:
                        hrt = idx2hrt(scores[idx][0])
                        if owl and (hrt[0],hrt[2]) not in htPairs:
                            trueCnt += 1
                            idx += 1
                            continue
                        if hrt in truthTriples:
                            if hrt in testTriples:
                                truthRank += idx + 1 - trueCnt
                                truthRevRank += 1 / (idx + 1 - trueCnt)
                                addedTrueCnt += 1
                            trueCnt += 1
                        idx += 1
                    metrics['add'] = idx - trueCnt + addedTrueCnt
                    metrics['true'] = addedTrueCnt
                    metrics['MR'] = truthRank / addedTrueCnt if addedTrueCnt else -1
                    metrics['MRR'] = truthRevRank / addedTrueCnt if addedTrueCnt else -1
                    metrics['F1'] = 2 / (metrics['add']/metrics['true'] + sqrt(len(testTriples)/metrics['true'])) if metrics['true']!=0 else 0
                    if test == 'VALID':
                        if metrics['F1'] > bestF1: bestF1 = metrics['F1']
                    with open(f"{args.save_path}/../metric-SP{args.perfix}.txt", 'a') as me:
                        me.write("{")
                        me.write('"split":"{}", "owl":"{}", "label":"{}-{}-{}", "F1":{}, "add":{}, "true":{}, "MR":{}, "MMR":{}'.format(
                            test, owl, args.model, steps, thr, *[metrics[m] for m in ['F1','add','true','MR','MRR']]))
                        me.write("}\n")
        logging.info('test_step done')
        metrics['F1'] = bestF1
        return metrics

    def GNNtest(self, model, data_reader, ht_scos, args, steps):
        logging.info('GNNtest')
        model.eval()
        nent = len(data_reader.entity_dict)
        nrel = len(data_reader.relation_dict)

        cnt = ht_scos[0].shape[0] // 4096
        batches = [torch.chunk(x, cnt, dim=0) for x in ht_scos]
        batches = [x for x in zip(*batches)]

        addscores = {}
        scoreSum = 0
        with torch.no_grad():
            for ht, sco in tqdm(batches, desc="1-ht", ncols=60):
                head = self.entity_embedding[ht[:, 0]].unsqueeze(1)
                tail = self.entity_embedding[ht[:, 1]].unsqueeze(1)
                relation = self.relation_embedding.unsqueeze(0)

                score = self.func(head, relation, tail, BatchType.RELA_BATCH)
                scoreSum += score.view(-1).exp().sum().item()

        def idx2hrt(ind):
            return (ind//nent//nrel, ind//nent % nrel, ind % nent)

        thrs = [args.thr]
        if args.thr < 0:
            thrs = [10, 3, 1, 0.3, 0.1, 0.05, 0.001]
        minThr = min(thrs)
        length = ht_scos[1].shape[0] * nrel
        addscores = {}
        with torch.no_grad():
            for ht, sco in tqdm(batches, desc="2-ht", ncols=60):
                head = self.entity_embedding[ht[:, 0]].unsqueeze(1)
                tail = self.entity_embedding[ht[:, 1]].unsqueeze(1)
                relation = self.relation_embedding.unsqueeze(0)

                score = self.func(head, relation, tail, BatchType.RELA_BATCH)
                score = score.view(-1).exp() * length / scoreSum

                addscores.update({hrt: sc for idx in torch.where(score >= minThr)[0] if addscores.get(
                    (hrt := (ht[(n := idx.item()//nrel),0]*nrel*nent + (idx%nrel)*nent + ht[n,1]).item()), -1e9) < (sc := score[idx].item())})

        addscores = sorted(addscores.items(), key=lambda x: -x[1])

        bestF1 = -1
        for owl in [True, False]:
            for test in ['VALID', 'TEST']:
                truthTriples = set(data_reader.train_data + data_reader.valid_data + (data_reader.test_data if test == 'TEST' else []))
                testTriples = set(data_reader.test_data if test == 'TEST' else data_reader.valid_data)
                if owl:
                    htPairs = {(h,t) for h,_,t in truthTriples}
                metrics = {}
                trueCnt, addedTrueCnt, idx = 0, 0, 0
                truthRank, truthRevRank, Len = 0, 0, len(addscores)
                for thr in sorted(thrs, reverse=True):
                    while idx < Len and addscores[idx][1] >= thr:
                        hrt = idx2hrt(addscores[idx][0])
                        if owl and (hrt[0],hrt[2]) not in htPairs:
                            trueCnt += 1
                            idx += 1
                            continue
                        if hrt in truthTriples:
                            if hrt in testTriples:
                                truthRank += idx + 1 - trueCnt
                                truthRevRank += 1 / (idx + 1 - trueCnt)
                                addedTrueCnt += 1
                            trueCnt += 1
                        idx += 1
                    metrics[f'add'] = idx - trueCnt + addedTrueCnt
                    metrics[f'true'] = addedTrueCnt
                    metrics[f'MR'] = truthRank / addedTrueCnt if addedTrueCnt else -1
                    metrics[f'MRR'] = truthRevRank / addedTrueCnt if addedTrueCnt else -1
                    metrics['F1'] = 2 / (metrics['add']/metrics['true'] + sqrt(len(testTriples)/metrics['true'])) if metrics['true']!=0 else 0
                    if test == 'VALID':
                        if metrics['F1'] > bestF1: bestF1 = metrics['F1']
                    with open(f"{args.save_path}/../GNN-metric-SP{args.perfix}.txt", 'a') as me:
                            me.write("{")
                            me.write('"split":"{}", "owl":"{}", "label":"{}-{}-{}", "F1":{}, "add":{}, "true":{}, "MR":{}, "MMR":{}'.format(
                                test, owl, args.model, steps, thr, *[metrics[m] for m in ['F1','add','true','MR','MRR']]))
                            me.write("}\n")
        logging.info('GNNtest done')
        return metrics


class ModE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(ModE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def func(self, head, rel, tail, batch_type):
        return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)


class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        elif batch_type == BatchType.RELA_BATCH:
            phase_score = (phase_head - phase_tail) + phase_relation
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
