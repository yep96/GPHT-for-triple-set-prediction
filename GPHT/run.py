import argparse
import logging
import os
import shutil
import time
from math import sqrt, ceil

import numpy as np
import torch
from data import DataBase, HeadDataSet, CompGCNDataSet
from models import GNNComplete, KGELoss
from torch.utils.data import DataLoader
from tqdm import tqdm


class Runner(object):
    def __init__(self, para):
        self.database = DataBase(para)
        self.para = para
        self.device = torch.device('cuda' if para.gpu != -1 else 'cpu')
        self.para.device = self.device
        para.embed_dim = para.k_w * para.k_h if para.embed_dim is None else para.embed_dim

        self.set_logger()
        logging.debug(vars(para))

        def get_data_loader(split, batch_size):
            if para.pretrain:
                return DataLoader(CompGCNDataSet(self.database, split, para), batch_size=para.batch_size, shuffle=True, num_workers=para.num_workers)
            return HeadDataSet(self.database, split, para)

        self.data_iter = {
            'train': get_data_loader('train', para.batch_size),
            'test': get_data_loader('test', para.batch_size),
            'valid': get_data_loader('valid', para.batch_size),
        }

        self.model = GNNComplete(self.database, para).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=para.lr, weight_decay=para.l2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=4, min_lr=1e-6)

        self.label_thr = torch.tensor(1.0 / self.database.num_ent if self.para.lbl_smooth != 0 else 0, requires_grad=False)
        self.mask_val = torch.tensor(-1e9, requires_grad=False)

        self.init_epoch = 0
        self.best_val, self.best_epoch = {'F1': -1}, -1

        self.all_ents = torch.arange(para.num_ent).to(self.device)

        if para.pretrain or para.testKGE != 0:
            self.database.edge_index = self.database.edge_index.to(self.device)
            self.database.edge_type = self.database.edge_type.to(self.device)

        if para.restore:
            self.load_model('best' if para.no_train else 'continue')

    def set_logger(self):
        log_file = os.path.join(self.para.save_path, ('test' if self.para.no_train else 'train') + '.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='%m/%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def save_model(self, desc, epoch):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'args': self.para,
            'epoch': epoch
        }
        torch.save(state, os.path.join(self.para.save_path, 'checkpoint.' + desc))

    def load_model(self, desc):
        state = torch.load(os.path.join(self.para.save_path, 'checkpoint.' + desc))

        self.best_val = state['best_val']
        self.best_epoch = state['best_epoch']
        self.init_epoch = state['epoch']

        self.model.load_state_dict(state['state_dict'])
        # self.optimizer.load_state_dict(state['optimizer'])

        logging.info(f"Load from checkpoint.{desc}, epoch={state['epoch']}")

    def read_batch(self, batch):
        triple, label = [_.to(self.device) for _ in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label

    def train(self, epoch):
        self.model.train()
        losses = []
        labelLosses, labelPre = [], []
        z2o, o2z = [], []
        oo2z = []
        allsteps = len(self.data_iter['train'])

        for step, batch in enumerate(self.data_iter['train']):
            if self.para.pretrain:
                self.optimizer.zero_grad()
                edge_index, edge_type, toTest = self.database.edge_index, self.database.edge_type, None
                triple, trp_label = [_.to(self.device) for _ in batch]
                fetch = self.model(edge_index, edge_type, toTest, "train", self.all_ents, triple)
                loss = self.model.loss_pre(fetch['kge'], trp_label)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            else:
                edge_index, edge_type, toTestBig, labelBig, subLabBig = [_.to(self.device) for _ in batch]
                cnt = ceil(toTestBig.shape[0] / 2048)
                for toTest, label, subLab in zip(*[torch.chunk(x, cnt, dim=0) for x in [toTestBig, labelBig, subLabBig]]):
                    self.optimizer.zero_grad()
                    fetch = self.model(edge_index, edge_type, toTest, "train")
                    loss = fetch['kge'].mean(dim=-1)

                    toAddLab = label ^ subLab
                    labelLoss = self.model.loss(fetch["testSco"], label, toAddLab)
                    loss = loss + labelLoss
                    
                    tmp = (fetch["testSco"] > 0.5).float().squeeze()
                    trueCnt = ((fetch["testSco"] > 0.5).squeeze() == label).sum(dim=-1).item()
                    labelLosses.append(labelLoss.item())
                    labelPre.append(trueCnt / label.shape[0])
                    z2o.append(tmp[~label].mean().item())
                    if toAddLab.any():
                        o2z.append(1 - tmp[toAddLab].mean().item())
                    oo2z.append(1 - tmp[subLab].mean().item())
                    
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

            print(f'{epoch}-{step:03d}/{allsteps}-{loss.item():.5f}', end='\r')

        if self.para.pretrain:
            self.scheduler.step(np.mean(losses))
            logging.info(f"{np.mean(losses):.4f} LR {self.optimizer.param_groups[0]['lr']:.6f}")
        else:
            self.scheduler.step(np.mean(o2z))
            logging.info(
                f"loss{np.mean(losses):.4f}   labelLoss{np.mean(labelLosses):.4f}-{len(labelLosses)}  labelPre{np.mean(labelPre):.4f}   "
                f"z2o{np.mean(z2o):.4f}  o2z{np.mean(o2z):.4f}   oo2z{np.mean(oo2z):.4f}  LR{self.optimizer.param_groups[0]['lr']:.6f}")
        loss = np.mean(losses)
        return loss

    def predict(self, split, epoch):
        self.model.eval()
        minconf = 0.3
        toKGE = dict()

        logging.info('start predict ht pair')
        timeStamp = time.time()
        with torch.no_grad():
            data_iter = self.data_iter[split]

            for batch in tqdm(data_iter, desc=split, ncols=60):
                edge_index, edge_type, toTest = [_.to(self.device) for _ in batch]
                if toTest.shape[0] != 0:
                    testSco = self.model(edge_index, edge_type, toTest, "predict")["testSco"]
                    toKGE.update({ht: sc for idx in torch.where(testSco > minconf)[0] if toKGE.get(
                        (ht := tuple(toTest[idx].tolist())), -1e9) < (sc := testSco[idx].item())})

        logging.info(f'ht pair finished, use {time.time() - timeStamp:.3f}s')
        if len(toKGE) == 0:
            return {'F1':0}

        for split in ['valid', 'test']:
            true_hts = self.database.htPair[split]
            right = [(ht in true_hts) for ht in toKGE]
            rec, pre, ll = np.sum(right)/len(true_hts), np.mean(right), len(right)
            logging.info(f'{split=}, recall={rec}, precision={pre}, cnt={ll}')
            if split == 'valid': vr, vp = rec*100, pre*100

        f1 = 2 / (1./vr + 1./vp)
        torch.save(toKGE, f"{self.para.save_path}/../toKGE_SP{self.para.perfix}V_f{f1:.4f}_r{int(vr)}_p{int(vp)}-T_r{int(rec*100)}_p{int(pre*100)}_l{ll}.pt")
        if self.para.testKGE <= 0 or len(toKGE) == 0:
            return {'F1': f1}

        metrics = {'add':0, 'true':0, 'MR':0, 'MRR':0, 'F1':0}
        with torch.no_grad():
            fetch = self.model(self.database.edge_index, self.database.edge_type, None, "train")
        ht_scos = [torch.tensor(x).to(self.device) for x in zip(*toKGE.items())]
        batches = [torch.chunk(x, self.para.testKGE, dim=0) for x in (ht_scos[0].unsqueeze(1), ht_scos[1])]
        batches = [x for x in zip(*batches)]

        rels = torch.arange(self.database.num_rel).to(self.device).unsqueeze(0)
        nent = self.database.num_ent
        nrel = self.database.num_rel
        length = len(toKGE) * nrel
        thrs = [3, 1, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 0.001]
        minThr = min(thrs)

        def idx2hrt(ind):
            return (ind//nent//nrel, ind//nent % nrel, ind % nent)

        scoreSum = 0
        addscores = dict()
        logging.info('start predict triple')
        timeStamp = time.time()

        with torch.no_grad():
            for ht, sco in tqdm(batches, desc='1-ht', ncols=60):
                scores = self.model.kgeloss(ht[:,:,0], rels, ht[:,:,1], fetch["ent_emb"], fetch["rel_emb"], self.model.trans, scos=sco)
                scoreSum += scores.exp().sum().item()
            for ht, sco in tqdm(batches, desc='2-ht', ncols=60):
                scores = self.model.kgeloss(ht[:,:,0], rels, ht[:,:,1], fetch["ent_emb"], fetch["rel_emb"], self.model.trans, scos=sco)
                scores = scores.exp() * length / scoreSum
                addscores.update({hrt: sc for idx in torch.where(scores >= minThr)[0] if addscores.get(
                    (hrt := (ht[(n := idx.item()//nrel),0,0]*nrel*nent + (idx%nrel)*nent + ht[n,0,1]).item()), -1e9) < (sc := scores[idx].item())})
        logging.info(f'triple end, use {time.time() - timeStamp:.3f}s')

        addscores = sorted(addscores.items(), key=lambda x: -x[1])
        bestF1 = -1
        for test in ['VALID', 'TEST']:
            truthTriples = set(self.database.data["train"] + self.database.data["valid"] + (self.database.data["test"] if test == 'TEST' else []))
            testTriples = set(self.database.data["test"] if test == 'TEST' else self.database.data["valid"])
            if self.para.owl:
                htPairs = {(h,t) for h,_,t in truthTriples}
            metrics = {}
            trueCnt, addedTrueCnt, idx = 0, 0, 0
            truthRank, truthRevRank, Len = 0, 0, len(addscores)
            for thr in sorted(thrs, reverse=True):
                while idx < Len and addscores[idx][1] >= thr:
                    hrt = idx2hrt(addscores[idx][0])
                    if self.para.owl and (hrt[0],hrt[2]) not in htPairs:
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
                with open(f"{self.para.save_path}/../{self.para.perfix}{'owl-' if self.para.owl else ''}metrics.info", 'a') as me:
                    me.write("{")
                    me.write('"split":"{}", "owl":{}, "label":"{}-{}-{}", "F1":{}, "add":{}, "true":{}, "MR":{}, "MMR":{}'.format(
                        test, self.para.owl, self.para.model, epoch, thr, *[metrics[m] for m in ['F1','add','true','MR','MRR']]))
                    me.write("}\n")
        logging.info(f'GNNtest done, valid best F1={bestF1}')

        return metrics

    def runEpochs(self):
        kill_cnt = 0
        for epoch in range(self.init_epoch + 1, self.para.max_epochs + 1):
            train_loss = self.train(epoch)
            (logging.info if epoch % 10 == 0 else logging.debug)(
                f'[Epoch:{epoch}]:  Training Loss:{train_loss:0.5f}  epoch:{epoch}   kill count={kill_cnt}   best F1={self.best_val["F1"]:0.5f}')

            if (epoch % self.para.valid_epochs == 0) or (epoch == self.para.max_epochs):
                self.save_model('continue', epoch)
                if self.para.pretrain:
                    continue
                val_results = self.predict('valid', epoch)
                if val_results['F1'] > self.best_val['F1']:
                    self.best_val = val_results
                    self.best_epoch = epoch
                    self.save_model('best', epoch)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    self.model.gamma += 2
                    if kill_cnt % 10 == 0:
                        self.model.gamma += 10
                        for params in self.optimizer.param_groups:
                            params['lr'] *= self.para.lr_scale
                        logging.info('Gamma and LR decay on saturation, updated value of gamma: {}'.format(self.model.gamma))
                    if kill_cnt > 15:
                        logging.info("Early Stopping!!")
                        break

        logging.info(f'best epoch:{self.best_epoch}')
        for metric, value in self.best_val.items():
            logging.info(f'best-{epoch}   {metric}:\t{value}')

    def run(self):
        if self.para.no_train:
            self.predict('test', self.best_epoch)
        else:
            self.runEpochs()


def getPara():
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-no_train', default=False, action='store_true')
    parser.add_argument('-data_dir', type=str, default='../DATA', help="set dataset folder's parent folder")
    parser.add_argument('-exp_dir', type=str, default='EXPS', help="set experient path")
    parser.add_argument('-perfix', type=str, default='', help="perfix of train/test/valid")
    parser.add_argument('-valid_epochs', type=int, default=10, help="do valid per valid_epochs")
    parser.add_argument('-lr_scale', type=float, default=0.7, help="learning rate decay when kill_cnt==10")
    parser.add_argument('-desc', type=str, default='tryModel', help="description for save folder")
    parser.add_argument('-subgraph', type=int, default=3, help="select N hops subgraph")
    parser.add_argument('-topred', type=float, default=0.05, help="set delete subgraph scale to predict while train")
    parser.add_argument('-padLen', type=int, default=0, help="set padding length of edge_index/edge_type")
    parser.add_argument('-padIdx', type=int, default=0, help="set padding idx of edge_index/edge_type")
    parser.add_argument('-topk', type=int, default=5, help="choose how many h,r,t candidates while test")
    parser.add_argument('-testKGE', type=int, default=0, help="0 does not perform kge test, otherwise it is the number of test groups")
    parser.add_argument('-del_exceed', default=False, action='store_true')

    parser.add_argument('-pretrain', default=False, action='store_true', help="pretrain gnn+hake")
    parser.add_argument('-owl', default=False, action='store_true', help="open world setting")

    parser.add_argument('-dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('-model', type=str, default='compgcn', help='Model Name', choices=['compgcn'])
    parser.add_argument('-score_func', type=str, default='hake')
    parser.add_argument('-opn', type=str, default='mult', help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch', type=int, default=1, dest='batch_size', help='Batch size')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', type=int, dest='max_epochs', default=500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', type=int, default=41504, help='Seed for randomization')

    parser.add_argument('-restore', type=str, default='', help='Restore from the previously saved model')
    parser.add_argument('-bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-num_bases', type=int, default=-1, help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim', type=int, default=100, help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', type=int, default=200, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', type=int, default=1024, help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', type=int, default=1, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', type=float, default=0.1, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', type=float, default=0.3, help='Dropout after GCN')

    parser.add_argument('-hid_drop2', type=float, default=0.3, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', type=float, default=0.3, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', type=int, default=10, help='ConvE: k_w')
    parser.add_argument('-k_h', type=int, default=20, help='ConvE: k_h')
    parser.add_argument('-num_filt', type=int, default=200, help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', type=int, default=7, help='ConvE: Kernel size to use')

    args = parser.parse_args()
    assert 0 <= args.lbl_smooth < 1.0, "label smooth error"
    assert args.restore or not args.no_train

    assert args.batch_size == 1 or args.pretrain
    if args.score_func.lower() == 'hake':
        args.embed_dim = (args.embed_dim // 3) * 3
    elif args.score_func.lower() in ['rotate']:
        args.embed_dim = (args.embed_dim // 2) * 2
    return args


if __name__ == "__main__":
    args = getPara()
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    if args.restore:
        args.save_path = args.restore
        args.desc = 'load'
    else:
        args.save_path = os.path.join(
            args.exp_dir, args.dataset, f'{args.desc}-{args.score_func}-{args.perfix}{os.uname().nodename}-' + time.strftime(r'%Y%m%d-%H:%M'))
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    runner = Runner(args)
    if args.desc == 'tryModel':
        logging.warning("\n" + "*"*66 + "\n!!!!!!! checkpoint will not be saved without 'desc' setted !!!!!!!\n" + "*"*66)
    try:
        runner.run()
    except KeyboardInterrupt:
        print('exit')
    except Exception:
        import traceback
        traceback.print_exc()
    if args.desc == 'tryModel':
        shutil.rmtree(args.save_path)
    elif not args.no_train:
        os.system(f'grep INFO {args.save_path}/train.log > {args.save_path}/train.info')
