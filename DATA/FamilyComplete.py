import pickle, torch, os
from random import shuffle
from os.path import join, exists


class CompleteFamily:
    def __init__(self, path_ori, path_mod) -> None:
        self.path_ori = path_ori
        self.path_mod = path_mod

    def filtGender(self):
        male = {'husband', 'father', 'son', 'brother', 'uncle', 'nephew'}
        female = {'wife', 'mother', 'daughter', 'sister', 'aunt', 'niece'}
        with open(join(self.path_ori, 'all_ori.txt')) as f:
            lines = [x.strip().split() for x in f.readlines()]
            lines = [(int(h), r, int(t)) for h, r, t in lines]

        ent_max = max(max([x[0] for x in lines]), max([x[2] for x in lines]))
        gender = torch.zeros(ent_max+1) - 1
        seen = torch.zeros(ent_max+1)
        for h, r, t in lines:
            if r in male:
                gender[h] = 1
            elif r in female:
                gender[h] = 0

        for h, r, t in lines:
            if gender[h] != -1 and gender[t] != -1:
                seen[h] = seen[t] = 1

        ents = torch.where(seen != 0)[0].tolist()
        self.e2i = {str(ents[i]): i for i in range(len(ents))}
        self.i2e = {v: k for k, v in self.e2i.items()}
        rels = ['husband', 'wife', 'father', 'mother', 'son', 'daughter', 'brother', 'sister', 'uncle', 'aunt', 'nephew', 'niece']
        self.r2i =  {key: i for i, key in enumerate(rels)}
        self.gender = gender[ents]

    def buildGraph(self):
        self.nents = len(self.e2i)
        self.relM = {key: torch.zeros(self.nents, self.nents) for key in self.r2i}

        with open(join(self.path_ori, 'all_ori.txt')) as f:
            for line in f:
                try:
                    h, r, t = line.strip().split()
                    h, t = self.e2i[h], self.e2i[t]
                    self.relM[r][h, t] = 1
                except KeyError:
                    pass

        assert (self.gender[self.relM['father'].max(dim=1).values != 0] == 1).all()

    def couple(self):
        self.relM['husband'] = (self.relM['husband'] + self.relM['wife'].T).clip(max=1)
        self.relM['wife'] = self.relM['husband'].T.clone()
        assert (self.gender[self.relM['husband'].max(dim=1).values != 0] == 1).all()
        assert (self.gender[self.relM['wife'].max(dim=1).values != 0] == 0).all()

        self.relCoup = self.relM['wife'] + self.relM['husband']
        assert (self.relCoup == self.relCoup.T).all()

    def parent(self):
        self.relM['father'] = (self.relM['father'] + self.relM['son'].T + self.relM['daughter'].T).clip(max=1)
        self.relM['father'] *= self.gender.unsqueeze(1)
        self.relM['mother'] = (self.relM['mother'] + self.relM['son'].T + self.relM['daughter'].T).clip(max=1)
        self.relM['mother'] *= 1 - self.gender.unsqueeze(1)
        assert (self.gender[self.relM['father'].max(dim=1).values != 0] == 1).all()
        assert (self.gender[self.relM['mother'].max(dim=1).values != 0] == 0).all()
        assert self.relM['father'].sum(dim=0).max() == 1
        assert self.relM['mother'].sum(dim=0).max() == 1

        self.relM['son'] = (self.relM['son'] + self.relM['father'].T + self.relM['mother'].T).clip(max=1)
        self.relM['son'] *= self.gender.unsqueeze(1)
        self.relM['daughter'] = (self.relM['daughter'] + self.relM['father'].T + self.relM['mother'].T).clip(max=1)
        self.relM['daughter'] *= 1 - self.gender.unsqueeze(1)
        assert (self.gender[self.relM['son'].max(dim=1).values != 0] == 1).all()
        assert (self.gender[self.relM['daughter'].max(dim=1).values != 0] == 0).all()
        assert self.relM['son'].sum(dim=1).max() == 2
        assert self.relM['daughter'].sum(dim=1).max() == 2

        self.relParent = self.relM['father'] + self.relM['mother']
        self.relChild = self.relM['son'] + self.relM['daughter']
        assert (self.relParent == self.relChild.T).all()

    def bro_sis(self):
        tmp = self.relM['brother'] + self.relM['sister']

        for par in torch.where(self.relParent.sum(dim=1) > 1)[0]:
            for bro in torch.where(self.relParent[par] != 0)[0]:
                tmp[bro] += self.relParent[par]
        tmp = (tmp + tmp.T).clip(max=1) - torch.eye(self.nents)
        tmp.clip_(min=0)

        self.relM['brother'] = tmp * self.gender.unsqueeze(1)
        self.relM['sister'] = tmp * (1 - self.gender.unsqueeze(1))
        assert (self.gender[self.relM['brother'].max(dim=1).values != 0] == 1).all()
        assert (self.gender[self.relM['sister'].max(dim=1).values != 0] == 0).all()

        cnt_unknow = 0
        add_parent = 0
        for one in torch.where(tmp.sum(dim=1) != 0)[0]:
            Mo = torch.where(self.relM['mother'][:, one] != 0)[0]
            Fa = torch.where(self.relM['father'][:, one] != 0)[0]
            Mo = Mo.item() if len(Mo) else -1
            Fa = Fa.item() if len(Fa) else -2
            for ano in torch.where(tmp[one] != 0)[0]:
                mo = torch.where(self.relM['mother'][:, ano] != 0)[0]
                fa = torch.where(self.relM['father'][:, ano] != 0)[0]
                mo = mo.item() if len(mo) else -3
                fa = fa.item() if len(fa) else -4

                if not (Mo == mo or Fa == fa):
                    flg = (Mo >= 0) + (Fa >= 0) + (mo >= 0) + (fa >= 0)
                    assert flg != 4
                    if flg < 3:
                        cnt_unknow += 1
                        continue
                    add_parent += 1
                    if Mo < 0:
                        Mo = mo
                        self.relM['mother'][Mo, one] = 1
                    elif Fa < 0:
                        Fa = fa
                        self.relM['father'][Fa, one] = 1
                    elif mo < 0:
                        self.relM['mother'][mo, ano] = 1
                    else:
                        self.relM['father'][fa, ano] = 1
        print('cnt_unknow:', cnt_unknow, '\tadd_parent:', add_parent)

        self.relBroSis = self.relM['brother'] + self.relM['sister']

    def un_an(self):
        tmp = self.relM['uncle'] + self.relM['aunt']
        tmp += self.relBroSis @ self.relParent
        tmp = tmp + self.relCoup @ tmp
        tmp.clip_(max=1)

        self.relM['uncle'] = tmp * self.gender.unsqueeze(1)
        self.relM['aunt'] = tmp * (1 - self.gender.unsqueeze(1))

    def ne_ni(self):
        tmp = self.relM['nephew'] + self.relM['niece']
        tmp += self.relChild @ self.relBroSis
        tmp = tmp + tmp @ self.relCoup
        tmp.clip_(max=1)

        self.relM['nephew'] = tmp * self.gender.unsqueeze(1)
        self.relM['niece'] = tmp * (1 - self.gender.unsqueeze(1))

    def saveData(self):
        self.triples = []
        complete = open(join(self.path_mod, 'all.txt'), 'w')
        for r in self.r2i.keys():
            tmp = [x.tolist() for x in torch.where(self.relM[r] != 0)]

            for i, j in zip(*tmp):
                h, t = self.i2e[i], self.i2e[j]
                complete.write(f'{h}\tis_{r}_of\t{t}\n')
                self.triples.append((h, f'is_{r}_of', t))
        complete.close()

    def splitData(self):
        shuffle(self.triples)
        tripCnt = len(self.triples)
        trainData = self.triples[: int(tripCnt*0.2*0.9)]
        testData = self.triples[int(tripCnt*0.2*0.9):]
        cnt = {}
        for hrt in trainData:
            for i in hrt:
                cnt[i] = cnt.get(i, 0) + 1

        index = -1
        while len(cnt) < self.nents + len(self.r2i):
            index += 1
            hrt = testData[index]
            if any([x not in cnt for x in hrt]):
                trainData.append(hrt)
                for i in hrt:
                    cnt[i] = cnt.get(i, 0) + 1
                del testData[index]

        index = -1
        while len(trainData) > tripCnt*0.2:
            index += 1
            hrt = trainData[index]
            if all([cnt[x] > 2 for x in hrt]):
                testData.append(hrt)
                for i in hrt:
                    cnt[i] = cnt.get(i, 0) - 1
                del trainData[index]

        ents = set()
        for h, _, t in trainData:
            ents.add(h)
            ents.add(t)
        assert len(ents) == len(self.e2i)

        for percent in [0.2, 0.4, 0.6, 0.8]:
            trainFile = open(join(self.path_mod, f'{percent}_train.txt'), 'w')
            validFile = open(join(self.path_mod, f'{percent}_valid.txt'), 'w')
            testFile = open(join(self.path_mod, f'{percent}_test.txt'), 'w')
            for h, r, t in trainData:
                trainFile.write(f'{h}\t{r}\t{t}\n')
            spli = int(tripCnt * (percent - 0.2))
            for h, r, t in testData[:spli]:
                trainFile.write(f'{h}\t{r}\t{t}\n')
            for h, r, t in testData[spli:]:
                testFile.write(f'{h}\t{r}\t{t}\n')
            trainFile.close()
            testFile.close()

    def __repr__(self) -> str:
        return '\t'.join([f'{r}:{self.relM[r].sum().item():.0f}' for r in self.r2i])


if __name__ == '__main__':
    path_ori = 'Family/data_ori'
    path_mod = 'Family/data_ori'
    if not exists(path_mod):
        os.mkdir(path_mod)
    comp = CompleteFamily(path_ori, path_mod)
    if not exists(join(path_mod, 'gender_sel.ple')):
        comp.filtGender()

    comp.buildGraph()
    print('init:\n', comp)
    comp.couple()
    comp.parent()
    comp.bro_sis()
    print('111:\n', comp)
    comp.parent()
    print('222:\n', comp)
    comp.un_an()
    comp.ne_ni()
    print('333:\n', comp)
    comp.saveData()
