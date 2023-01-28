import argparse, os, pickle, random
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=True)

dataset = parser.parse_args().dataset
path_ori = '{}/data_ori'.format(dataset)
path_mod = '{}/data'.format(dataset)

if not os.path.exists(path_mod):
    os.mkdir(path_mod)

with open(join(path_ori, 'all.txt')) as f, open(join(path_mod, 'complete.txt'), 'w') as comp:
    lines = f.readlines()
    comp.writelines(lines)
    lines = [x.strip().split() for x in lines]

ents = {x[0] for x in lines} | {x[2] for x in lines}
rels = {x[1] for x in lines}

with open(join(path_mod, 'e2i_sel.ple'), 'wb') as e2iF, open(join(path_mod, 'entities.dict'), 'w') as e2idic:
    pickle.dump({ent: i for i,ent in enumerate(ents)}, e2iF)
    for i,ent in enumerate(ents):
        e2idic.write(f"{i}\t{ent}\n")
with open(join(path_mod, 'r2i_sel.ple'), 'wb') as r2iF, open(join(path_mod, 'relations.dict'), 'w') as r2idic:
    pickle.dump({key: i for i,key in enumerate(rels)}, r2iF)
    for i,key in enumerate(rels):
        r2idic.write(f"{i}\t{key}\n")

random.shuffle(lines)
tripCnt = len(lines)
trainData = lines[: int(tripCnt*0.2*0.9)]
validtestData = lines[int(tripCnt*0.2*0.9):]
cnt = {}
for hrt in trainData:
    for i in hrt:
        cnt[i] = cnt.get(i, 0) + 1

index = -1
while len(cnt) < len(ents) + len(rels):
    index += 1
    hrt = validtestData[index]
    if any([x not in cnt for x in hrt]):
        trainData.append(hrt)
        for i in hrt:
            cnt[i] = cnt.get(i, 0) + 1
        del validtestData[index]
        index -= 1
    print(f'{len(cnt)} / {len(ents)}+{len(rels)}', end='  \r')

for delNum in [4,3,2]:
    index = -1
    while len(trainData) > tripCnt*0.2*0.9 and index < len(trainData)-1:
        index += 1
        hrt = trainData[index]
        if all([cnt[x] > delNum for x in hrt]):
            validtestData.append(hrt)
            for i in hrt:
                cnt[i] = cnt.get(i, 0) - 1
            del trainData[index]
            index -= 1

entCnt = set()
for h, _, t in trainData:
    entCnt.add(h)
    entCnt.add(t)
assert len(entCnt) == len(ents)

print(f"\n{len(trainData)}")

for percent in [0.2, 0.4, 0.6, 0.8]:
    trainFile = open(join(path_mod, f'{percent}_train.txt'), 'w')
    validFile = open(join(path_mod, f'{percent}_valid.txt'), 'w')
    testFile = open(join(path_mod, f'{percent}_test.txt'), 'w')
    for h, r, t in trainData:
        trainFile.write(f'{h}\t{r}\t{t}\n')

    trainRem = int(tripCnt * percent * 0.9 - len(trainData))
    trainRem = trainRem if trainRem > 0 else 0
    for h, r, t in validtestData[:trainRem]:
        trainFile.write(f'{h}\t{r}\t{t}\n')

    spli = int(tripCnt * percent - len(trainData))
    spli = spli if spli > 0 else 0
    for h, r, t in validtestData[trainRem:spli]:
        validFile.write(f'{h}\t{r}\t{t}\n')

    for h, r, t in validtestData[spli:]:
        testFile.write(f'{h}\t{r}\t{t}\n')

    trainFile.close()
    validFile.close()
    testFile.close()
