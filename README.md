# anonymous

## Preprocess

Construct a close world dataset `Family` and split the dataset in different integrity randomly. Run the following command.

```bash
bash init.sh
```

## RuleTensor-TSP

```bash
python GraphRule.py -dataset=DATASET -rule_len=LEN -hc_thr=HC -sc_thr=SC -percent=PER -gpu=GPU
```

`DATASET`: choose the dataset in `DATA/`

`LEN`: set the length of rule

`HC`: set the head coverage threshold of rule

`SC`: set the standard confidence threshold of rule

`PER`: set the integrity of the dataset

`GPU`: -1 for cpu, otherwise the gpu id

## HAKE-TSP

```bash
python runs.py -train -test -data=DATASET -gpu=GPU -perfix=PERFIX --valid_steps=STEP
```

`PERFIX`: set the integrity of the dataset in the format of `percent_`, like `0.6_`

`STEP`: do valid every `STEP` steps

## GPHT

### 1. generate subgraphs

```bash
python run.py -dataset=DATASET -subgraph=SUBLEN -perfix=PERFIX
```

`SUBLEN`: set max hops of subgraph from center to edge

### 2. pre-train embeddings

```bash
python run.py -dataset=DATASET -subgraph=SUBLEN -perfix=PERFIX -batch=BATCH -pretrain -desc=DESC
```

### 3. train the model

```bash
python run.py -dataset=DATASET -subgraph=SUBLEN -perfix=PERFIX -lr=LR -restore=RESTORE
```

`LR`: a little scale number for learning rate, like 0.00003 or less

### 4. predict triples(in `HAKE-TSP`)

```bash
python runs.py -train -test -data=DATASET -gpu=GPU -perfix=PERFIX --valid_steps=STEP -testGNN ../GPHT/EXPS/DATASET/toKGE_XXX.pt
```

## Acknowledgement

We refer to the code of [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE) and [CompGCN](https://github.com/malllabiisc/CompGCN). Thanks for their contributions.
