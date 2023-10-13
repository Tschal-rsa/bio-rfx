# Bio-RFX

This repository is the official implementation of Bio-RFX.

**Bio-RFX: Refining Biomedical Extraction via Advanced Relation Classification and Structural Constraints**

## Requirements

1. Python 3.8 / 3.9
2. `torch` 1.10.2 (recommended)
3. `scikit-learn` 1.0.2 (recommended)
4. `allennlp`

## NER

### Train an entity detector

Available datasets:

1. BB (Bacteria Biotope)
2. DrugVar
3. DrugVar-500
4. DrugVar-200
5. DrugProt
6. DrugProt-500
6. DrugProt-200

```bash
python run_entity_span.py --dataset BB --name Ent --NER
```

### Train a number predictor

```bash
python run_count.py --dataset BB --name Cnt --NER
```

### Evaluate

Suppose the model paths of the entity detector and the number predictor are:

1. `./train/checkpoint_Ent_20230924_094832.pth.tar`
2. `./train/checkpoint_Cnt_20230924_095333.pth.tar`

```bash
python eval_entity.py --dataset BB --ent_name Ent --ent_id 20230924_094832 --cnt_name Cnt --cnt_id 20230924_095333
```

## RE

### Train a relation classifier

```bash
python run_relation.py --dataset BB --name Rel
```

### Train an entity detector

```bash
python run_entity_span.py --dataset BB --name Ent
```

### Train a number predictor

```bash
python run_count.py --dataset BB --name Cnt
```

### Evaluate

Suppose the model paths of the relation classifier, the entity detector and the number predictor are:

1. `./train/checkpoint_Rel_20230924_104232.pth.tar`
2. `./train/checkpoint_Ent_20230924_104109.pth.tar`
3. `./train/checkpoint_Cnt_20230924_104305.pth.tar`

```bash
python eval_relation.py --dataset BB --rel_name Rel --rel_id 20230924_104232 --ent_name Ent --ent_id 20230924_104109 --cnt_name Cnt --cnt_id 20230924_104305
```