# Bio-RFX

This repository is the official implementation of Bio-RFX.

**Bio-RFX: Refining Biomedical Extraction via Advanced Relation Classification and Structural Constraints**

## Requirements

1. Python 3.8 / 3.9
2. `torch` 1.10.2 (recommended)
3. `scikit-learn` 1.0.2 (recommended)
4. `allennlp` 2.10.1 (recommended)

## Datasets

Preprocessed datasets (in data/):

1. DrugVar
2. DrugVar-500
3. DrugVar-200
4. DrugProt
5. DrugProt-500
6. DrugProt-200
7. BC5CDR
8. CRAFT

## NER

### Train an entity detector

```bash
python run_entity_span.py --dataset DrugVar --name Ent --NER
```

### Train a number predictor

```bash
python run_count.py --dataset DrugVar --name Cnt --NER
```

### Evaluate

Suppose the model paths of the entity detector and the number predictor are:

1. `./train/checkpoint_Ent_20230924_094832.pth.tar`
2. `./train/checkpoint_Cnt_20230924_095333.pth.tar`

```bash
python eval_entity.py --dataset DrugVar --ent_name Ent --ent_id 20230924_094832 --cnt_name Cnt --cnt_id 20230924_095333
```

## RE

### Train a relation classifier

```bash
python run_relation.py --dataset DrugVar --name Rel
```

### Train an entity detector

```bash
python run_entity_span.py --dataset DrugVar --name Ent
```

### Train a number predictor

```bash
python run_count.py --dataset DrugVar --name Cnt
```

### Evaluate

Suppose the model paths of the relation classifier, the entity detector and the number predictor are:

1. `./train/checkpoint_Rel_20230924_104232.pth.tar`
2. `./train/checkpoint_Ent_20230924_104109.pth.tar`
3. `./train/checkpoint_Cnt_20230924_104305.pth.tar`

```bash
python eval_relation.py --dataset DrugVar --rel_name Rel --rel_id 20230924_104232 --ent_name Ent --ent_id 20230924_104109 --cnt_name Cnt --cnt_id 20230924_104305
```