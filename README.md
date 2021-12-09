# Master Thesis

This repository contains the codebase for experiments related to my master thesis.

## Research Question

## Quickstart

To run a training use the following command from the project root.

```
python train.py                   <configuration yaml file>           <optional arguments>
python .\thesis\training\train.py .\resources\configs\train_grid.yaml --model risk
```

The `train.py` script accepts several arguments to customize the training.

```
--name          Names the training run.
                This helps finding the training results and allows for runs to be resumed.

--iterations    The number of iterations the training will run for.
---model        The model to be trained. Use "risk" to train the risk aware model.
--resume        Resumes a previously started training.
--root          The root directory to which all relative paths relate.
```
