# Master Thesis

This repository contains the codebase for experiments related to my master thesis.

## Research Question

## Quickstart

### Training

To run a training use the following command from the project root.

```
python train.py <params> [--name <str>] [--resume]
python .\thesis\training\train.py .\resources\configs\train_grid.yaml --name example
```

The `train.py` script accepts several arguments to customize the training.

```
params          The path to the configuration YAML file.

--name          Names the training run.
                This helps finding the training results and allows for runs to be resumed.
--resume        Resumes a previously started training.
```

### Validation

To validate a previously trained model use the following command from the project root.

```
python validate.py <params> <checkpoint>
```

The `validate.py` script accepts several arguments to customize the training.

```
params          The path to the configuration YAML file.
checkpoint      The path to the checkpoint file to load the model from.
```

## Data

The data for the training is sourced from the Bitfinex api. The data consists of five-minute snapshots for the ten crypto currencies with the highest market capitalization available on Bitfinex for the year 2021. The data interval in UNIX milliseconds stretches from `1609459200000` to `1640995200000`.

## Tensorboard

To visualize the results of training runs Tensorboard can be used. The training results alongside the checkpoints are saved to the `. \checkpoints` directory. To host a Tensorboard for a specific run or experiment use the `tensorboard` command.

```
tensorboard --logdir <path to run or experiment folder>
```

## Weights & Balances

The training results are also logged to Weights & Balances. For this a valid api key must be placed in a file named `wandb.key` at the root of the project. Additionally a project names `master thesis` must be available as a logging target.

Note that for this to work on Windows some changes had to be made to the wandb RlLib integration. Depending on the platform these steps must be repeated after project setup. Please refer to `train.py` for more information.

## Scripts

The `crypto_data_cleanup.py` script takes the raw crypto data from Bitfinex and converts it into a single dataset.

```
python crypto_data_cleanup.py <path> <out> <start> <end> [--interval <int>]
```

The script accepts the following arguments:

```
path            The path to the folder containing the source .csv files.
out             The path to the output .csv file to which the results will be written.
start           The start UNIX timestamp in ms for the data collection.
end             The end UNIX timestamp in ms for the data collection.

--interval      The interval in which data will be samples in ms.
```
