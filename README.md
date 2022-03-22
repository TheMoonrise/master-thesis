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
--resume        Resumes a previously started training.
```

## Data

The data for the training is sourced from the Bitfinex api. The data consists of five-minute snapshots for the ten crypto currencies with the highest market capitalization available on Bitfinex for the year 2021. The data interval in UNIX milliseconds stretches from `1609459200000` to `1640995200000`.

## Scripts

The `crypto_data_cleanup.py` script takes the raw crypto data from Bitfinex and converts it into a single dataset. The script accepts the following arguments:

```
path            The path to the folder containing the source .csv files.
out             The path to the output .csv file to which the results will be written.
start           The start UNIX timestamp in ms for the data collection.
end             The end UNIX timestamp in ms for the data collection.
--interval      The interval in which data will be samples in ms.
```
