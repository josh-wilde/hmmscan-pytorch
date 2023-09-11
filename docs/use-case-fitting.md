# Use Case Fitting Documentation

This file contains instructions for Parameter Estimation when applying the HMMScan method to the use case dataset with lot size-weighted likelihoods as in Section S7 of Online Resource 1 of the paper "Development and Application of a Data-Driven Signal Detection Method for Surveillance of Adverse Event Variability Across Manufacturing Lots of Biologics".
The instructions can be used either to replicate the use case in the supplement results or to run HMMScan on a new sequence.

## Required Inputs

1. `sequence_name`: string, name of lot sequence (e.g., `dfa_by_date_ex_iqr_outliers`).
2. `ae_type`: string, name of AE type. This is `serious` for paper result replication.
3. `lot_size_type`: string, name of lot size type. This is `lot_size_doses` for paper result replication.
4. `output_subdir`: string, name of subdirectory of `ae-project/results/use_case/random_initializations` to store the HMM fitting results. This is `by_date_ex_iqr_lotsize` for paper result replication.

## 1. Input Data Setup

Ensure that the file `ae-project/data/use_case/[sequence_name].csv` has been created. If providing your own sequence, not replicating the paper results, then see the [User-Provided Input Data](new-use-case-data.md) for details.

### Paper Result Replication

Ensure that the [`ae-project` repository](https://doi.org/10.17632/zzd5vbj7yn.1) has been downloaded locally (see the repo readme file [here](../README.md) for instructions).
If this repository is downloaded, then the necessary input data files will already be available.

## 2. Run model fitting on the Engaging cluster

From the top level of this directory on Engaging, run the following command:

`sbatch --array=0-9 --time=0-00:30:00 hmmscan/cluster/scan_use_case.sh use_case [sequence_name] [ae_type] [lot_size_type] -1 [output_subdir]`

### Paper Result Replication

The commands above must be run for each of the following `sequence_name`s:

1. `dfa_by_date_ex_iqr_outliers`
2. `dfb_by_date_ex_iqr_outliers`
3. `dfc_by_date_ex_iqr_outliers`

Here is the command:

`sbatch --array=0-9 --time=0-00:30:00 hmmscan/cluster/scan_use_case.sh use_case [sequence_name] serious lot_size_doses -1 by_date_ex_iqr_lotsize`

Step 2 generates a file for each combination of `sequence_name`, `ae_type`, number of hidden states, number of mixture components, and batches of random initializations in `ae-project/results/use_case/random_initializations/[output_subdir]`.

## 3. Aggregate the parameter estimation information

On Engaging, run the following commands in an interactive session:

1. Load `R 4.1`: `module load R/4.1.0`.
2. Run `aggregate_scan_results.R`: `Rscript hmmscan/scripts/scans/aggregate_scan_results.R scans/use_case/random_initializations/[output_subdir]`

This script will generate a CSV file called `ae-project/results/scans/use_case/random_initializations/[output_subdir].csv`.

### Paper Result Replication

Use this command: `Rscript hmmscan/scripts/scans/aggregate_scan_results.R scans/use_case/random_initializations/by_date_ex_iqr_lotsize`

## 4. Choose the best random initialization for each model structure

On Engaging, run the following commands in an interactive session:

1. Load `R 4.1`: `module load R/4.1.0`.
2. Run `get_best_initializations.R`: `Rscript hmmscan/scripts/scans/get_best_initializations.R [output_subdir].csv`

This script will generate a CSV file called `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv`.

### Paper Result Replication

Use this command: `Rscript hmmscan/scripts/scans/get_best_initializations.R by_date_ex_iqr_lotsize.csv`

## 5. Run state prediction

For this section, you will need to look at the CSV file generated in step 4 and find the best number of states and mixture components for each `sequence_name` and `ae_type` combination.
The best structure is referred to below as `best_n_states` and `best_n_mix_comps`.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `scripts/state_prediction.py`: `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/[output_subdir].csv use_case [sequence_name] [ae_type] [lot_size_type] [best_n_states] [best_n_mix_comps]`.

This script will generate a file in `ae-project/results/state_prediction/use_case` for each `sequence_name`, `ae_type`, `best_n_states`, and `best_n_mix_comps`.

### Paper Result Replication

Run these commands:

1. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr_lotsize.csv use_case dfa_by_date_ex_iqr_outliers serious lot_size_doses 3 3`
2. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr_lotsize.csv use_case dfb_by_date_ex_iqr_outliers serious lot_size_doses 3 2`
3. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr_lotsize.csv use_case dfc_by_date_ex_iqr_outliers serious lot_size_doses 2 2`

## 6. Visualize the results

It is probably easiest to generate the necessary plots locally off Engaging. To do so, copy `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv` and the contents of `ae-project/results/state_prediction/use_case` into the same relative file locations in your local version of `ae-project`.

Then, you can run `hmmscan/scripts/viz/bic.R` to view the BICs of the HMM model candidates, and `hmmscan/scripts/viz/best_model_dists_and_predictions.R` to view the characteristics of the models with the best BICs.

If you are using your own lot sequence and not replicating the paper results, then you will need to adjust these visualization scripts.
