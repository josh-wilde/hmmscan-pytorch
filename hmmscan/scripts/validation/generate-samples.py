import pandas as pd
import torch
import argparse
import os
import sys
from typing import Optional

from hmmscan.utils.load_data_utils import get_ae_path, load_data
from hmmscan.utils.model_initialization_utils import initialize_model


def save_samples(
    n_samples: int,
    init_series: pd.Series,
    init_file_row: int,
    init_file_name: str,
    samples_output_fpath: str,
    paths_output_fpath: str,
    lot_size_path: Optional[torch.Tensor] = None,
):
    # Use the series to instantiate the model and generate samples and state paths
    m = initialize_model(
        init_data=init_series,
        init_type="",
        init_data_type="fully_specified",
        init_file_row=init_file_row,
        init_file_name=init_file_name,
    )
    len_sample: int = int(init_series["n_lots_in_seqs"])
    samples: list[torch.Tensor]
    paths: list[torch.Tensor]
    samples, paths = m.get_model_samples(n=n_samples, length=len_sample, lot_size_path=lot_size_path, path=True)

    # Write the samples and paths out to a CSV
    pd.DataFrame(torch.stack(samples, dim=0).numpy()).to_csv(samples_output_fpath)
    pd.DataFrame(torch.stack(paths, dim=0).numpy()).to_csv(paths_output_fpath)


def run_use_case_sampling(init_file_name: str, sequence_name: str, ae_type: str, lot_size_type: str, n_samples: int):
    # This function needs to define the rows of the init_file_name that generate samples
    # And loop over these rows
    # Calling save_samples() in each loop iteration

    # Create directories to save samples and state paths to
    samples_output_dir = os.path.join(get_ae_path("validation"), "use_case", "samples")
    paths_output_dir = os.path.join(get_ae_path("validation"), "use_case", "state_paths")
    os.makedirs(samples_output_dir, exist_ok=True)
    os.makedirs(paths_output_dir, exist_ok=True)

    # Get the initialization series
    init_df_fstubpath: str = os.path.join("results", init_file_name)
    init_df: pd.DataFrame = pd.read_csv(os.path.join(get_ae_path("results"), init_file_name))
    init_df: pd.DataFrame = init_df.loc[
        (init_df["sequence_name"] == sequence_name)
        & (init_df["ae_type"] == ae_type)
        & (init_df["lot_size_type"] == lot_size_type),
        :,
    ]

    # Need to grab the lot size path
    data_df = load_data(f"gap_filled/{ae_type}_{lot_size_type}/gap_1/", f"{sequence_name}.csv")
    lot_size_path: torch.Tensor = torch.tensor(data_df["lot_size"].values, dtype=torch.int)

    for init_file_row, init_series in init_df.iterrows():
        # Create file name for samples and paths
        fname: str = (
            f"{sequence_name}_{ae_type}_s{str(init_series['n_states'])}_c{str(init_series['n_mix_comps'])}.csv"
        )
        print(f"Generating samples for {fname}")

        # This function will initialize the model, grab the samples, and save the samples
        save_samples(
            n_samples,
            init_series,
            init_file_row,
            init_df_fstubpath,
            os.path.join(samples_output_dir, fname),
            os.path.join(paths_output_dir, fname),
            lot_size_path=lot_size_path,
        )


def run_simulation(init_file_name: str, exp_ids: list[str], exp_id_min: int, exp_id_max: int, n_samples: int) -> None:
    # Create the experiment list
    if len(exp_ids) > 0:
        exp_list: list[str] = exp_ids
    elif -1 < exp_id_min <= exp_id_max and exp_id_max > -1:
        exp_list: list[int] = list(range(exp_id_min, exp_id_max + 1))
    else:
        raise ValueError(f"Problem with experiments. exp_ids: {exp_ids}. min, max: {exp_id_min}, {exp_id_max}.")

    # Create directories to save samples and state paths to
    samples_output_dir = os.path.join(get_ae_path("validation"), "simulation", "samples")
    paths_output_dir = os.path.join(get_ae_path("validation"), "simulation", "state_paths")
    os.makedirs(samples_output_dir, exist_ok=True)
    os.makedirs(paths_output_dir, exist_ok=True)

    for eid in exp_list:
        # Create file name for samples and paths
        fname: str = f"eid_{eid:04d}.csv"

        # Get the initialization series
        init_df_fstubpath: str = os.path.join("validation", "simulation", init_file_name)
        init_df: pd.DataFrame = pd.read_csv(os.path.join(get_ae_path("validation"), "simulation", init_file_name))
        init_series: pd.Series = init_df[init_df["eid"] == eid].iloc[0]
        init_file_row: int = init_df[init_df["eid"] == eid].index[0]

        # This function will initialize the model, grab the samples, and save the samples
        save_samples(
            n_samples,
            init_series,
            init_file_row,
            init_df_fstubpath,
            os.path.join(samples_output_dir, fname),
            os.path.join(paths_output_dir, fname),
        )
        print(f"Finished {eid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_type", type=str, default="simulation")
    parser.add_argument("--init_file_name", type=str, default="full_simulation_input.csv")
    parser.add_argument("--exp_ids", type=str, nargs="*", default=[])
    parser.add_argument("--exp_id_min", type=int, default=-1)
    parser.add_argument("--exp_id_max", type=int, default=-1)

    parser.add_argument("--sequence_name", type=str, default="dfa_by_date_ex_iqr_outliers")
    parser.add_argument("--ae_type", type=str, default="serious")
    parser.add_argument("--lot_size_type", type=str, default="lot_size_doses")

    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    if args.validation_type == "simulation":
        run_simulation(args.init_file_name, args.exp_ids, args.exp_id_min, args.exp_id_max, args.n_samples)
    elif args.validation_type == "use_case":
        run_use_case_sampling(
            args.init_file_name, args.sequence_name, args.ae_type, args.lot_size_type, args.n_samples
        )
    else:
        raise ValueError(f"input type must be simulation or use_case, not {args.validation_type}.")


if __name__ == "__main__":
    main()
