import pandas as pd
import sys
import os

from hmmscan.DataLoader import AELoader
from hmmscan.distributions.BinomialModel import BinomialModel
from hmmscan.distributions.BinomialMixtureModel import BinomialMixtureModel
from hmmscan.distributions.BinomialDenseHMM import BinomialDenseHMM
from hmmscan.utils.load_data_utils import write_result_csv, write_data_csv
from hmmscan.utils.scanner_utils import report_model_dl_data, convert_output_dict_to_df


def initialize_model(n_states, n_components, rseed):
    if n_states == 1:
        if n_components == 1:
            return BinomialModel(n=100000, random_state=rseed)
        else:
            return BinomialMixtureModel(
                n=100000,
                distributions=n_components * [BinomialModel(n=100000, random_state=rseed)],
                random_state=rseed,
                init_type="random",
                verbose=False,
            )
    else:
        states = []
        for _ in range(n_states):
            if n_components == 1:
                states.append(BinomialModel(n=100000, random_state=rseed))
            else:
                states.append(
                    BinomialMixtureModel(
                        n=100000,
                        distributions=n_components * [BinomialModel(n=100000, random_state=rseed)],
                        random_state=rseed,
                        init_type="random",
                        verbose=False,
                    )
                )
        return BinomialDenseHMM(n=100000, distributions=states, random_state=rseed, init_type="random", verbose=False)


def run_use_case(directory, sequence_name, ae_type, lot_size_type, grid_index, batch_index, output_subdir):
    # Get the n_states and n_components
    n_states: int = grid_index % 4 + 1
    n_components: int = grid_index // 4 + 1

    output_fname: str = f"{sequence_name}_{ae_type}_s{str(n_states)}c{n_components}_batch{batch_index}"

    # Create the data loader
    dl = AELoader(directory, sequence_name, ae_type, lot_size_type, gap=1)
    write_data_csv(dl.filled_df, f"gap_filled/{ae_type}_{lot_size_type}/gap_1", sequence_name)

    output = {}
    for rseed in [2702 + 5 * batch_index + i for i in range(5)]:
        # Initialize the model
        model = initialize_model(n_states, n_components, 1000 * rseed)

        # Fit the model
        model.fit_loader(dl)

        # Output
        output: dict = report_model_dl_data(model, dl, None, output)

    # Convert the output to a dataframe
    # Save the output

    write_result_csv(
        convert_output_dict_to_df(output, max_s=4, max_c=9),
        os.path.join("scans", "use_case", "random_initializations", output_subdir),
        output_fname,
    )


def main():
    # There is going to be an index that maps to a point in the (n_states, n_components) grid
    # Assumed that the grid is (1, ..., 4) states and (1,...,9) components
    # This can be used by Slurm array index

    # Accept command line arguments
    directory: str = sys.argv[1]
    sequence_name: str = sys.argv[2]
    ae_type: str = sys.argv[3]
    lot_size_type: str = sys.argv[4]
    input_grid_index: int = int(sys.argv[5])
    input_batch_index: int = int(sys.argv[6])  # 0-9 is intended
    output_subdir: str = sys.argv[7]  # 'by_date_ex_iqr' for paper results

    if input_grid_index == -1:
        # Run all the grid indexes
        for grid_index in range(36):
            if input_batch_index == -1:
                for batch_index in range(10):
                    print(f"Starting grid index {grid_index} and batch index {batch_index}")
                    run_use_case(
                        directory, sequence_name, ae_type, lot_size_type, grid_index, batch_index, output_subdir
                    )
            else:
                print(f"Starting grid index {grid_index} and batch index {input_batch_index}")
                run_use_case(
                    directory, sequence_name, ae_type, lot_size_type, grid_index, input_batch_index, output_subdir
                )
    else:
        if input_batch_index == -1:
            for batch_index in range(10):
                print(f"Starting grid index {input_grid_index} and batch index {batch_index}")
                run_use_case(
                    directory, sequence_name, ae_type, lot_size_type, input_grid_index, batch_index, output_subdir
                )
            else:
                print(f"Starting grid index {input_grid_index} and batch index {input_batch_index}")
                run_use_case(
                    directory,
                    sequence_name,
                    ae_type,
                    lot_size_type,
                    input_grid_index,
                    input_batch_index,
                    output_subdir,
                )


if __name__ == "__main__":
    main()
