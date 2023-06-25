from typing import Optional, Union
import torch
import pandas as pd
import numpy as np

from pomegranate._utils import eps

from hmmscan.distributions.BinomialModel import BinomialModel
from hmmscan.distributions.BinomialMixtureModel import BinomialMixtureModel
from hmmscan.distributions.BinomialDenseHMM import BinomialDenseHMM


def initialize_bin(info, state_id, component_id, init_file_name, init_file_row):
    info = info.dropna()
    n = float(info["n"])

    try_param = f"s{str(int(state_id))}_{str(int(component_id))}_param"
    if try_param in info and not np.isnan(info[try_param]):
        p = float(info[try_param])
    else:
        p = eps

    return BinomialModel(n=n, p=p, init_file_name=init_file_name, init_file_row=init_file_row)


def initialize_mix(info, state_id, init_file_name, init_file_row):
    info = info.dropna()
    n_mix_comps = int(info["n_mix_comps"])
    n = float(info["n"])

    # Parameters for emission distribution
    components = []
    weights = []
    for mix_comp in range(n_mix_comps):
        try_param = f"s{str(int(state_id))}_{str(mix_comp)}_param"
        try_weight = f"s{str(int(state_id))}_{str(mix_comp)}_wt"
        if try_param in info and not np.isnan(info[try_param]):
            wt = float(info[try_weight])
        else:
            wt = 0.0

        components = components + [initialize_bin(info, state_id, mix_comp, init_file_name, init_file_row)]
        weights = weights + [wt]

    return BinomialMixtureModel(
        n=n, distributions=components, priors=weights, init_file_name=init_file_name, init_file_row=init_file_row
    )


def initialize_hmm(info, init_file_name, init_file_row):
    info = info.dropna()

    n_states = int(info["n_states"])
    n_mix_comps = int(info["n_mix_comps"])
    n = float(info["n"])

    dists = []  # list of binomial distributions
    trans_mat = torch.zeros((n_states, n_states))
    starts = torch.zeros(n_states)
    ends = torch.zeros(n_states)

    for state in range(n_states):
        # Parameters for emission distribution
        if n_mix_comps > 1:
            dists = dists + [initialize_mix(info, state, init_file_name, init_file_row)]
        else:
            dists = dists + [initialize_bin(info, state, 0, init_file_name, init_file_row)]

        # Transition matrix
        for to_state in range(n_states):
            try_trans_entry = "s" + str(state) + "_s" + str(to_state)
            if try_trans_entry in info:
                trans_mat[state, to_state] = float(info[try_trans_entry])

        # Starts
        try_start_entry = "s" + str(state) + "_log_start"
        if try_start_entry in info:
            starts[state] = float(np.exp(info[try_start_entry]))

    return BinomialDenseHMM(
        n=n,
        distributions=dists,
        edges=trans_mat,
        starts=starts,
        ends=ends,
        init_file_name=init_file_name,
        init_file_row=init_file_row,
    )


def _initialize_full_data_model(
    init_data: pd.Series, init_file_row: Optional[str] = None, init_file_name: Optional[str] = None
):
    n_states = int(init_data["n_states"])
    n_comps = int(init_data["n_mix_comps"])

    if n_states == 1 and n_comps == 1:
        return initialize_bin(
            init_data, state_id=0, component_id=0, init_file_name=init_file_name, init_file_row=init_file_row
        )
    elif n_states == 1 and n_comps > 1:
        return initialize_mix(init_data, state_id=0, init_file_name=init_file_name, init_file_row=init_file_row)
    else:
        return initialize_hmm(init_data, init_file_name=init_file_name, init_file_row=init_file_row)


def _initialize_minimal_data_model(
    init_data: Union[dict, pd.Series], init_type: str, verbose: bool, max_iter: int, tol: float
):
    n_states: int = init_data["n_states"]
    n_mix_comps: int = init_data["n_mix_comps"]
    n: int = init_data["n"]
    rseed: int = init_data["random_state"]
    rand_init_ub: int = init_data["random_init_upper_bound"]

    if n_states == 1 and n_mix_comps == 1:
        return BinomialModel(n=n, init_type=init_type, random_state=rseed, random_init_upper_bound=rand_init_ub)
    elif n_states == 1 and n_mix_comps > 1:
        distributions = n_mix_comps * [
            BinomialModel(n=n, init_type=init_type, random_state=rseed, random_init_upper_bound=rand_init_ub)
        ]
        return BinomialMixtureModel(
            n=n,
            distributions=distributions,
            init_type=init_type,
            random_state=rseed,
            random_init_upper_bound=rand_init_ub,
            verbose=verbose,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        # Initialize a binomial HMM
        if n_mix_comps == 1:
            distributions = n_states * [
                BinomialModel(n=n, init_type=init_type, random_state=rseed, random_init_upper_bound=rand_init_ub)
            ]
        else:
            mixture_distribution = BinomialMixtureModel(
                n=n,
                distributions=n_mix_comps
                * [BinomialModel(n=n, random_state=rseed, random_init_upper_bound=rand_init_ub)],
                init_type=init_type,
                random_state=rseed,
                random_init_upper_bound=rand_init_ub,
                verbose=verbose,
                max_iter=max_iter,
                tol=tol,
            )
            distributions = n_states * [mixture_distribution]

        return BinomialDenseHMM(
            n=n,
            distributions=distributions,
            init_type=init_type,
            random_state=rseed,
            random_init_upper_bound=rand_init_ub,
            verbose=verbose,
            max_iter=max_iter,
            tol=tol,
        )


def initialize_model(
    init_data: Union[dict, pd.Series],
    init_type: str,
    init_data_type: str,
    verbose: bool = True,
    max_iter: int = int(1e8),
    tol: float = 1e-9,
    init_file_row: str = None,
    init_file_name: str = None,
):
    # Can either initialize from a pandas Series that has all of the individual parameters
    # Or minimal init that does random initialization
    if init_data_type == "fully_specified":
        return _initialize_full_data_model(init_data, init_file_row, init_file_name)
    else:
        return _initialize_minimal_data_model(init_data, init_type, verbose, max_iter, tol)
