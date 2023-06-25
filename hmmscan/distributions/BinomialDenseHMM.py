from typing import Union, Optional
import torch
import numpy as np
import time
import sys

from pomegranate.distributions._distribution import Distribution
from pomegranate.hmm import DenseHMM
from pomegranate.hmm._base import _check_inputs
from pomegranate._utils import (
    _cast_as_parameter,
    _cast_as_tensor,
    _check_parameter,
    partition_sequences,
    _update_parameter,
    eps,
)
from pomegranate.kmeans import KMeans
from .BinomialModel import BinomialModel
from .BinomialMixtureModel import BinomialMixtureModel
from .ReportingMixin import ReportingMixin


def _check_inputs(model, X, emissions, priors):
    if X is None and emissions is None:
        raise ValueError("Must pass in one of `X` or `emissions`.")

    emissions = _check_parameter(_cast_as_tensor(emissions), "emissions", ndim=3)
    if emissions is None:
        emissions = model._emission_matrix(X, priors=priors)

    return emissions


class BinomialDenseHMM(ReportingMixin, DenseHMM):
    def __init__(
        self,
        n: int,
        distributions,
        edges=None,
        starts=None,
        ends=None,
        init_type: str = "kmeans",
        random_state: int = 0,
        random_init_upper_bound: float = 0.01,
        max_iter: int = int(1e8),
        tol: float = 1e-9,
        init_file_row: Optional[int] = None,
        init_file_name: Optional[str] = None,
        inertia: float = 0,
        frozen: bool = False,
        check_data: bool = True,
        verbose: bool = True,
    ):
        # Initialize the dense HMM
        DenseHMM.__init__(
            self,
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=max_iter,
            tol=tol,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
            random_state=random_state,
            verbose=verbose,
        )

        # Initialize metadata
        self.name = "BinomialDenseHMM"
        self.random_state = random_state
        self.n = n
        self.n_states = len(self.distributions)
        self.n_mix_comps = self.distributions[0].n_mix_comps
        self.init_type = init_type
        self.init_file_name = init_file_name
        self.init_file_row = init_file_row
        self.random_init_upper_bound = random_init_upper_bound
        self.initialize_metadata()

        # Need to hack custom initialization
        if self.init_type == "custom":
            self._initialized = False
            for d in self.distributions:
                d._initialized = False
                if d.name == "GeneralMixtureModel":
                    for dd in d.distributions:
                        dd._initialized = False

    def _initialize(self, X):
        if self.init_type == "random":
            self._random_initialize(X.shape[1] - 1)
        elif self.init_type == "kmeans":
            self._kmeans_initialize(X)

        self.d = X.shape[-1] - 1
        Distribution._initialize(self, X.shape[-1] - 1)
        self._initialized = True
        self._reset_cache()  # this function from DenseHMM is good
        # print(f"Initialized dense HMM. State dict: {self.state_dict()}")
        return self.state_dict()

    def _kmeans_initialize(self, X, sample_weight=None):
        n = self.n_distributions
        if self.starts is None:
            self.starts = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))

        if self.ends is None:
            self.ends = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))

        _init = all(d._initialized for d in self.distributions)

        if X is not None and not _init:
            if sample_weight is None:
                sample_weight = torch.ones(1, dtype=self.dtype, device=self.device).expand(X.shape[0], 1)
            else:
                sample_weight = _check_parameter(
                    _cast_as_tensor(sample_weight).reshape(-1, 1),
                    "sample_weight",
                    min_value=0.0,
                    ndim=1,
                    shape=(len(X),),
                    check_parameter=self.check_data,
                ).reshape(-1, 1)

            model = KMeans(self.n_distributions, init="random", max_iter=1, random_state=self.random_state).to(
                self.device
            )

            y_hat = model.fit_predict(X, sample_weight=sample_weight)

            if self.distributions[0].name == "BinomialModel":
                for i in range(self.n_distributions):
                    self.distributions[i] = BinomialModel(
                        n=self.n,
                        random_state=self.random_state,
                        random_init_upper_bound=self.random_init_upper_bound,
                        inertia=self.inertia,
                        frozen=self.frozen,
                        check_data=self.check_data,
                        init_file_row=self.init_file_row,
                        init_file_name=self.init_file_name,
                    )
                    if X[y_hat == i].shape[0] > 0:
                        self.distributions[i].fit(X[y_hat == i], sample_weight=sample_weight[y_hat == i])
                    else:
                        self.distributions[i]._initialize(X.shape[-1] - 1)
            else:
                for i in range(self.n_distributions):
                    state_dists = [
                        BinomialModel(
                            n=self.n,
                            random_state=self.random_state,
                            random_init_upper_bound=self.random_init_upper_bound,
                            inertia=self.inertia,
                            frozen=self.frozen,
                            check_data=self.check_data,
                            init_file_row=self.init_file_row,
                            init_file_name=self.init_file_name,
                        )
                        for _ in range(self.n_mix_comps)
                    ]
                    self.distributions[i] = BinomialMixtureModel(
                        n=self.n,
                        distributions=state_dists,
                        init_type="kmeans",
                        random_state=self.random_state,
                        random_init_upper_bound=self.random_init_upper_bound,
                        inertia=self.inertia,
                        frozen=self.frozen,
                        check_data=self.check_data,
                        init_file_row=self.init_file_row,
                        init_file_name=self.init_file_name,
                        verbose=self.verbose,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
                    if X[y_hat == i].shape[0] > 0:
                        self.distributions[i].fit(X[y_hat == i], sample_weight=sample_weight[y_hat == i])
                    else:
                        self.distributions[i]._initialize(X[y_hat == i], sample_weight=sample_weight[y_hat == i])

            self.distributions = torch.nn.ModuleList(self.distributions)

        # Transition matrix
        if self.edges == None:
            self.edges = _cast_as_parameter(torch.log(torch.ones(n, n, dtype=self.dtype, device=self.device) / n))

    def _random_initialize(self, d):
        # This takes X as a 2d tensor, (length, dim), that concatenates all of sequences
        n = self.n_distributions

        # Starts and ends
        if self.starts is None:
            self.starts = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))

        if self.ends is None:
            self.ends = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device) / n))

        # Components
        _init = all(d._initialized for d in self.distributions)

        if not _init:
            torch.manual_seed(self.random_state)
            rand_ps = [
                torch.rand(self.n_mix_comps, dtype=self.dtype, device=self.device)
                * self.random_init_upper_bound
                * 500
                / self.n
                for _ in range(self.n_states)
            ]

            if self.distributions[0].name == "BinomialModel":
                for i in range(self.n_distributions):
                    self.distributions[i] = BinomialModel(
                        n=self.n,
                        p=rand_ps[i][0],
                        random_state=self.random_state,
                        random_init_upper_bound=self.random_init_upper_bound,
                        inertia=self.inertia,
                        frozen=self.frozen,
                        check_data=self.check_data,
                        init_file_row=self.init_file_row,
                        init_file_name=self.init_file_name,
                    )
            else:
                for i in range(self.n_distributions):
                    state_dists = [
                        BinomialModel(
                            n=self.n,
                            p=rand_ps[i][j],
                            random_state=self.random_state,
                            random_init_upper_bound=self.random_init_upper_bound,
                            inertia=self.inertia,
                            frozen=self.frozen,
                            check_data=self.check_data,
                            init_file_row=self.init_file_row,
                            init_file_name=self.init_file_name,
                        )
                        for j in range(self.n_mix_comps)
                    ]
                    self.distributions[i] = BinomialMixtureModel(
                        n=self.n,
                        distributions=state_dists,
                        random_state=self.random_state,
                        random_init_upper_bound=self.random_init_upper_bound,
                        inertia=self.inertia,
                        frozen=self.frozen,
                        check_data=self.check_data,
                        init_file_row=self.init_file_row,
                        init_file_name=self.init_file_name,
                        verbose=self.verbose,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )

        self.distributions = torch.nn.ModuleList(self.distributions)

        # Transition matrix
        if self.edges == None:
            self.edges = _cast_as_parameter(torch.log(torch.ones(n, n, dtype=self.dtype, device=self.device) / n))

    def get_model_samples(self, n=1, length=1, lot_size_path=None, path=False, random_state=0):
        # Initial settings before sending to the DenseHMM sampler
        self.random_state = np.random.RandomState(random_state)
        self.sample_length = length
        self.return_sample_paths = path
        torch.manual_seed(random_state)
        raw_samples = self.sample(n=n, lot_size_path=lot_size_path)

        return raw_samples

    def sample(self, n, lot_size_path=None):
        """Sample from the probability distribution.

        This method will return `n` samples generated from the underlying
        probability distribution. Because a HMM describes variable length
        sequences, a list will be returned where each element is one of
        the generated sequences.


        Parameters
        ----------
        n: int
            The number of samples to generate.


        Returns
        -------
        X: list of torch.tensor, shape=(n,)
            A list of randomly generated samples, where each sample of
            size (length, self.d).
        """

        if self.sample_length is None and self.ends is None:
            raise ValueError("Must specify a length or have explicit " + "end probabilities.")

        distributions, emissions = [], []

        edge_probs = self.get_trans_mat().numpy()
        starts = torch.exp(self.starts).numpy()

        if np.any(starts < 0) and np.all(starts >= -0.01):
            starts[starts < 0] = eps
            starts = starts / np.sum(starts)
        elif np.any(starts < 0):
            print(f"edge_probs: {edge_probs}")
            print(f"starts: {starts}")
            raise ValueError("Start probabilities must be non-negative.")

        for i in range(n):
            print(f"Sampling sequence {i + 1}/{n}")
            node_i = torch.tensor(self.random_state.choice(self.n_distributions, p=starts))
            lot_size_i = lot_size_path[0].unsqueeze(0) if lot_size_path is not None else None
            emission_i = self.distributions[node_i].sample(n=1, lot_size_path=lot_size_i)
            distributions_, emissions_ = [node_i], [emission_i]

            for j in range(1, self.sample_length or int(1e8)):
                node_i = torch.tensor(self.random_state.choice(self.n_distributions, p=edge_probs[node_i]))
                lot_size_i = lot_size_path[j].unsqueeze(0) if lot_size_path is not None else None
                emission_i = self.distributions[node_i].sample(n=1, lot_size_path=lot_size_i)

                distributions_.append(node_i)
                emissions_.append(emission_i)

            distributions.append(torch.vstack(distributions_).squeeze(1))
            emissions.append(torch.vstack(emissions_).squeeze(1))

        if self.return_sample_paths == True:
            return emissions, distributions
        return emissions

    def _emission_matrix(self, X, priors=None):
        """Return the emission/responsibility matrix.

        This method returns the log probability of each example under each
        distribution contained in the model with the log prior probability
        of each component added.


        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
            A set of examples to evaluate.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.


        Returns
        -------
        e: torch.Tensor, shape=(-1, len, self.k)
            A set of log probabilities for each example under each distribution.
        """
        # print("In densehmm._emission_matrix()")
        # print(f"prior: {priors}")
        X = _check_parameter(
            _cast_as_tensor(X), "X", ndim=3, shape=(-1, -1, self.d + 1), check_parameter=self.check_data
        )

        n, k, _ = X.shape
        X = X.reshape(n * k, self.d + 1)

        priors = _check_parameter(
            _cast_as_tensor(priors),
            "priors",
            ndim=3,
            shape=(n, k, self.k),
            min_value=0.0,
            max_value=1.0,
            value_sum=1.0,
            value_sum_dim=-1,
            check_parameter=self.check_data,
        )

        if not self._initialized:
            self._initialize(X)

        e = torch.empty((k, self.k, n), dtype=self.dtype, requires_grad=False, device=self.device)

        for i, node in enumerate(self.distributions):
            logp = node.log_probability(X)
            # if torch.isnan(logp).any():
            #    print(f"Nan in logp for node {i}")
            #    print(f"node {i} state_dict: {node.state_dict()}")
            if isinstance(logp, torch.masked.MaskedTensor):
                logp = logp._masked_data

            e[:, i] = logp.reshape(n, k).T

        e = e.permute(2, 0, 1)

        if priors is not None:
            e += torch.log(priors)

        return e

    def fit_loader(self, dataloader):
        train_data = torch.cat(dataloader.sequences, dim=0)
        self._initialize(train_data)  # 2d tensor, (length, dim + 1)
        super().fit(train_data.unsqueeze(0))
        # self._fit(train_data.unsqueeze(0))

    def _fit(self, X, sample_weight=None, priors=None):
        """Fit the model to sequences with optional weights and priors.

        This method implements the core of the learning process. For hidden
        Markov models, this is a form of EM called "Baum-Welch" or "structured
        EM". This iterative algorithm will proceed until converging, either
        according to the threshold set by `tol` or until the maximum number
        of iterations set by `max_iter` has been hit.

        This method is largely a wrapper around the `summarize` and
        `from_summaries` methods. It's primary contribution is serving as a
        loop around these functions and to monitor convergence.

        Unlike other HMM methods, this method can handle variable length
        sequences by accepting a list of tensors where each tensor has a
        different sequence length. Then, summarization is done on each tensor
        sequentially. This will provide an exact update as if the entire data
        set was seen at the same time but will allow batched operations to be
        performed on each variable length tensor.


        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
            A set of examples to evaluate. Because sequences can be variable
            length, there are three ways to format the sequences.

                1. Pass in a tensor of shape (n, length, dim), which can only
                be done when each sequence is the same length.

                2. Pass in a list of 3D tensors where each tensor has the shape
                (n, length, dim). In this case, each tensor is a collection of
                sequences of the same length and so sequences of different
                lengths can be trained on.

                3. Pass in a list of 2D tensors where each tensor has the shape
                (length, dim). In this case, sequences of the same length will
                be grouped together into the same tensor and fitting will
                proceed as if you had passed in data like way 2.

        sample_weight: list, numpy.ndarray, torch.Tensor or None, optional
            A set of weights for the examples. These must follow the same format
            as X.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Must be formatted in the same
            shape as X. Default is None.


        Returns
        -------
        self
        """

        X, sample_weight, priors = partition_sequences(X, sample_weight=sample_weight, priors=priors)

        # Initialize by concatenating across sequences
        if not self._initialized:
            X_ = torch.cat(X, dim=1)

            if sample_weight is None:
                self._initialize(X_)
            else:
                w_ = torch.cat(sample_weight, dim=1)
                self._initialize(X_, sample_weight=w_)

        logp, last_logp = None, None
        for i in range(self.max_iter):
            start_time = time.time()

            # Train loop across all tensors
            logp = 0
            for j, X_ in enumerate(X):
                w_ = None if sample_weight is None else sample_weight[j]
                p_ = None if priors is None else priors[j]

                logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()

            # Calculate and check improvement and optionally print it
            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time

                if self.verbose:
                    print("[{}] Improvement: {}, Time: {:4.4}s".format(i, improvement, duration))

                if improvement < self.tol:
                    self._reset_cache()
                    return self

            last_logp = logp
            self.from_summaries()

        # Calculate for the last iteration
        if self.verbose:
            logp = 0
            for j, X_ in enumerate(X):
                w_ = None if sample_weight is None else sample_weight[j]
                p_ = None if priors is None else priors[j]

                logp += self.summarize(X_, sample_weight=w_, priors=p_).sum()

            improvement = logp - last_logp
            duration = time.time() - start_time

            print("[{}] Improvement: {}, Time: {:4.4}s".format(i + 1, improvement, duration))

        self._reset_cache()
        return self

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

        This method calculates the sufficient statistics from optionally
        weighted data and adds them to the stored cache. The examples must be
        given in a 2D format. Sample weights can either be provided as one
        value per example or as a 2D matrix of weights for each feature in
        each example.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)
            A set of examples to summarize.

        sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
            A set of weights for the examples. This can be either of shape
            (-1, length, self.d) or a vector of shape (-1,). Default is ones.

        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.n_distributions)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities).


        Returns
        -------
        logp: torch.Tensor, shape=(-1,)
            The log probability of each example.
        """

        # print("In dense Hmm summarize")
        X = _check_parameter(
            _cast_as_tensor(X), "X", ndim=3, shape=(-1, -1, self.d + 1), check_parameter=self.check_data
        )
        emissions = _check_inputs(self, X, emissions, priors)

        if sample_weight is None:
            sample_weight = torch.ones(1, device=self.device).expand(emissions.shape[0], 1)
        else:
            sample_weight = _check_parameter(
                _cast_as_tensor(sample_weight),
                "sample_weight",
                min_value=0.0,
                ndim=1,
                shape=(emissions.shape[0],),
                check_parameter=self.check_data,
            ).reshape(-1, 1)

        if not self._initialized:
            self._initialize(X)

        t, r, starts, ends, logps = self.forward_backward(emissions=emissions)

        self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
        self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
        self._xw_sum += torch.sum(t * sample_weight.unsqueeze(-1), dim=0)

        X = X.reshape(-1, X.shape[-1])
        r = torch.exp(r) * sample_weight.unsqueeze(-1)
        for i, node in enumerate(self.distributions):
            w = r[:, :, i].reshape(-1, 1)
            # if torch.isnan(w).any():
            #    print("w has nan")
            #    print(f"w: {w}")
            node.summarize(X, sample_weight=w)

        return logps

    def from_summaries(self):
        for node in self.distributions:
            node.from_summaries()

        if self.frozen:
            return

        node_out_count = torch.sum(self._xw_sum, dim=1, keepdims=True)
        ### node_out_count += self._xw_ends_sum.unsqueeze(1)

        ### ends = torch.log(self._xw_ends_sum / node_out_count[:, 0])
        starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
        edges = torch.log(self._xw_sum / node_out_count)

        ### _update_parameter(self.ends, ends, inertia=self.inertia)
        _update_parameter(self.starts, starts, inertia=self.inertia)
        _update_parameter(self.edges, edges, inertia=self.inertia)
        self._reset_cache()

    def forward_backward(self, X=None, emissions=None, priors=None):
        """Run the forward-backward algorithm on some data.

        Runs the forward-backward algorithm on a batch of sequences. This
        algorithm combines the best of the forward and the backward algorithm.
        It combines the probability of starting at the beginning of the sequence
        and working your way to each observation with the probability of
        starting at the end of the sequence and working your way backward to it.

        A number of statistics can be calculated using this information. These
        statistics are powerful inference tools but are also used during the
        Baum-Welch training process.


        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.

        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.


        Returns
        -------
        transitions: torch.Tensor, shape=(-1, n, n)
            The expected number of transitions across each edge that occur
            for each example. The returned transitions follow the structure
            of the transition matrix and so will be dense or sparse as
            appropriate.

        responsibility: torch.Tensor, shape=(-1, -1, n)
            The posterior probabilities of each observation belonging to each
            state given that one starts at the beginning of the sequence,
            aligns observations across all paths to get to the current
            observation, and then proceeds to align all remaining observations
            until the end of the sequence.

        starts: torch.Tensor, shape=(-1, n)
            The probabilities of starting at each node given the
            forward-backward algorithm.

        ends: torch.Tensor, shape=(-1, n)
            The probabilities of ending at each node given the forward-backward
            algorithm.

        logp: torch.Tensor, shape=(-1,)
            The log probabilities of each sequence given the model.
        """

        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape

        f = self.forward(emissions=emissions)
        b = self.backward(emissions=emissions)

        ### logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)
        logp = torch.logsumexp(f[:, -1], dim=1)

        f_ = f[:, :-1].unsqueeze(-1)
        b_ = (b[:, 1:] + emissions[:, 1:]).unsqueeze(-2)

        t = f_ + b_ + self.edges.unsqueeze(0).unsqueeze(0)
        t = t.reshape(n, l - 1, -1)
        t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
        t = t.reshape(n, int(t.shape[1] ** 0.5), -1)

        starts = self.starts + emissions[:, 0] + b[:, 0]
        starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

        ### ends = self.ends + f[:, -1]
        ends = f[:, -1]
        ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T

        r = f + b
        r = r - torch.logsumexp(r, dim=2).reshape(n, -1, 1)
        return t, r, starts, ends, logp

    def backward(self, X=None, emissions=None, priors=None):
        """Run the backward algorithm on some data.

        Runs the backward algorithm on a batch of sequences. This is not to be
        confused with a "backward pass" when talking about neural networks. The
        backward algorithm is a dynamic programming algorithm that begins at end
        of the sequence and returns the probability, over all paths through the
        model, that result in the alignment of symbol i to node j, working
        backwards.

        Note that, as an internal method, this does not take as input the
        actual sequence of observations but, rather, the emission probabilities
        calculated from the sequence given the model.


        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.

        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.


        Returns
        -------
        b: torch.Tensor, shape=(-1, length, self.n_distributions)
            The log probabilities calculated by the backward algorithm.
        """

        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape

        b = torch.zeros(l, n, self.n_distributions, dtype=self.dtype, device=self.device) + float("-inf")
        ### b[-1] = self.ends
        b[-1] = 0.0

        t_max = self.edges.max()
        t = torch.exp(self.edges.T - t_max)

        for i in range(l - 2, -1, -1):
            p = b[i + 1] + emissions[:, i + 1]
            p_max = torch.max(p, dim=1, keepdims=True).values
            p = torch.exp(p - p_max)

            b[i] = torch.log(torch.matmul(p, t)) + t_max + p_max

        b = b.permute(1, 0, 2)
        return b

    def log_probability(self, X, priors=None):
        """Calculate the log probability of each example.

        This method calculates the log probability of each example given the
        parameters of the distribution. The examples must be given in a 3D
        format.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
            A set of examples to evaluate.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, length, self.d)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities).


        Returns
        -------
        logp: torch.Tensor, shape=(-1,)
            The log probability of each example.
        """

        f = self.forward(X, priors=priors)
        ### return torch.logsumexp(f[:, -1] + self.ends, dim=1)
        return torch.logsumexp(f[:, -1], dim=1)

    def get_log_prob(self, sequences):
        """
        Input: list of tensors containing ae and lot size sequences
        Output: log probability of all of the sequences concatenated
        """
        if len(sequences) > 0:
            return torch.sum(self.log_probability(torch.cat(sequences, dim=0).unsqueeze(0)))
        else:
            return torch.tensor(float("nan"))

    def get_log_prob_array(self, sequences):
        """
        Input: list of tensors containing ae and lot size sequences
        Output: log probability of each of the sequences
        """
        if len(sequences) > 0:
            return torch.tensor([self.log_probability(sequence.unsqueeze(0)) for sequence in sequences])
        else:
            return torch.tensor(float("nan"))

    def get_comp_params(self, threshold=1e-7):
        params = {}
        weights = {}

        for component_idx, component_dist in enumerate(self.distributions):
            component_params = component_dist.get_comp_params(threshold=threshold)
            params[f"s{component_idx}"] = component_params[0]["s0"]
            weights[f"s{component_idx}"] = component_params[1]["s0"]

        return params, weights

    def get_starts(self):
        return self.starts.data

    def get_trans_mat(self):
        # We want the transition matrix conditional on the sequence not ending
        # This means that we need to renormalize the rows of self.edges
        exp_edges = torch.exp(self.edges)
        return exp_edges / exp_edges.sum(dim=1, keepdim=True)

    def get_stat_dist(self):
        p = self.get_trans_mat()

        # Get the eigenvectors and eigenvalues
        eig = torch.linalg.eig(p.T)

        # Index of eigenvector corresponding to eigenvalue 1
        eig_vec_idx = torch.where(torch.isclose(torch.real(eig[0]), torch.tensor(1.0)))[0][0]

        # Normalized eigenvector
        stat_dist = torch.real(eig[1][:, eig_vec_idx] / sum(eig[1][:, eig_vec_idx]))

        return stat_dist

    def predict_states(self, sequences: Union[torch.Tensor, list[torch.Tensor]]):
        if type(sequences) == torch.Tensor:
            sequences: list[torch.Tensor] = [sequences]

        viterbi_predictions: list[torch.Tensor] = []
        individual_state_predictions: list[torch.Tensor] = []
        for sequence in sequences:
            viterbi_predictions.append(self.viterbi(sequence.unsqueeze(0)))
            individual_state_predictions.append(self.predict_individual_states(sequence.unsqueeze(0)))

        return {"viterbi": viterbi_predictions, "individual": individual_state_predictions}

    def predict_individual_states(self, sequence: torch.Tensor):
        return super().predict(sequence)

    def viterbi(self, sequence: torch.Tensor):
        # Adapted from https://medium.com/@zhe.feng0018/coding-viterbi-algorithm-for-hmm-from-scratch-ca59c9203964
        # sequence dim is (1, seq_len, d+1)

        y = sequence.squeeze(0)  # (seq_len, d+1)
        log_a = torch.log(self.get_trans_mat())  # n_states x n_states, log transition matrix
        log_b = self._emission_matrix(sequence).squeeze(0).T  # n_states x seq_len, log emission matrix
        log_pi = self.starts  # n_states, log initial state distribution

        # Initialize
        x_seq = torch.zeros([self.n_states, 0])
        V = log_b[:, 0] + log_pi  # n_states, log probability of most likely path ending in state i at time 0

        # Forward pass to calculate the value function and fill in the DP table
        for i in range(1, y.shape[0]):
            _V = torch.tile(log_b[:, i], dims=[self.n_states, 1]).T + log_a.T + torch.tile(V, dims=[self.n_states, 1])
            x_ind = torch.argmax(_V, axis=1)
            x_seq = torch.hstack([x_seq, x_ind.unsqueeze(1)])
            V = _V[torch.arange(self.n_states), x_ind]

        # Backward pass to find the optimal path
        x_T = torch.argmax(V)
        x_seq_opt, i = torch.empty(x_seq.shape[1] + 1), x_seq.shape[1]
        prev_ind = x_T
        while i >= 0:
            x_seq_opt[i] = prev_ind
            i -= 1
            prev_ind = x_seq[int(prev_ind), i]

        return x_seq_opt
