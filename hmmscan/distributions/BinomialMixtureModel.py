from typing import Optional, Union
import time
import sys
import torch
import numpy as np

from pomegranate.distributions._distribution import Distribution
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.kmeans import KMeans
from pomegranate._utils import _cast_as_parameter, _cast_as_tensor, _check_parameter, _reshape_weights, eps

from .BinomialModel import BinomialModel
from .ReportingMixin import ReportingMixin


class BinomialMixtureModel(ReportingMixin, GeneralMixtureModel):
    def __init__(
        self,
        n: int,
        distributions,
        priors=None,
        init_type: str = "kmeans",
        random_state: int = 0,
        random_init_upper_bound: float = 0.01,
        inertia: float = 0.0,
        frozen: bool = False,
        check_data: bool = True,
        init_file_row: Optional[int] = None,
        init_file_name: Optional[str] = None,
        verbose: bool = True,
        max_iter: int = int(1e8),
        tol: float = 1e-9,
    ):
        # Initialize GMM
        GeneralMixtureModel.__init__(
            self,
            distributions=distributions,
            priors=priors,
            max_iter=max_iter,
            tol=tol,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
            random_state=random_state,
            verbose=verbose,
        )

        # Initalize metadata
        self.name = "BinomialMixtureModel"
        self.n = n
        self.n_states = 1
        self.n_mix_comps = len(self.distributions)
        self.init_type = init_type
        self.init_file_name = init_file_name
        self.init_file_row = init_file_row
        self.random_init_upper_bound = random_init_upper_bound
        self.initialize_metadata()

    def _initialize(self, X, sample_weight=None):
        """Initialize the probability distribution.

        This method is meant to only be called internally. It initializes the
        parameters of the distribution and stores its dimensionality. For more
        complex methods, this function will do more.


        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, self.d + 1)
            The data to use to initialize the model.

        sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
            A set of weights for the examples. This can be either of shape
            (-1, self.d) or a vector of shape (-1,). Default is ones.
        """
        # print("Starting BinMixModel initialization")
        if self.init_type == "random":
            self._random_initialize()
        else:
            self._kmeans_initialize(X, sample_weight)

        self._initialized = True
        self._reset_cache()
        Distribution._initialize(self, X.shape[1] - 1)
        # print("Finished BinMixModel initialization")
        # print(f"state_dict: {self.state_dict()}")

    def _random_initialize(self):
        self.priors = _cast_as_parameter(torch.ones(self.k, dtype=self.dtype, device=self.device) / self.k)
        # print(f"Initializing priors to {self.priors}")

        torch.manual_seed(self.random_state)
        rand_ps = (
            torch.rand(self.k, dtype=self.dtype, device=self.device) * self.random_init_upper_bound * 500 / self.n
        )

        for i in range(self.k):
            # print(f"Initializing distribution {i} with p={rand_ps[i]}")
            self.distributions[i] = BinomialModel(
                n=self.n,
                p=rand_ps[i],
                random_state=self.random_state,
                random_init_upper_bound=self.random_init_upper_bound,
                inertia=self.inertia,
                frozen=self.frozen,
                check_data=self.check_data,
                init_file_row=self.init_file_row,
                init_file_name=self.init_file_name,
            )

    def _kmeans_initialize(self, X, sample_weight=None):
        X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)

        if sample_weight is None:
            sample_weight = torch.ones(1, dtype=self.dtype, device=self.device).expand(X.shape[0], 1)
        else:
            sample_weight = _check_parameter(
                _cast_as_tensor(sample_weight), "sample_weight", min_value=0.0, check_parameter=self.check_data
            )

        model = KMeans(self.k, init="random", max_iter=3, random_state=self.random_state)

        if self.device != model.device:
            model.to(self.device)

        # print(f"Sample weight sum: {torch.sum(sample_weight)}")
        try:
            y_hat = model.fit_predict(X, sample_weight=sample_weight)
        except:
            print("Error in KMeans fit_predict. Reverting to random initialization.")
            self._random_initialize()
        else:
            self.priors = _cast_as_parameter(torch.empty(self.k, dtype=self.dtype, device=self.device))

            for i in range(self.k):
                idx = y_hat == i

                # print(f"Starting fit to inititalize distribution {i}")
                p = eps if torch.sum(idx) == 0 else None
                self.distributions[i] = BinomialModel(
                    n=self.n,
                    p=p,
                    random_state=self.random_state,
                    random_init_upper_bound=self.random_init_upper_bound,
                    inertia=self.inertia,
                    frozen=self.frozen,
                    check_data=self.check_data,
                    init_file_row=self.init_file_row,
                    init_file_name=self.init_file_name,
                )
                if not self.distributions[i]._initialized:
                    self.distributions[i].fit(X[idx], sample_weight=sample_weight[idx])
                    self.priors[i] = idx.type(torch.float32).mean()
                    if self.priors[i] == 1.0:
                        self.priors[i] = 1 - eps
                    if self.priors[i] == 0.0:
                        self.priors[i] = eps
                else:
                    self.priors[i] = eps

    def fit_loader(self, dataloader):
        train_data = torch.cat(dataloader.sequences, dim=0)

        # Fit model
        # self.fit(train_data)
        self._fit(train_data)

    def _fit(self, X, sample_weight=None, priors=None):
        """Fit the model to optionally weighted examples.

        This method implements the core of the learning process. For a
        mixture model, this involves performing EM until the distributions that
        are being fit converge according to the threshold set by `tol`, or
        until the maximum number of iterations has been hit.

        This method is largely a wrapper around the `summarize` and
        `from_summaries` methods. It's primary contribution is serving as a
        loop around these functions and to monitor convergence.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
            A set of examples to evaluate.

        sample_weight: list, tuple, numpy.ndarray, torch.Tensor optional
            A set of weights for the examples. This can be either of shape
            (-1, self.d) or a vector of shape (-1,). Default is ones.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            This can be used when only some labels are known by using a
            uniform distribution when the labels are not known. Note that
            this can be used to assign hard labels, but does not have the
            same semantics for soft labels, in that it only influences the
            initial estimate of an observation being generated by a component,
            not gives a target. Default is None.


        Returns
        -------
        self
        """

        # print("In fit for BinMixModel.")
        # print(f"X shape: {X.shape}")
        logp = None
        for i in range(self.max_iter):
            start_time = time.time()

            last_logp = logp

            logp = self.summarize(X, sample_weight=sample_weight, priors=priors)

            if i > 0:
                improvement = logp - last_logp
                duration = time.time() - start_time

                if self.verbose:
                    print("[{}] Improvement: {}, Time: {:4.4}s".format(i, improvement, duration))

                if improvement < self.tol:
                    break

                if np.isnan(improvement):
                    print(f"logp: {logp}")
                    print(f"last_logp: {last_logp}")
                    print(self.state_dict())
                    print(self.init_type)
                    sys.exit()

            self.from_summaries()

        self._reset_cache()
        return self

    def _emission_matrix(self, X, priors=None):
        """Return the emission/responsibility matrix.

        This method returns the log probability of each example under each
        distribution contained in the model with the log prior probability
        of each component added.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
            A set of examples to evaluate.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
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
        e: torch.Tensor, shape=(-1, self.k)
            A set of log probabilities for each example under each distribution.
        """

        X = _check_parameter(_cast_as_tensor(X), "X", ndim=2, shape=(-1, self.d + 1), check_parameter=self.check_data)

        priors = _check_parameter(
            _cast_as_tensor(priors),
            "priors",
            ndim=2,
            shape=(X.shape[0], self.k),
            min_value=0.0,
            max_value=1.0,
            value_sum=1.0,
            value_sum_dim=-1,
            check_parameter=self.check_data,
        )

        d = X.shape[0]
        e = torch.empty(d, self.k, device=self.device, dtype=self.dtype)
        for i, d in enumerate(self.distributions):
            e[:, i] = d.log_probability(X)

            if torch.isnan(e[:, i]).any():
                print(f"e: {e}")
                print(f"i: {i}")
                print(f"d: {d}")
                print(f"d(X): {d.log_probability(X)}")
                print(f"state_dict: {self.state_dict()}")
                sys.exit()

        if priors is not None:
            e += torch.log(priors)

        return e + self._log_priors

    def summarize(self, X, sample_weight=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

        This method calculates the sufficient statistics from optionally
        weighted data and adds them to the stored cache. The examples must be
        given in a 2D format. Sample weights can either be provided as one
        value per example or as a 2D matrix of weights for each feature in
        each example. Labels can be provided for examples but, if provided,
        must be incomplete such that semi-supervised learning can be performed.

        For a mixture model, this step is essentially performing the 'E' part
        of the EM algorithm on a batch of data, where examples are soft-assigned
        to distributions in the model and summaries are derived from that.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
            A set of examples to summarize.

        sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
            A set of weights for the examples. This can be either of shape
            (-1, self.d) or a vector of shape (-1,). Default is ones.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)
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
        logp: float
            The log probability of X given the model.
        """
        # print("Entering summarize in BinomialMixtureModel")

        X = _check_parameter(_cast_as_tensor(X), "X", ndim=2)
        if not self._initialized:
            # print("Initializing in BinomialMixtureModel.summarize")
            self._initialize(X, sample_weight=sample_weight)
            # print("Finished initializing in BinomialMixtureModel.summarize")

        sample_weight = _reshape_weights(
            X[:, 0:1], _cast_as_tensor(sample_weight, dtype=torch.float32), device=self.device
        )

        e = self._emission_matrix(X, priors=priors)
        logp = torch.logsumexp(e, dim=1, keepdims=True)
        y = torch.exp(e - logp)

        z = torch.clone(self._w_sum)

        for i, d in enumerate(self.distributions):
            d.summarize(X, y[:, i : i + 1] * sample_weight)

            if self.frozen == False:
                self._w_sum[i] = self._w_sum[i] + (y[:, i : i + 1] * sample_weight).mean(dim=-1).sum()

        return torch.sum(logp)

    def get_log_prob(self, sequences):
        """
        Input: list of tensors containing ae and lot size sequences
        Output: log probability of all of the sequences concatenated
        """
        if len(sequences) > 0:
            return torch.sum(self.log_probability(torch.cat(sequences, dim=0)))
        else:
            return torch.tensor(float("nan"))

    def get_log_prob_array(self, sequences):
        """
        Input: list of tensors containing ae and lot size sequences
        Output: log probability of each of the sequences
        """
        # BinomialDistribution class only accepts 2d numpy arrays for log_probability evaluation
        if len(sequences) > 0:
            return torch.tensor([torch.sum(self.log_probability(sequence)) for sequence in sequences])
        else:
            return torch.tensor(float("nan"))

    def get_comp_params(self, threshold=1e-7):
        ps = torch.tensor([d.p for d in self.distributions])
        return {"s0": torch.where(self.priors > threshold, ps, torch.nan)}, {
            "s0": torch.tensor([weight for weight in self.priors])
        }

    def get_trans_mat(self):
        # HMM transition matrix is in the form of a 2D numpy array
        return torch.tensor([[1.0]])

    def get_stat_dist(self):
        return torch.tensor([1.0])

    def get_starts(self):
        return torch.tensor([0.0])

    def get_model_samples(self, n=1, length=1, lot_size_path=None, path=False, random_state=0):
        samples: list[torch.Tensor] = []
        for i in range(n):
            torch.manual_seed(random_state + i * 10000)
            samples.append(self.sample(n=length, lot_size_path=lot_size_path).squeeze(1))
        if path:
            paths = [torch.tensor([0] * length) for _ in range(n)]
            return samples, paths
        else:
            return samples

    def sample(self, n, lot_size_path=None):
        """Sample from the probability distribution.

        This method will return `n` samples generated from the underlying
        probability distribution. For a mixture model, this involves first
        sampling the component using the prior probabilities, and then sampling
        from the chosen distribution.


        Parameters
        ----------
        n: int
            The number of samples to generate.


        Returns
        -------
        X: torch.tensor, shape=(n, self.d)
            Randomly generated samples.
        """

        X = []
        for distribution in self.distributions:
            X_ = distribution.sample(n, lot_size_path=lot_size_path)
            X.append(X_)

        X = torch.stack(X)
        idxs = torch.multinomial(self.priors, num_samples=n, replacement=True)
        return X[idxs, torch.arange(n)]

    @staticmethod
    def predict_states(sequences: Union[torch.Tensor, list[torch.Tensor]]):
        if type(sequences) == torch.Tensor:
            sequences: list[torch.Tensor] = [sequences]

        predictions: list[torch.Tensor] = []
        for sequence in sequences:
            predictions.append(torch.tensor([0] * sequence.shape[0]))

        return {"viterbi": predictions, "individual": predictions}
