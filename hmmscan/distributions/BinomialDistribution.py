from typing import Optional
import torch
import sys

from pomegranate.distributions._distribution import Distribution
from pomegranate._utils import (
    _cast_as_parameter,
    _cast_as_tensor,
    _update_parameter,
    _check_parameter,
    _reshape_weights,
    eps,
)


class BinomialDistribution(Distribution):
    def __init__(
        self,
        n: int,
        p: Optional[float] = None,
        random_state: int = 0,
        random_init_upper_bound: float = 0.01,
        inertia: float = 0.0,
        frozen: bool = False,
        check_data: bool = True,
    ):
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = "BinomialDistribution"
        self.random_state = random_state
        self.random_init_upper_bound = random_init_upper_bound

        n = torch.tensor([n])
        if p is not None:
            if p < eps:
                p = eps
            elif p > 1 - eps:
                p = 1 - eps
            p = torch.tensor([p])

        self.p = _check_parameter(
            _cast_as_parameter(p),
            "p",
            min_value=eps,
            max_value=1 - eps,
            ndim=1,
        )
        self.n = _check_parameter(_cast_as_parameter(n), "n", min_value=1, ndim=1)

        self._initialized = self.p is not None
        self.d = 1 if self._initialized else None
        self._reset_cache()

    def _initialize(self, d):
        torch.manual_seed(self.random_state)

        self.p = _cast_as_parameter(
            torch.rand(d, dtype=self.dtype, device=self.device) * self.random_init_upper_bound * 500 / self.n
        )

        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        if self._initialized is False:
            return

        self.register_buffer("_kw_sum", torch.zeros(self.d, device=self.device))
        self.register_buffer("_nw_sum", torch.zeros(self.d, device=self.device))

    def sample(self, n, lot_size_path=None, sample_random_state: Optional[int] = None):
        if sample_random_state is not None:
            torch.manual_seed(sample_random_state)
        if lot_size_path is None:
            return torch.distributions.Binomial(total_count=self.n, probs=self.p).sample(
                [n]
            )  # returns tensor of size (n, 1)
        else:
            return torch.cat(
                [torch.distributions.Binomial(total_count=n, probs=self.p).sample([1]) for n in lot_size_path.numpy()]
            )

    def log_probability(self, X):
        X = _check_parameter(
            _cast_as_tensor(X, dtype=self.p.dtype),
            "X",
            min_value=0,
            ndim=2,
            shape=(-1, self.d + 1),
            check_parameter=self.check_data,
        )
        ks = X[:, 0]
        ns = X[:, 1]

        lprob = (
            torch.lgamma(ns + 1)
            - torch.lgamma(ks + 1)
            - torch.lgamma(ns - ks + 1)
            + ks * torch.log(self.p)
            + (ns - ks) * torch.log(1 - self.p)
        )

        if torch.isnan(lprob).any():
            print("NAN in BinomialDistribution.log_probability")
            print(f"Initialized: {self._initialized}")
            print(f"ks: {ks}")
            print(f"ns: {ns}")
            print(f"self.p: {self.p}")

        return lprob

    def summarize(self, X, sample_weight=None):
        if self.frozen is True:
            return

        # Initialize
        if not self._initialized:
            self._initialize(len(X[0]) - 1)

        # Format inputs
        X = _cast_as_tensor(X)
        _check_parameter(
            X,
            "X",
            min_value=0,
            ndim=2,
            shape=(-1, self.d + 1),
            check_parameter=self.check_data,
        )
        # print(f"Shape of X input in BinomialDistribution input: {X[:, 0:1].shape}")
        # print(f"Shape of sample weight input in BinomialDistribution input: {_cast_as_tensor(sample_weight).shape}")
        sample_weight = _reshape_weights(X[:, 0:1], _cast_as_tensor(sample_weight), device=self.device)

        # Update summary statistics
        # print("In BinomialDistribution.summarize")
        # if torch.isnan(sample_weight).any():
        #    print("nan in sample_weight")
        #    print(f"sample_weight: {sample_weight}")
        #    sys.exit()
        # if torch.isnan(X[:, 0:1]).any():
        #    print("nan in X[:, 0:1]")
        #    print(f"X[:, 0:1]: {X[:, 0:1]}")
        #    sys.exit()
        # if torch.isnan(X[:, 1:2]).any():
        #    print("nan in X[:, 1:2]")
        #    print(f"X[:, 1:2]: {X[:, 1:2]}")
        #    sys.exit()
        self._kw_sum += torch.sum(X[:, 0:1] * sample_weight, dim=0)
        self._nw_sum += max(eps, torch.sum(X[:, 1:2] * sample_weight, dim=0))
        # print(f"self._kw_sum: {self._kw_sum}")
        # print(f"self._nw_sum: {self._nw_sum}")

    def from_summaries(self):
        if self.frozen is True:
            return

        # Update parameters
        p = self._kw_sum / self._nw_sum
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps

        # print("In BinomialDistribution.from_summaries")
        # print(f"self.p: {self.p}")
        # print(f"p: {p}")
        # print(f"self._kw_sum: {self._kw_sum}")
        # print(f"self._nw_sum: {self._nw_sum}")

        _update_parameter(self.p, p, self.inertia)
        self._reset_cache()
