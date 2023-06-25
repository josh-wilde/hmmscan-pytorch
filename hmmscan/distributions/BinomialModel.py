from typing import Optional, Union
import torch

from .BinomialDistribution import BinomialDistribution
from .ReportingMixin import ReportingMixin


class BinomialModel(ReportingMixin, BinomialDistribution):
    def __init__(
        self,
        n: int,
        p: Optional[float] = None,
        init_type: str = "",
        random_state: int = 0,
        random_init_upper_bound: float = 0.01,
        inertia: float = 0.0,
        frozen: bool = False,
        check_data: bool = True,
        init_file_row: Optional[int] = None,
        init_file_name: Optional[str] = None,
    ):
        # Initialize distribution
        BinomialDistribution.__init__(
            self,
            n=n,
            p=p,
            random_state=random_state,
            random_init_upper_bound=random_init_upper_bound,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
        )

        # Initalize extra metadata
        self.name = "BinomialModel"
        self.n_states = 1
        self.n_mix_comps = 1
        self.init_type = init_type
        self.init_file_name = init_file_name
        self.init_file_row = init_file_row
        self.max_iter = None
        self.tol = None
        self.initialize_metadata()

    def fit_loader(self, dataloader):
        train_data = torch.cat(dataloader.sequences, dim=0)

        # Fit model
        super().fit(train_data)

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
        return {"s0": torch.tensor([self.p])}, {"s0": torch.tensor([1.0])}

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
            samples.append(
                self.sample(
                    n=length, lot_size_path=lot_size_path, sample_random_state=random_state + i * 10000
                ).squeeze(1)
            )
        if path:
            paths = [torch.tensor([0] * length) for _ in range(n)]
            return samples, paths
        else:
            return samples

    @staticmethod
    def predict_states(sequences: Union[torch.Tensor, list[torch.Tensor]]):
        if type(sequences) == torch.Tensor:
            sequences: list[torch.Tensor] = [sequences]

        predictions: list[torch.Tensor] = []
        for sequence in sequences:
            predictions.append(torch.tensor([0] * sequence.shape[0]))

        return {"viterbi": predictions, "individual": predictions}
