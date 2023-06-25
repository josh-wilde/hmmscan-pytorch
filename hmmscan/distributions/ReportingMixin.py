import torch
from math import log


class ReportingMixin:
    def get_starts(self):
        raise NotImplementedError

    def get_comp_params(self, threshold=1e-5):
        raise NotImplementedError

    def get_stat_dist(self):
        raise NotImplementedError

    def get_trans_mat(self):
        raise NotImplementedError

    def get_log_prob(self, X):
        raise NotImplementedError

    def get_log_prob_array(self, sequences):
        raise NotImplementedError

    def get_bic(self, sequences):
        # Reference: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        # k = ((parameters for comp dist + mix)*n_mix) - 1
        k = (1 + 1) * self.n_mix_comps - 1
        p = self.n_states**2 + self.n_states * k - 1
        t = sum([s.shape[0] for s in sequences])

        log_prob = self.get_log_prob(sequences)

        return -2 * log_prob + p * log(t)

    def get_bic_array(self, sequences):
        # Reference: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        # k = ((parameters for comp dist + mix)*n_mix) - 1
        k = (1 + 1) * self.n_mix_comps - 1
        p = self.n_states**2 + self.n_states * k - 1
        t = torch.tensor([s.shape[0] for s in sequences])

        log_probs = self.get_log_prob_array(sequences)

        return -2 * log_probs + p * torch.log(t)

    def get_mmdl(self, sequences):
        # Reference: Bicego 2003, p. 1399
        trans_mat_params = self.n_states * (self.n_states - 1)
        init_dist_params = self.n_states - 1
        comp_params = 2 * self.n_mix_comps - 1
        t = sum([s.shape[0] for s in sequences])
        stat_dist_x_t = t * self.get_stat_dist()
        stat_dist_x_t_adj = stat_dist_x_t[stat_dist_x_t >= 1]

        log_prob = self.get_log_prob(sequences)

        state_penalty = (trans_mat_params + init_dist_params) * log(t)
        comp_penalty = comp_params * torch.nansum(torch.log(stat_dist_x_t_adj))

        return -2 * log_prob + state_penalty + comp_penalty

    def initialize_metadata(self):
        self.metadata = {
            "model_type": self.name,
            "n_states": self.n_states,
            "n_mix_comps": self.n_mix_comps,
            "n": int(self.n),
            "init_type": self.init_type,
            "init_file_name": self.init_file_name,
            "init_file_row": self.init_file_row,
            "random_state": self.random_state,
            "random_init_upper_bound": self.random_init_upper_bound,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
