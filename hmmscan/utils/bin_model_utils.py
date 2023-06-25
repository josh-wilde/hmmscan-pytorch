import torch


def get_stat_from_trans(p):
    # Get the eigenvectors and eigenvalues
    eig = torch.linalg.eig(p.T)

    # Index of eigenvector corresponding to eigenvalue 1
    eig_vec_idx = torch.where(torch.isclose(torch.real(eig[0]), torch.tensor(1.0)))[0][0]

    # Normalized eigenvector
    stat_dist = torch.real(eig[1][:, eig_vec_idx] / sum(eig[1][:, eig_vec_idx]))

    return stat_dist
