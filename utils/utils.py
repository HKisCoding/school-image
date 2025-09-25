import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch


def EProjSimplex_new_matrix(v, k=1):
    row_means = torch.mean(v, dim=1)
    v0 = v - row_means.view(-1, 1) + k / v.size(1)
    vmin = torch.min(v0, dim=1).values

    negative_mask = vmin < 0
    positive_mask = ~negative_mask

    v1 = v0.clone()

    ft = torch.ones_like(vmin)
    lambda_m = torch.zeros_like(vmin)
    iteration = 100
    for _ in range(iteration):
        v1[negative_mask] = v0[negative_mask] - lambda_m[negative_mask].unsqueeze(1)
        posidx = v1 > 0
        npos = torch.sum(posidx, dim=1)
        g = -npos
        f = torch.sum(torch.where(posidx, v1, torch.zeros_like(v1)), dim=1) - k
        lambda_m = lambda_m - f / g
        ft += 1
        x = torch.max(v1, torch.zeros_like(v1))
        x[ft > iteration] = torch.max(
            v1[ft > iteration], torch.zeros_like(v1[ft > iteration])
        )

    x_list = torch.where(negative_mask.unsqueeze(1), x, v0)
    x_list = torch.where(positive_mask.unsqueeze(1), x_list, v1)

    return torch.clamp(x_list, min=0)


def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1)  # [B, N, f]
    A = -2 * torch.bmm(y, x)  # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True)  # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
    return A.squeeze()


def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor | None = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy() if Y is not None else None
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W
