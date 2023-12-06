# -*- coding:utf-8 -*-
# @Time:    2023/6/2 17:16
# @Author:  YYZG
#
import os.path

import numpy as np
import torch

LOG_9999_N = np.log(9999)
LOG_9999_T = torch.log(torch.tensor(9999))


def row_sim(adj: np.ndarray, method) -> np.ndarray:
    params = []
    if method in [sim_gaussian, sim_gaussian1]:
        params.append(adj.shape[0] / (adj ** 2).sum())
    res = np.zeros((adj.shape[0],) * 2)
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            res[i][j] = method(adj, i, j, params)
            res[j][i] = res[i][j]
    return res


def row_sim_tensor(adj: torch.Tensor, method) -> torch.Tensor:
    params = []
    if method in [sim_gaussian_tensor, sim_gaussian1_tensor]:
        params.append(adj.shape[0] / (adj ** 2).sum())
    res = torch.zeros((adj.shape[0],) * 2, device=adj.device)
    for i in range(adj.shape[0]):
        for j in range(i, adj.shape[0]):
            res[i][j] = method(adj, i, j, params)
            res[j][i] = res[i][j]
    return res


def sim_gaussian(adj: np.ndarray, i: int, j: int, params: list) -> float:
    """
    Gaussian interaction profile kernel similarity
    """
    return np.exp(-params[0] * np.sum((adj[i] - adj[j]) ** 2))


def sim_gaussian_tensor(adj: torch.Tensor, i: int, j: int, params: list) -> torch.Tensor:
    return torch.exp(-params[0] * torch.sum(torch.pow(adj[i] - adj[j], 2)))


def sim_gaussian1(adj: np.ndarray, i: int, j: int, params: list) -> float:
    """
    with result 1/(1+exp(-15*res+np.log(9999)))
    """
    return 1 / (1 + np.exp(-15 * sim_gaussian(adj, i, j, params) + LOG_9999_N))


def sim_gaussian1_tensor(adj: torch.Tensor, i: int, j: int, params: list) -> torch.Tensor:
    return 1 / (1 + torch.exp(-15 * sim_gaussian_tensor(adj, i, j, params) + LOG_9999_T))


def sim_cosine(adj: np.ndarray, i: int, j: int, params: list) -> float:
    """
    Cosine similarity
    """
    tmp = np.linalg.norm(adj[i]) * np.linalg.norm(adj[j])
    return np.dot(adj[i], adj[j]) / tmp if tmp else 0


def sim_cosine_tensor(adj: torch.Tensor, i: int, j: int, params: list) -> torch.Tensor:
    tmp = torch.linalg.norm(adj[i]) * torch.linalg.norm(adj[j])
    return torch.dot(adj[i], adj[j]) / tmp if tmp else 0


def sim_jaccard(adj: np.ndarray, i: int, j: int, params: list) -> float:
    """
    Jaccard similarity
    """
    inter = np.intersect1d(np.where(adj[i] != 0), np.where(adj[j] != 0)).shape[0]
    union = np.union1d(np.where(adj[i] != 0), np.where(adj[j] != 0)).shape[0]
    return inter / union


def sim_jaccard_tensor(adj: torch.Tensor, i: int, j: int, params: list) -> torch.Tensor:
    return torch.as_tensor(sim_jaccard(adj.cpu().numpy(), i, j, params))


def sim_microbe2microbe_fs(adj: np.ndarray, i: int, j: int, params: list) -> float:
    """
    Kamneva

    """
    pass


def sim_hamming(adj: np.ndarray, i: int, j: int, params: list) -> float:
    return 1 - np.count_nonzero(adj[i] != adj[j]) / adj.shape[1]


def sim_hamming_tensor(adj: torch.Tensor, i: int, j: int, params: list) -> torch.Tensor:
    return 1 - torch.count_nonzero(adj[i] != adj[j]) / adj.shape[1]


def RWR_N(SM: np.ndarray) -> np.ndarray:
    alpha = 0.1
    E = np.identity(SM.shape[0])
    M = SM / SM.sum(1, keepdims=True)
    res = np.zeros_like(SM)
    for i in range(SM.shape[0]):
        e_i = E[i]
        p_i = e_i
        for j in range(10):
            p_i = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        res[i] = p_i
    return res


def get_drug_sims_tensor(sims_name: list) -> torch.Tensor:
    pass


