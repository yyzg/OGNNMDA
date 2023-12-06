# -*- coding:utf-8 -*-
# @Time:    2023/6/2 17:20
# @Author:  YYZG
#
import gc
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

import sim_calc
from util import get_logger, adj_to_edge_index, func

logger = get_logger('dataset', './log/dataset.log')
re_index = np.random.shuffle

def concat(res, m0, m1, m2):
    res[:m0.shape[0], :m0.shape[0]] = m1
    res[m0.shape[0]:, m0.shape[0]:] = m2
    res[:m0.shape[0], m0.shape[0]:] = m0
    res[m0.shape[0]:, :m0.shape[0]] = m0.T


get_index = np.where


def get_heterogeneous_network_tensor(adj_matrix: torch.Tensor, sim_matrix1: torch.Tensor,
                                     sim_matrix2: torch.Tensor, res: Optional[torch.Tensor]) -> torch.Tensor:
    assert adj_matrix.shape[0] == sim_matrix1.shape[0]
    assert adj_matrix.shape[1] == sim_matrix2.shape[0]
    device = adj_matrix.device
    if res is None:
        res = torch.zeros((adj_matrix.shape[0] + adj_matrix.shape[1],) * 2, device=device)
    concat(res, adj_matrix, sim_matrix1, sim_matrix2)
    return res


def get_heterogeneous_network(adj_matrix: np.ndarray, sim_matrix1: np.ndarray,
                              sim_matrix2: np.ndarray) -> np.ndarray:
    assert adj_matrix.shape[0] == sim_matrix1.shape[0]
    assert adj_matrix.shape[1] == sim_matrix2.shape[0]
    res = np.zeros((adj_matrix.shape[0] + adj_matrix.shape[1],) * 2)
    concat(res, adj_matrix, sim_matrix1, sim_matrix2)
    return res


def transform(label,pred):
    indices1 = np.where(label == 1)[0]
    indices2 = np.where(label == 0)[0]
    np.random.shuffle(indices2)
    indices = np.hstack((indices1, indices2[:indices1.shape[0]]))
    label = label[indices]
    pred = pred[indices]
    return label, pred


def get_drug_microbe_adj(data_dir: Path, dataset_name: str, logger: Logger = logger) -> np.ndarray:
    path = data_dir / dataset_name / 'drug_microbe_matrix.txt'
    logger.debug(f'loading: {path}')
    return np.loadtxt(path, dtype='float32')


def get_drug_microbe_adj_tensor(data_dir: Path, dataset_name: str, device: torch.device,
                                logger: Logger = logger) -> torch.Tensor:
    adj = get_drug_microbe_adj(data_dir, dataset_name, logger)
    return torch.as_tensor(adj, device=device)

def get_drug_microbe_heterogeneous_network_tensor(data_dir: Path, dataset_name: str, adj: torch.Tensor,
                                                  device: torch.device, logger: Logger = logger) -> torch.Tensor:
    drug_structure_sim_file = data_dir / dataset_name / 'drug_structure_sim.txt'
    logger.debug(f'loading: {drug_structure_sim_file}')
    sim_drug_structure = np.loadtxt(drug_structure_sim_file)
    microbe_function_sim_file = data_dir / dataset_name / 'microbe_function_sim.txt'
    logger.debug(f'loading {microbe_function_sim_file}')
    sim_microbe_function = np.loadtxt(microbe_function_sim_file)

    adj_drug_microbe = adj.cpu().numpy()
    logger.info('Calculate Gaussian interaction profile kernel similarity for drugs ...')
    sim_drug_gaussian = sim_calc.row_sim(adj_drug_microbe, sim_calc.sim_gaussian1)
    logger.info('Calculate Gaussian interaction profile kernel similarity for microbes ...')
    sim_microbe_gaussian = sim_calc.row_sim(adj_drug_microbe.T, sim_calc.sim_gaussian)

    sim_drug_structure[sim_drug_structure < 1e-4] = 0
    sim_microbe_function[sim_microbe_function < 1e-4] = 0
    logger.info('Calculate the final similarity for drugs ')
    sim_drug = np.where(sim_drug_structure != 0, (sim_drug_gaussian + sim_drug_structure) / 2, sim_drug_gaussian)
    logger.info('Calculate the final similarity for microbes ')
    sim_microbe = np.where(sim_microbe_function != 0, (sim_microbe_gaussian + sim_microbe_function) / 2,
                           sim_microbe_gaussian)
    logger.info('Get the heterogeneous network ')
    res = get_heterogeneous_network(adj_drug_microbe, sim_drug, sim_microbe)
    res = torch.as_tensor(res, device=device)
    del sim_drug_structure
    del sim_drug_gaussian
    del sim_microbe_function
    del sim_microbe_gaussian
    gc.collect()
    return res

def get_initial_embedding_feature_tensor(adj: torch.Tensor) -> torch.Tensor:
    h0 = torch.zeros((adj.shape[0] + adj.shape[1],) * 2, device=adj.device)
    h0[:adj.shape[0], -adj.shape[1]:] = adj
    h0[-adj.shape[1]:, :adj.shape[0]] = adj.T
    return h0


def get_drug_microbe_sims(data_dir, dataset_name, adj, device):
    drug_sims = torch.zeros(3, adj.shape[0], adj.shape[0], device=device)
    print('calc drug gip similarity ...')
    # drug_sims[0] = row_sim_tensor(adj,sim_gaussian1_tensor)
    drug_sims[0] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'drug_gaussian_sim.txt')).to(
        device)
    print('calc drug hamming similarity ...')
    # drug_sims[1] = row_sim_tensor(adj,sim_hamming_tensor)
    drug_sims[1] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'drug_hamming_sim.txt')).to(
        device)
    drug_sims[2] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'drug_structure_sim.txt')).to(
        device)
    microbe_sims = torch.zeros(3, adj.shape[1], adj.shape[1], device=device)
    print('calc microbe gip similarity ...')
    # microbe_sims[0] = row_sim_tensor(adj.T,sim_gaussian_tensor)
    microbe_sims[0] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'microbe_gaussian_sim.txt'))
    print('calc microbe hamming similarity ...')
    # microbe_sims[1] = row_sim_tensor(adj.T,sim_hamming_tensor)
    microbe_sims[1] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'microbe_hamming_sim.txt'))
    microbe_sims[2] = torch.from_numpy(np.loadtxt(Path(data_dir) / dataset_name / 'microbe_function_sim.txt'))
    print('similarities calculation finished.')
    return drug_sims, microbe_sims


def get_PyG_Data(data_dir: Path, dataset_name: str, adj: torch.Tensor,
                 device: torch.device = torch.device('cpu')) -> Data:
    # adj = get_drug_microbe_adj_tensor(dataset_name, data_dir, device)
    h_net = get_drug_microbe_heterogeneous_network_tensor(data_dir, dataset_name, adj, device)
    H0 = get_initial_embedding_feature_tensor(adj=adj)
    edge_index, edge_weights = adj_to_edge_index(h_net.cpu().numpy(), device)
    data = Data(
        x=H0, edge_index=edge_index, edge_attr=edge_weights,
        h_net=h_net, num_drugs=adj.shape[0], num_microbes=adj.shape[1]
    )
    return data


def get_PyG_Data_(data_dir: Path, dataset_name: str, device: torch.device = torch.device('cpu')) -> Data:
    return get_PyG_Data(data_dir, dataset_name, get_drug_microbe_adj_tensor(data_dir, dataset_name, device), device)


def k_fold_split(k: int, adj: torch.Tensor, *, eq: bool = False):
    # (train_edges_index,test_edges_index)
    logger.info('get all edges ...')
    known_edges_index = torch.vstack(torch.where(adj == 1)).T
    logger.info(f'all known edges num: {known_edges_index.shape[0]}')
    unknown_edges_index = torch.vstack(torch.where(adj == 0)).T
    logger.info(f'all unknown edges num: {unknown_edges_index.shape[0]}')
    _k1 = known_edges_index.shape[0] % k
    _size1 = known_edges_index.shape[0] // k
    _k2 = unknown_edges_index.shape[0] % k
    _size2 = unknown_edges_index.shape[0] // k
    if eq:
        _size2 = _size1
    logger.info('shuffle the edges ...')
    shuffled_known = known_edges_index[torch.randperm(known_edges_index.size(0))]
    shuffled_unknown = unknown_edges_index[torch.randperm(unknown_edges_index.size(0))]
    for i in range(k):
        test_index = torch.zeros_like(adj)
        if i == k - 1:
            train_index = shuffled_known[:(k - 1) * _size1]
            test_known_edges_index = shuffled_known[(k - 1) * _size1:]
            if eq:
                test_unknown_edges_idnex = shuffled_unknown[(k - 1) * _size2:shuffled_known.shape[0]]
            else:
                test_unknown_edges_idnex = shuffled_unknown[(k - 1) * _size2:]
            test_index[tuple(torch.vstack((
                test_known_edges_index,
                test_unknown_edges_idnex
            )).T)] = 1
        else:
            train_index = torch.vstack((
                shuffled_known[:i * _size1],
                shuffled_known[(i + 1) * _size1:]
            ))
            test_known_edges_index = shuffled_known[i * _size1:(i + 1) * _size1]
            test_unknown_edges_idnex = shuffled_unknown[i * _size2:(i + 1) * _size2]
            test_index[tuple(torch.vstack((
                test_known_edges_index,
                test_unknown_edges_idnex
            )).T)] = 1
        train_adj = torch.zeros_like(adj)
        train_adj[tuple(train_index.T)] = 1
        test_index = torch.where(test_index.reshape(-1) == 1)[0]
        logger.info(f'split {i + 1}: {train_adj.sum()} known edges for training, '
                    f'{test_known_edges_index.shape[0]} known edges and {test_unknown_edges_idnex.shape[0]} '
                    f'unknown edges for testing')
        yield (train_adj, test_index)
    pass

