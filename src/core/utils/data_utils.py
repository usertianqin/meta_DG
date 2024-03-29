import torch
import numpy as np
import os
import sys
import scipy.io
import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import csv
import json
from torch_geometric.datasets import WebKB
from torch_geometric.utils import degree, to_dense_adj
from sklearn.preprocessing import label_binarize

from .generic_utils import *


def reset_seed(config):
    seed = config.get('split_seed', 42)
    if seed < 0:
        seed = np.random.randint(1, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)



def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_dense_feature(mx):
    '''Row-normalize dense features'''
    rowsum = np.sum(mx, axis=1)
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_fb100(dataset):
    assert dataset in {'amherst', 'cornell', 'jh', 'penn', 'reed','caltech','brown','duke','carnegie','prince','brand','carne','yale'}
    name_map = {
        'amherst': 'Amherst41',
        'cornell': 'Cornell5',
        'jh': 'Johns Hopkins55',
        'penn': 'Penn94',
        'reed': 'Reed98',
        'caltech': 'Caltech36',
        'brown': 'Brown11',
        'duke': 'Duke14',
        'carnegie': 'Carnegie49',
        'prince': 'Princeton12',
        'brand': 'Brandeis99', 
        'yale': 'Yale4'
        
    }
    DATAPATH = '/data/fb'
    root_dir = os.path.join(DATAPATH, dataset)
    adj = pkl.load(open(os.path.join(root_dir, 'adj.pkl'), 'rb'))
    features = pkl.load(open(os.path.join(root_dir, 'features.pkl'), 'rb'))
    label = pkl.load(open(os.path.join(root_dir, 'labels.pkl'), 'rb'))
    
    idx_train = pkl.load(open(os.path.join(root_dir, 'idx_train.pkl'), 'rb'))
    idx_val = pkl.load(open(os.path.join(root_dir, 'idx_val.pkl'), 'rb'))
    idx_test = pkl.load(open(os.path.join(root_dir, 'idx_test.pkl'), 'rb'))

    return adj, label, features, idx_train, idx_val, idx_test

def load_twitch(dataset):
    assert dataset in {'DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'}
    DATAPATH ='/data/twitch'
    root_dir = os.path.join(DATAPATH, dataset)
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{root_dir}/musae_{dataset}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))
        node_ids = np.array(node_ids, dtype=int)
    with open(f"{root_dir}/musae_{dataset}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
            
    with open(f"{root_dir}/musae_{dataset}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    
    new_label = label[reorder_node_ids]
    label = new_label
    y=label.shape[0]
    idx_train = torch.LongTensor(range(y))
    idx_val = torch.LongTensor(range(y))
    idx_test = torch.LongTensor(range(y))

    
    return A, label, features, idx_train, idx_val, idx_test

def load_webkb(dataset):    
    
    DATAPATH ='/data/webKB'
    dataset = WebKB(root=DATAPATH, name=dataset)
    graph = dataset[0]
    
    A  = to_dense_adj(graph.edge_index)[0].numpy()
    edge_index = scipy.sparse.csr_matrix(A)
    label = graph.y.numpy()
    
    node_feat = graph.x
    
    
    idx_train = torch.LongTensor(range(len(label)))
    idx_val = torch.LongTensor(range(len(label)))
    idx_test = torch.LongTensor(range(len(label)))
    
    return edge_index, label, node_feat, idx_train, idx_val, idx_test

def load_data(dataset_str, prob_del_edge=None, prob_add_edge=None, seed=1234, sparse_init_adj=False):

    if dataset_str.startswith('fb'):
        sub_name = dataset_str.strip().split('_')
        assert len(sub_name) == 2
        sub_name = sub_name[1]
        adj, labels, features, idx_train, idx_val, idx_test = load_fb100(
            sub_name)
        features = normalize_dense_feature(features)
        features = torch.Tensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
    elif dataset_str.startswith('tw'):
        sub_name = dataset_str.strip().split('_')
        assert len(sub_name) == 2
        sub_name = sub_name[1]
        adj, labels, features, idx_train, idx_val, idx_test = load_twitch(sub_name)
        features = normalize_dense_feature(features)
        features = torch.Tensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
    elif dataset_str.startswith('wb'):
        sub_name = dataset_str.strip().split('_')
        assert len(sub_name) == 2
        sub_name = sub_name[1]
        adj, labels, features, idx_train, idx_val, idx_test = load_webkb(sub_name)
 
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

  

    else:
        assert dataset_str in {'cora', 'citeseer', 'pubmed'}
        data_dir = os.path.join('/data', dataset_str)
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(os.path.join(
            data_dir, 'ind.{}.test.index'.format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        raw_features = sp.vstack((allx, tx)).tolil()
        raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]
        features = normalize_features(raw_features)
        raw_features = torch.Tensor(raw_features.todense())
        features = torch.Tensor(features.todense())

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
      
        labels = torch.LongTensor(np.argmax(labels, axis=1))

        idx_train = torch.LongTensor(range(len(y)))
        idx_val = torch.LongTensor(range(len(y), len(y) + 500))
        idx_test = torch.LongTensor(test_idx_range.tolist())
       

    if prob_del_edge is not None:
        adj = graph_delete_connections(
            prob_del_edge, seed, adj.toarray(), enforce_connected=False)
        adj = adj + np.eye(adj.shape[0])
        adj_norm = normalize_adj(torch.Tensor(adj))
        adj_norm = sp.csr_matrix(adj_norm)

    elif prob_add_edge is not None:
        adj = graph_add_connections(
            prob_add_edge, seed, adj.toarray(), enforce_connected=False)
        adj = adj + np.eye(adj.shape[0])
        adj_norm = normalize_adj(torch.Tensor(adj))
        adj_norm = sp.csr_matrix(adj_norm)

    else:
        adj = adj + sp.eye(adj.shape[0])
        adj_norm = normalize_sparse_adj(adj)

    if sparse_init_adj:
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    else:
        adj_norm = torch.Tensor(adj_norm.todense())

    return adj_norm, features, labels, idx_train, idx_val, idx_test



def graph_delete_connections(prob_del, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    del_adj = np.array(adj, dtype=np.float32)
    pre_num_edges = np.sum(del_adj)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del],
                      size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)

    cur_num_edges = np.sum(del_adj)
    print('[ Deleted {}% edges ]'.format(
        100 * (pre_num_edges - cur_num_edges) / pre_num_edges))
    return del_adj


def graph_add_connections(prob_add, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    add_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[1. - prob_add, prob_add],
                      size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    add_adj += smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(add_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                add_adj[k, other_node] = 1
                add_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)
    add_adj = (add_adj > 0).astype(float)
    return add_adj


def prepare_datasets(dataset_name, config):
    
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_name,
                                                                    prob_del_edge=config.get(
                                                                        'prob_del_edge', None), prob_add_edge=config.get('prob_add_edge', None), seed=config.get('data_seed',
                                                                                                                                                                 config['seed']),
                                                                    sparse_init_adj=config.get(
                                                                        'sparse_init_adj', False))


    device = config['device']

    data = {'adj': adj.to(device) if device else adj,
            'features': features.to(device) if device else features,
            'labels': labels.to(device) if device else labels,
            'idx_train': idx_train.to(device) if device else idx_train,
            'idx_val': idx_val.to(device) if device else idx_val,
            'idx_test': idx_test.to(device) if device else idx_test}


    return data
