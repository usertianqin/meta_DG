import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random

from .model import Model
from .models.pivot import compute_pivot_adj, sample_pivots
from .utils import AverageMeter
from .utils import constants as Constants
from .utils.data_utils import prepare_datasets
from ..model2.func import get_models, ParamGroupsCollector, ShrinkRatio, evaluate_acc
from ..model2.distr import edic


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network.
    """

    def __init__(self, config, ag, dataset_name, log_f, save_dir):
        # Evaluation Metrics:
        self._train_loss = AverageMeter() 
        self._dev_loss = AverageMeter()
        self._train_metrics = {
            'nloss': AverageMeter(),
            'acc': AverageMeter()
        }
        self._dev_metrics = {
            'nloss': AverageMeter(),
            'acc': AverageMeter()
        }

        self.log_f = log_f
        self.config = config
        self.ag = ag
        self.dirname = os.path.join(save_dir, dataset_name)
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        if not config['no_cuda'] and torch.cuda.is_available():
            self.device = torch.device(
                'cuda' if config['cuda_id'] < 0 else 'cuda:%d' %
                config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        datasets = prepare_datasets(dataset_name, config) 
        self.dataset = dataset_name
        config['num_feat'] = datasets['features'].shape[-1]
        config['num_class'] = datasets['labels'].max().item() + 1
        

        seed = config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        # Initialize the model
        self.model = Model(config)
        self.model.network = self.model.network.to(self.device)
        
        dim_x = config['hidden_size']
        dim_y = config['num_class']
        if dim_y == 2:
            dim_y = 1
        
        print("data, node_num, edge_num, dim_x, dim_y", dataset_name, datasets['features'].shape[0], datasets['adj'].shape[1], config['num_feat'], config['num_class'])
        
     
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.run_epoch = self._scalable_run_whole_epoch
        self.train_loader = datasets
        self.dev_loader = datasets
        self.test_loader = datasets
        self.config = self.model.config
        self.is_test = False
        self.has_save = False
        self.model.network.train() 
      

    def reset_parameters(self):
        self.model.network.encoder.reset_parameters()
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        for k in self._train_metrics:
            self._train_metrics[k] = AverageMeter()
        for k in self._dev_metrics:
            self._dev_metrics[k] = AverageMeter()

   

    def _scalable_run_whole_epoch(self,
                                  data_loader,
                                  gl_handler,
                                  opt_gl_handler,
                                  training=True,
                                  train_gl=True):
        '''first GNN, then GL'''
       
        
        mode = "train" 
        
        init_adj, features, labels = data_loader['adj'], data_loader[
            'features'], data_loader['labels']
        
        node_num = features.shape[0]
        support_num = int(node_num * self.ag.support_split)
        
        perm = np.random.permutation(node_num)
        
        support_idx = perm[:support_num]
        query_idx = perm[support_num:]
        query_lable = labels[query_idx]
        support_labels = labels[support_idx]
        
        
        
        
        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']   
        else:
            idx = data_loader['idx_test']
      
        

        network = self.model.network
        features = F.dropout(features,
                             network.config.get('feat_adj_dropout', 0),
                             training=network.training)
        init_node_vec = features
        
      
        node_vec = network.encoder.graph_encoders[0](init_node_vec,
                                                     init_adj,
                                                     pivot_mp=False,
                                                     batch_norm=False)

        init_pivot_vec, sampled_node_idx = sample_pivots(
            node_vec, int(0.5 * node_vec.shape[0]))
     
        # Compute n x p node-pivot relationship matrix 
        cur_node_pivot_adj = gl_handler.learn_graph(
            node_vec, pivot_features=init_pivot_vec)
        
        
        # Compute p x p pivot graph 1000*1000
        cur_pivot_adj = compute_pivot_adj(cur_node_pivot_adj)
        
        if self.config.get('max_iter', 10) > 0:
            cur_node_pivot_adj = F.dropout(cur_node_pivot_adj,
                                           network.config.get(
                                               'feat_adj_dropout', 0),
                                           training=network.training)

        cur_pivot_adj = F.dropout(cur_pivot_adj,
                                  network.config.get('feat_adj_dropout', 0),
                                  training=network.training)
        # Update node embeddings via node-pivot-node message passing
        init_agg_vec = network.encoder.graph_encoders[0](init_node_vec,
                                                         init_adj,
                                                         pivot_mp=False,
                                                         batch_norm=False)
        node_vec = (1 - gl_handler.graph_skip_conn) * network.encoder.graph_encoders[0](init_node_vec,
                                                                                        cur_node_pivot_adj,
                                                                                        pivot_mp=True,
                                                                                        batch_norm=False) + \
            gl_handler.graph_skip_conn * init_agg_vec
       
        if network.encoder.graph_encoders[0].bn is not None:
            node_vec = network.encoder.graph_encoders[0].compute_bn(node_vec)

        node_vec = torch.relu(node_vec)
        node_vec = F.dropout(node_vec,
                             network.dropout,
                             training=network.training)
        pivot_vec = node_vec[sampled_node_idx]

        first_node_pivot_adj, first_pivot_adj = cur_node_pivot_adj, cur_pivot_adj
        first_init_agg_vec = network.encoder.graph_encoders[0](
            init_node_vec,
            first_node_pivot_adj,
            pivot_mp=True,
            batch_norm=False)
        # Add mid GNN layers
        for encoder in network.encoder.graph_encoders[1:-1]: 
            node_vec = (1 - gl_handler.graph_skip_conn) * encoder(node_vec, cur_node_pivot_adj, pivot_mp=True,
                                                                  batch_norm=False) + \
                gl_handler.graph_skip_conn * \
                encoder(node_vec, init_adj, pivot_mp=False, batch_norm=False)
            
            if encoder.bn is not None:
                node_vec = encoder.compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec,
                                 network.dropout,
                                 training=network.training)
            pivot_vec = node_vec[sampled_node_idx]

        # Compute output via node-pivot-node message passing
        
        
        output = (1 - gl_handler.graph_skip_conn) * network.encoder.graph_encoders[-1](node_vec,
                                                                                       cur_node_pivot_adj,
                                                                                       pivot_mp=True,
                                                                                       batch_norm=False) + \
            gl_handler.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj,
                                                                            pivot_mp=False, batch_norm=False)
        output = F.log_softmax(output, dim=-1)
      
        score = self.model.score_func(labels[idx], output[idx])
     
        loss1 = self.model.criterion(output[idx], labels[idx]) 
       

        if self.config['graph_learn_regularization']:  
            loss1 += self.add_graph_loss(cur_pivot_adj, init_pivot_vec)
       
        max_iter_ = self.config.get('max_iter', 10)  
            
       

        if training:
            eps_adj = float(
                self.config.get('eps_adj', 0)
            )  
        else:
            eps_adj = float(
                self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))

        pre_node_pivot_adj = cur_node_pivot_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(
                cur_node_pivot_adj, pre_node_pivot_adj,
                cur_node_pivot_adj).item() > eps_adj) and iter_ < max_iter_: 
            iter_ += 1
           
            pre_node_pivot_adj = cur_node_pivot_adj
            # Compute n x p node-pivot relationship matrix
            cur_node_pivot_adj = gl_handler.learn_graph(
                node_vec, pivot_features=pivot_vec)

            # Compute p x p pivot graph
            cur_pivot_adj = compute_pivot_adj(cur_node_pivot_adj)

            cur_agg_vec = network.encoder.graph_encoders[0](
                init_node_vec,
                cur_node_pivot_adj,
                pivot_mp=True,
                batch_norm=False)
            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (
                    1 - update_adj_ratio) * first_init_agg_vec

            node_vec = (1 - gl_handler.graph_skip_conn) * cur_agg_vec + \
                gl_handler.graph_skip_conn * init_agg_vec
            if network.encoder.graph_encoders[0].bn is not None:
                node_vec = network.encoder.graph_encoders[0].compute_bn(
                    node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec,
                                 self.config.get('gl_dropout', 0),
                                 training=network.training)
            pivot_vec = node_vec[sampled_node_idx]  

            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                mid_cur_agg_vec = encoder(node_vec,
                                          cur_node_pivot_adj,
                                          pivot_mp=True,
                                          batch_norm=False)
                if update_adj_ratio is not None:
                    mid_first_agg_vecc = encoder(node_vec,
                                                 first_node_pivot_adj,
                                                 pivot_mp=True,
                                                 batch_norm=False)
                    mid_cur_agg_vec = update_adj_ratio * mid_cur_agg_vec + (
                        1 - update_adj_ratio) * mid_first_agg_vecc
                  
                node_vec = (1 - gl_handler.graph_skip_conn) * mid_cur_agg_vec + \
                    gl_handler.graph_skip_conn * \
                    encoder(node_vec, init_adj,
                            pivot_mp=False, batch_norm=False)
                
                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec,
                                     self.config.get('gl_dropout', 0),
                                     training=network.training)
                pivot_vec = node_vec[sampled_node_idx]

            cur_agg_vec = network.encoder.graph_encoders[-1](
                node_vec,
                cur_node_pivot_adj,
                pivot_mp=True,
                batch_norm=False)  
            if update_adj_ratio is not None:
                first_agg_vec = network.encoder.graph_encoders[-1](
                    node_vec,
                    first_node_pivot_adj,
                    pivot_mp=True,
                    batch_norm=False)
                cur_agg_vec = update_adj_ratio * cur_agg_vec + (
                    1 - update_adj_ratio) * first_agg_vec

            # Add message passing results on original graph structure
            output = (1 - gl_handler.graph_skip_conn) * cur_agg_vec + \
               gl_handler.graph_skip_conn * network.encoder.graph_encoders[-1](node_vec, init_adj,
                                                                              pivot_mp=False, batch_norm=False)
            output = F.log_softmax(output, dim=-1)

            
            if self.config['graph_learn_regularization']:
                # sample
                sample_times = 5
                loss_list = []
                prob_list = []

                distribution = distribute(cur_pivot_adj)
                for i in range(sample_times):
                    pivot_adj_sample = torch.bernoulli(distribution).to(
                        cur_pivot_adj.device)
                    prob_list.append(get_prob(distribution, pivot_adj_sample))
                    with torch.no_grad():
                        loss_list.append(
                            self.get_pg_loss(pivot_adj_sample.to_sparse(),
                                             init_pivot_vec))
                loss_list, prob_list = torch.stack(loss_list), torch.stack(
                    prob_list)
                loss_mean = torch.mean(loss_list)

                loss += torch.sum(
                    (loss_list - loss_mean) * prob_list) / sample_times
                loss += self.add_graph_loss(cur_pivot_adj, init_pivot_vec)

            if not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(
                    cur_node_pivot_adj -
                    pre_node_pivot_adj) * self.config.get('graph_learn_ratio')

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
       
        
        if training:
            self.model.optimizer.zero_grad()
            
            loss.backward(retain_graph=True)
            self.model.clip_grad()
            self.model.optimizer.step()
         
            if train_gl:
                gl_handler.clip_grad()
                opt_gl_handler.step()

        self._update_metrics(loss.item(), {
            'nloss': -loss.item(),
            self.model.metric_name: score
        },
            1,
            training=training)
        

        
        return node_vec, support_idx, query_idx, support_labels, query_lable
     
   
    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(),
                                                   metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True

    def add_graph_loss(self, out_adj, features): 
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        if self.config['smoothness_ratio'] > 0:
            graph_loss += self.config['smoothness_ratio'] * torch.trace(
                torch.mm(features.transpose(-1, -2), torch.mm(
                    L, features))) / int(np.prod(out_adj.shape))

        if self.config['sparsity_ratio'] > 0:
            graph_loss += self.config['sparsity_ratio'] * torch.sum(
                torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape)) 
        return graph_loss

    def get_pg_loss(self, out_adj, features):
        '''
        get rewards for policy gradient update
        '''
        graph_loss = 0

        if out_adj._nnz() > 0:
            L = torch.diagflat(torch.sparse.sum(
                out_adj, -1).to_dense()) - out_adj
            if self.config['smoothness_ratio'] > 0:
                graph_loss += self.config['smoothness_ratio'] * torch.trace(
                    torch.mm(features.transpose(-1, -2), torch.spmm(
                        L, features))) / int(np.prod(out_adj.shape))

        if self.config['sparsity_ratio'] > 0:
            graph_loss += self.config['sparsity_ratio'] * torch.sparse.sum(
                torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss


def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER) 
    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


def get_prob(p, sample):
    p1 = torch.log(p + Constants.VERY_SMALL_NUMBER) * sample
    p2 = torch.log(1 - p + Constants.VERY_SMALL_NUMBER) * (1 - sample)
    return torch.sum(p1 + p2)


def distribute(adj):
    return adj / (torch.max(adj, dim=-1)[0].unsqueeze(1) +
                  Constants.VERY_SMALL_NUMBER)
    

