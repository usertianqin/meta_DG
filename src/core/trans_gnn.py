import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from .graph_learner_handler import GL_handler
from .model_handler import ModelHandler
from ..model2.func import ood_methods, get_models, ParamGroupsCollector, ShrinkRatio, evaluate_acc
import os
import copy
from copy import deepcopy
from ..model2.distr import edic
import numpy as np



# Rest of your code goes here

params = ['seed', 'num_pivots', 'hidden_size', 'graph_metric_type', 'graph_skip_conn',
          'update_adj_ratio', 'smoothness_ratio', 'degree_ratio', 'sparsity_ratio', 'graph_learn_num_pers', 'learning_rate', 'weight_decay', 'max_iter', 'eps_adj', 'max_episodes', 'graph_learn_regularization', 'prob_del_edge', 'dropout', 'gl_dropout', 'feat_adj_dropout']


class trainer:
    '''
    Initialize models and coordinates the training and transferring process.
    '''

    def __init__(self, config, ag):
        self.device = get_device(config)

        # use dropout value if feat_adj_dropout<0 or gl_dropout<0
        if config.get('feat_adj_dropout', -1) < 0:
            config['feat_adj_dropout'] = config['dropout']
        if config.get('gl_dropout', -1) < 0:
            config['gl_dropout'] = config['dropout']

        dataset_name_list = config['dataset_name'].split('+')

        if 'trans_dataset_name' in config:
            trans_dataset_name_list = config.get(
                'trans_dataset_name', '').split('+')
        else:
            trans_dataset_name_list = []

        print_str = config['dataset_name']
        if config.get('trans_dataset_name', '') != '':
            print_str += '_trans_'+config.get('trans_dataset_name', '')

        log_folder = 'results/'
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        log_path = log_folder+print_str+'.txt'
        self.log_f = open(log_path, 'a')
        self.log_f.write(f'=============={print_str}================\n')
        print(f'=============={print_str}================')

        model_folder = f'results/{print_str}'
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        param_str = ''
        for key in params:
            if key in config:
                param_str += f'{key}:{config[key]}, '
        self.log_f.write(f'{param_str}\n')
        self.config = config
        self.ag = ag
        self.logger = None
        self.model_handler_list = []
        self.trans_model_handler_list = []

        for dataset in dataset_name_list:
            self.model_handler_list.append(ModelHandler(
                self.config, self.ag, dataset_name=dataset, log_f=self.log_f, save_dir=model_folder))
        

        for dataset in trans_dataset_name_list:
            self.trans_model_handler_list.append(ModelHandler(
                self.config, self.ag, dataset_name=dataset, log_f=self.log_f, save_dir=model_folder))

        self.gl_handler = GL_handler(self.config, save_dir=model_folder) 
        self.gl_handler.to(self.device)
        self.opt_gl_handler = init_optimizer(self.gl_handler, config)
        dim_x = self.model_handler_list[0].dim_x
        dim_y = self.model_handler_list[0].dim_y
        self.discr, self.gen, self.frame = get_models('mlp', edic(locals()) | vars(self.ag), None, device=self.device)
        self.params = list(self.discr.parameters()) + list(self.gen.parameters())
        self.opt_vae = optim.Adam(self.params, lr=self.config['learning_rate'])    
        self.dim_y = dim_y
        self.discr_path = os.path.join(self.model_handler_list[0].dirname, 'discr.pth')
        self.gen_path = os.path.join(self.model_handler_list[0].dirname, 'gen.pth')
        

   
     
    def Meta(self):
        
        losses_s = [0 for _ in range(self.ag.update_step)] 
        losses_q = [0 for _ in range(self.ag.update_step + 1)] 
     
        for model_handler in self.model_handler_list: 
            
            lossfn = ood_methods(self.discr, self.frame, self.ag, model_handler.dim_y, cnbb_actv="Sigmoid")
            format_str = f'{model_handler.dataset}, '
       
            self.log_f.write(format_str)
            
            output, support_idx, query_idx, support_labels, query_lables = model_handler.run_epoch(model_handler.train_loader,
                self.gl_handler, self.opt_gl_handler, training=True, train_gl=True) 
            output.detach()
            x_s = output[support_idx] 
            x_q = output[query_idx]  
            
        
            vars = None
            s, v = self.discr(x_s.to(self.device), vars)
            f_c = Classifier(s.shape[1], model_handler.dim_y, 100, 1)
            opt_fc = optim.Adam(f_c.parameters(), lr=self.config['learning_rate'])
            f_c.to(self.device)
            x_rec = self.gen(s,v, vars)
            y_pre = f_c(s)
            
            data_s = (x_s.to(self.device), support_labels.to(self.device), y_pre, x_rec)
           
            loss = lossfn(*data_s)
          
            losses_s[0] += loss
        
           
            distr_para = self.discr.parameters()
            gen_para = self.gen.parameters()
            para = list(distr_para) + list(gen_para)
            
            grad_d = torch.autograd.grad(loss, para, retain_graph=True) 
            fast_weights_d = list(map(lambda p: p[1] - self.ag.update_lr * p[0], zip(grad_d, para))) 
            
            with torch.no_grad():
                s, v = self.discr(x_q.to(self.device), para)
                y_pre = f_c(s)
                x_rec = self.gen(s, v, para)
                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                losses_q[0] += loss_q
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
                
            with torch.no_grad():
                s, v = self.discr(x_q.to(self.device), fast_weights_d)
                y_pre = f_c(s)
                x_rec = self.gen( s, v, fast_weights_d)
                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                losses_q[1] += loss_q
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
              
            for k in range(1, self.ag.update_step): 
                
                s, v = self.discr(x_s.to(self.device), fast_weights_d)
                y_pre = f_c(s)
                x_rec = self.gen( s, v, fast_weights_d)
                
                data_s = (x_s.to(self.device), support_labels.to(self.device), y_pre, x_rec)
                loss = lossfn(*data_s)
                losses_s[k] += loss
                
                grad_d = torch.autograd.grad(loss, fast_weights_d, retain_graph=True) 
                fast_weights_d = list(map(lambda p: p[1] - self.ag.update_lr * p[0], zip(grad_d, fast_weights_d))) 
                s, v = self.discr(x_q.to(self.device), fast_weights_d)
                y_pre = f_c(s)
                x_rec = self.gen( s, v, fast_weights_d)
           

                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
                losses_q[k + 1] += loss_q
                
                opt_fc.zero_grad()
                loss_q.backward(retain_graph=True)
                opt_fc.step()
                
        loss_q = losses_q[-1] / len(self.model_handler_list)
        if torch.isnan(loss_q):
            pass
        else:    
            
            self.opt_vae.zero_grad()
            loss_q.backward()
            self.opt_vae.step()

        torch.save(self.discr.state_dict(), self.discr_path)
        torch.save(self.gen.state_dict(), self.gen_path)
    
    def finetunning(self):
        discr = self.discr
        gen = self.gen
        frame = self.frame
       
        acc_final = []
        for model_handler in self.trans_model_handler_list:
            lossfn = ood_methods(discr, frame, self.ag, model_handler.dim_y, cnbb_actv="Sigmoid")
            corrects = [0 for _ in range(self.ag.update_step_test + 1)]
            format_str = f'{model_handler.dataset}, '
            print(format_str) 
            output, support_idx, query_idx, support_labels, query_lables = model_handler.run_epoch(model_handler.test_loader,
                self.gl_handler, self.opt_gl_handler, training=True, train_gl=True) #
            output.detach()
            x_s = output[support_idx] 
            x_q = output[query_idx] 
         
            s, v = discr(x_s.to(self.device))
            fc = Classifier(s.shape[1], model_handler.dim_y, 100, 1)
            opt_fc = optim.Adam(fc.parameters(), lr=self.config['learning_rate'])
            fc.to(self.device)
            y_pre = fc(s)
            x_rec = gen(s,v)
            
            data_s = (x_s.to(self.device), support_labels.to(self.device), y_pre, x_rec)
            loss = lossfn(*data_s)
            distr_para = discr.parameters()
            gen_para = gen.parameters()
            para = list(distr_para) + list(gen_para)
            grad_d = torch.autograd.grad(loss, para, retain_graph=True)
            fast_weights_d = list(map(lambda p: p[1] - self.ag.update_lr * p[0], zip(grad_d, para))) 
            
            
            with torch.no_grad():
            
                s, v =discr(x_q.to(self.device), para)
                y_pre = fc(s)
                x_rec = gen(s, v, para)
                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
                corrects[0] = corrects[0] + acc_q
                
            with torch.no_grad():
               
                s, v = discr(x_q.to(self.device), fast_weights_d)
                y_pre = fc(s)
                x_rec = gen( s, v, fast_weights_d)
                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
                corrects[1] = corrects[1] + acc_q
                
            for k in range(1, self.ag.update_step_test): 
            # 1. run the i-th task and compute loss for k=1~K-1
                
           
                s, v = discr(x_s.to(self.device), fast_weights_d)
                y_pre = fc(s)
                x_rec = gen( s, v, fast_weights_d)
               
                data_s = (x_s.to(self.device), support_labels.to(self.device), y_pre, x_rec)
                loss = lossfn(*data_s)
        
                
                grad_d = torch.autograd.grad(loss, fast_weights_d, retain_graph=True) 
                fast_weights_d = list(map(lambda p: p[1] - self.ag.update_lr * p[0], zip(grad_d, fast_weights_d))) 
         
                s, v = discr(x_q.to(self.device), fast_weights_d)
                y_pre = fc(s)
                x_rec =gen( s, v, fast_weights_d)
           

                data_q = (x_q.to(self.device), query_lables.to(self.device), y_pre, x_rec)
                loss_q = lossfn(*data_q)
                acc_q = compute_acc(y_pre, query_lables, model_handler.dim_y)
                corrects[k + 1] = corrects[k + 1] + acc_q

                opt_fc.zero_grad()
                loss_q.backward(retain_graph=True)
                opt_fc.step()
                   
            acc_q = max(corrects)
            acc_final.append(acc_q)
    
        return acc_final
        

    
def joint_train(config, ag):
    max_episodes = config.get('max_episodes', 1)
    acc_all_run = [0 for _ in range(ag.test_num)]
    for episode in range(max_episodes):
        episode_str = f'==================episode[{episode + 1}/{max_episodes}]=========================='
        print(episode_str + '\n')

        my_trainer = trainer(config, ag)
        max_acc = [0 for _ in range(ag.test_num)]
        for epoch in range(config['max_epochs']):
            my_trainer.Meta()
            if (epoch+1) % 5 == 0:
                epoch_str = f'==================epoch[{epoch + 1}/{config["max_epochs"]}]=========================='
                print(epoch_str + '\n')
                accs = my_trainer.finetunning()
                for i in range(len(accs)):
                    if accs[i] > max_acc[i]:
                        max_acc[i] = accs[i]
                print("max_acc", max_acc)
        for i in range(len(max_acc)):
            if max_acc[i] > acc_all_run[i]:
                acc_all_run[i] = max_acc[i]
        
        
    print("acc_all_run", acc_all_run) 

def init_optimizer(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(parameters, config['learning_rate'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adamax':
        optimizer = optim.Adamax(parameters, lr=config['learning_rate'])
    else:
        raise RuntimeError('Unsupported optimizer: %s' % config['optimizer'])

    return optimizer


def get_device(config):
    if not config['no_cuda'] and torch.cuda.is_available():
        device = torch.device(
            'cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    return device
def compute_acc(logits, y, dim_y):
    if dim_y ==1:
        is_binary=True
    else:
        is_binary=False
    ypred = (logits > 0).long() if is_binary else logits.argmax(dim=-1)
    return (ypred == y).float().mean().item()
    
class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h=64, n_layers=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_h = dim_h
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim_in, dim_h))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(dim_h, dim_h))
        self.layers.append(nn.Linear(dim_h, dim_out))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        logits = self.layers[-1](x).squeeze(-1)
        return logits