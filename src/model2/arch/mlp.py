#!/usr/bin/env python3.6
'''Multi-Layer Perceptron Architecture.

For causal discriminative model and the corresponding generative model.
'''
import sys, os
import json
import torch as tc
import torch.nn as nn
import math
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from ..distr import tensorify, is_same_tensor, wrap4_multi_batchdims

class Linear(nn.Module):

    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tc.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(tc.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        return nn.functional.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

def init_linear(nnseq, wmean, wstd, bval):
    for mod in nnseq:
        if type(mod) is nn.Linear:
            mod.weight.data.normal_(wmean, wstd)
            mod.bias.data.fill_(bval)

def mlp_constructor(dims, actv = "Sigmoid", lastactv = True): # `Sequential()`, or `Sequential(*[])`, is the identity map for any shape!
    if type(actv) is str: actv = getattr(nn, actv)
    if len(dims) <= 1: return nn.Sequential()
    else: return nn.Sequential(*(
        sum([[Linear(dims[i], dims[i+1]), actv()] for i in range(len(dims)-2)], []) + \
        [Linear(dims[-2], dims[-1])] + ([actv()] if lastactv else [])
    ))
    
def mlp_constructor_2(dims, actv = "Sigmoid", lastactv = True): # `Sequential()`, or `Sequential(*[])`, is the identity map for any shape!
    if type(actv) is str: actv = getattr(nn, actv)
    if len(dims) <= 1: return nn.Sequential()
    else: return nn.Sequential(*(
        sum([[nn.Linear(dims[i], dims[i+1]), actv()] for i in range(len(dims)-2)], []) + \
        [nn.Linear(dims[-2], dims[-1])] + ([actv()] if lastactv else [])
    ))


class MLPBase(nn.Module):
    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()
    def load_or_save(self, filename):
        dirname = "init_models_mlp/"
        os.makedirs(dirname, exist_ok=True)
        path = dirname + filename
        if os.path.exists(path): self.load(path)
        else: self.save(path)

class MLP(MLPBase):
    def __init__(self, dims, actv = "Sigmoid"):
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLP, self).__init__()
        self.f_x2y = mlp_constructor(dims, actv, lastactv = False)
    def forward(self, x): return self.f_x2y(x).squeeze(-1)

class MLPsvy1x(MLPBase):
    def __init__(self, dim_x, dims_postx2prev, dim_v, dim_parav, dims_postv2s, dims_posts2prey, dim_y, actv = "Sigmoid",
            std_v1x_val: float=1., std_s1vx_val: float=-1., after_actv: bool=True): # if <= 0, then learn the std.
        """
                       /->   v   -\
        x ====> prev -|            |==> s ==> y
                       \-> parav -/
        """
        
        super(MLPsvy1x, self).__init__()
        self.vars = nn.ParameterList()
        self.graph_conv = []
        
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_v, self.dim_y = dim_x, dim_v, dim_y
        dim_prev, dim_s = dims_postx2prev[-1], dims_postv2s[-1]
        self.dim_prev, self.dim_s = dim_prev, dim_s
        
        self.shape_x, self.shape_v, self.shape_s = (dim_x,), (dim_v,), (dim_s,)
        self.dims_postx2prev, self.dim_parav, self.dims_postv2s, self.dims_posts2prey, self.actv \
                = dims_postx2prev, dim_parav, dims_postv2s, dims_posts2prey, actv
        f_x2prev = mlp_constructor([dim_x] + dims_postx2prev, actv)
        self.graph_conv.append(f_x2prev)
        self.vars.append(f_x2prev[0].weight.data)
        self.vars.append(f_x2prev[0].bias.data)
        
        
        if after_actv:
            f_prev2v = nn.Sequential( Linear(dim_prev, dim_v), actv() )
            f_prev2parav = nn.Sequential(Linear(dim_prev, dim_parav), actv() ) #3
            f_vparav2s = mlp_constructor([dim_v + dim_parav] + dims_postv2s, actv) #4
            f_s2y = mlp_constructor_2([dim_s] + dims_posts2prey + [dim_y], actv, lastactv = False)#5
           
            
            
        else:
            f_prev2v = Linear(dim_prev, dim_v)
            f_prev2parav = Linear(dim_prev, dim_parav)
            f_vparav2s = nn.Sequential( actv(), mlp_constructor([dim_v + dim_parav] + dims_postv2s, actv, lastactv = False) )
            f_s2y = nn.Sequential( actv(), mlp_constructor([dim_s] + dims_posts2prey + [dim_y], actv, lastactv = False) )
            
        self.graph_conv.append(f_prev2v)
        self.graph_conv.append(f_prev2parav)
        self.graph_conv.append(f_vparav2s)
        self.graph_conv.append(f_s2y)
        
        
        
        list = [f_prev2v[0].weight.data, f_prev2v[0].bias.data, f_prev2parav[0].weight.data, f_prev2parav[0].bias.data, f_vparav2s[0].weight.data, f_vparav2s[0].bias.data]
        for item in list:
            self.vars.append(item)
       
        
        self.std_v1x_val = std_v1x_val; self.std_s1vx_val = std_s1vx_val
        self.learn_std_v1x = std_v1x_val <= 0 if type(std_v1x_val) is float else (std_v1x_val <= 0).any()
        self.learn_std_s1vx = std_s1vx_val <= 0 if type(std_s1vx_val) is float else (std_s1vx_val <= 0).any()

        self._prev_cache = self._x_cache_prev = None
        self._v_cache = self._x_cache_v = None
        self._parav_cache = self._x_cache_parav = None

        
        if self.learn_std_s1vx:
            self.nn_std_s = nn.Sequential(
                    nn.BatchNorm1d(dim_v + dim_parav),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp_constructor(
                        [dim_v + dim_parav] + dims_postv2s,
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = wrap4_multi_batchdims(self.nn_std_s, ndim_vars=1)

    
    
    def _get_prev(self, x, vars=None): #1
        if vars is None:
            vars = self.vars
            
        if not is_same_tensor(x, self._x_cache_prev): #zhixing
            self._x_cache_prev = x
            x2prev = self.graph_conv[0][0]
            _prev_cache = x2prev(x, vars[0], vars[1])
            _prev_cache = self.graph_conv[0][1](_prev_cache)
            return _prev_cache
        else:
            return self._prev_cache
            
    

    def v1x(self, x, vars=None): #2
        if vars is None:
            vars = self.vars
        if not is_same_tensor(x, self._x_cache_v):#执行
            self._x_cache_v = x
            #x2prev = self.graph_conv[0][0]
            prev = self._get_prev(x, vars)
            f_prev2v = self.graph_conv[1][0]
            if not is_same_tensor(x, self._x_cache_v):#执行
                self._x_cache_v = x
                self._v_cache = f_prev2v(prev, vars[2], vars[3]) 
                self._v_cache = self.graph_conv[1][1](self._v_cache)
        return self._v_cache
    
    def std_v1x(self, x, vars=None):
        if vars is None:
            vars = self.vars
        #print(self.learn_std_v1x)
        if self.learn_std_v1x:
            prev = self._get_prev(x, vars) 
            f_prev2v = self.graph_conv[1][0]
            v = f_prev2v(prev, vars[2], vars[3])
            v= self.graph_conv[1][1](v)
            return v
        else:
            return tensorify(x.device, self.std_v1x_val)[0].expand(x.shape[:-1]+(self.dim_v,))

       

    def _get_parav(self, x, vars=None):
        if vars is None:
            vars = self.vars
        if not is_same_tensor(x, self._x_cache_parav):
            self._x_cache_parav = x
            f_prev2parav = self.graph_conv[2][0]
            self._parav_cache =f_prev2parav(self._get_prev(x, vars), vars[4], vars[5])
            self._parav_cache = self.graph_conv[2][1](self._parav_cache)
        return self._parav_cache

    def s1vx(self, v, x, vars=None):
        
        if vars is None:
            vars = self.vars
        parav = self._get_parav(x, vars)
        f_vparav2s = self.graph_conv[3][0]
        s = f_vparav2s(tc.cat([v, parav], dim=-1), vars[6], vars[7]) 
        s = self.graph_conv[3][1](s) 

        return s
    
    def std_s1vx(self, v, x, vars=None):
        if vars is None:
            vars = self.vars
      
        if self.learn_std_s1vx:
            
            parav = self._get_parav(x, vars)
            std_s = self.f_std_s(tc.cat([v, parav], dim=-1))
            return s
        else:
            return tensorify(x.device, self.std_s1vx_val)[0].expand(x.shape[:-1]+(self.dim_s,))

    def s1x(self, x, vars=None):
        if vars is None:
            vars = self.vars
        return self.s1vx(self.v1x(x, vars), x, vars)
    
    def std_s1x(self, x, vars=None):
        if vars is None:
            vars = self.vars
        return self.std_s1vx(self.v1x(x, vars), x, vars)

    def y1s(self, s):
        
        f_s2y = self.graph_conv[4][0].to(s.device)
   
        y = f_s2y(s).squeeze(-1)
        return y 

    def ys1x(self, x, vars=None):
        if vars is None:
            vars = self.vars
        s = self.s1x(x, vars)
        return self.y1s(s, vars), s
    
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
    
 
        x2prev = self.graph_conv[0][0]
  
        if not is_same_tensor(x, self._x_cache_prev):
            self._x_cache_prev = x
            self._prev_cache = x2prev(x, vars[0], vars[1])
        prev = self.graph_conv[0][1](self._prev_cache) 
        
        
        f_prev2v = self.graph_conv[1][0]
        if not is_same_tensor(x, self._x_cache_v):
            self._x_cache_v = x
            self._v_cache = f_prev2v(prev, vars[2], vars[3])
        v = self.graph_conv[1][1](self._v_cache)
        
        f_prev2parav = self.graph_conv[2][0]
        if not is_same_tensor(x, self._x_cache_parav):
            self._x_cache_parav = x
            self._parav_cache = f_prev2parav(prev, vars[4], vars[5])
        parav = self.graph_conv[2][1](self._parav_cache)
        
        f_vparav2s = self.graph_conv[3][0]
        s = f_vparav2s(tc.cat([v, parav], dim=-1), vars[6], vars[7]) 
        s =    self.graph_conv[3][1](s)
        return s, v
    
    def parameters(self):
        return self.vars

   
    



class MLPx1sv(MLPBase):
    def __init__(self, dim_s = None, dims_pres2parav = None, dim_v = None, dims_prev2postx = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_v is None: dim_v = discr.dim_v
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2parav is None: dims_pres2parav = discr.dims_postv2s[::-1][1:] + [discr.dim_parav]
        if dims_prev2postx is None: dims_prev2postx = discr.dims_postx2prev[::-1]
        
        super(MLPx1sv, self).__init__()
        self.para = nn.ParameterList()
        self.graph_conv = []
     
        self.dim_s, self.dim_v, self.dim_x = dim_s, dim_v, dim_x
        self.dims_pres2parav, self.dims_prev2postx, self.actv = dims_pres2parav, dims_prev2postx, actv
        f_s2parav = mlp_constructor([dim_s] + dims_pres2parav, actv)
        self.graph_conv.append(f_s2parav)
        self.para.append(f_s2parav[0].weight.data)
        self.para.append(f_s2parav[0].bias.data)
        
        f_vparav2x = mlp_constructor([dim_v + dims_pres2parav[-1]] + dims_prev2postx + [dim_x], actv)

        self.graph_conv.append(f_vparav2x)
       
        
        self.para.append(f_vparav2x[0].weight.data)
        self.para.append(f_vparav2x[0].bias.data)
        self.para.append(f_vparav2x[2].weight.data)
        self.para.append(f_vparav2x[2].bias.data)
        

    def x1sv(self, s, v, vars=None): 
   
        if vars is None:
            vars = self.para
            idx = 0
        else:
            idx = 10
            
        f_s2parav = self.graph_conv[0][0]
        s2parav = f_s2parav(s, vars[idx], vars[idx+1])
        s2parav = self.graph_conv[0][1](s2parav) 
        
        f_vparav2x_1 = self.graph_conv[1][0]
        vparav2x_1 = f_vparav2x_1(tc.cat([v, s2parav], dim=-1), vars[idx+2], vars[idx+3])
        vparav2x_1 = self.graph_conv[1][1](vparav2x_1)
        
        f_vparav2x_2 = self.graph_conv[1][2]
        x = f_vparav2x_2(vparav2x_1, vars[idx+4], vars[idx+5])
        x = self.graph_conv[1][3](x)
 
        
          
        return x
   
    
    def forward(self, s, v, vars=None): 
        if vars is None:
            vars = self.para
            idx=0
        else:
            idx = 8

        f_s2parav = self.graph_conv[0][0]
        s2parav = f_s2parav(s, vars[idx], vars[idx+1])
        s2parav = self.graph_conv[0][1](s2parav) 
        
        f_vparav2x_1 = self.graph_conv[1][0]
        vparav2x_1 = f_vparav2x_1(tc.cat([v, s2parav], dim=-1), vars[idx+2], vars[idx+3])
        vparav2x_1 = self.graph_conv[1][1](vparav2x_1)
        
        f_vparav2x_2 = self.graph_conv[1][2]
        x = f_vparav2x_2(vparav2x_1, vars[idx+4], vars[idx+5])
        x = self.graph_conv[1][3](x)
   
        return x
    
    def parameters(self):
        return self.para

class MLPx1s(MLPBase):
    def __init__(self, dim_s = None, dims_pres2postx = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postx is None:
            dims_pres2postx = discr.dims_postv2s[::-1][1:] + [discr.dim_v + discr.dim_parav] + discr.dims_postx2prev[::-1]
        super(MLPx1s, self).__init__()
        self.dim_s, self.dim_x, self.dims_pres2postx, self.actv = dim_s, dim_x, dims_pres2postx, actv
        self.f_s2x = mlp_constructor([dim_s] + dims_pres2postx + [dim_x], actv)

    def x1s(self, s): return self.f_s2x(s)
    def forward(self, s): return self.x1s(s)

class MLPv1s(MLPBase):
    def __init__(self, dim_s = None, dims_pres2postv = None, dim_v = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_v is None: dim_v = discr.dim_v
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postv is None: dims_pres2postv = discr.dims_postv2s[::-1][1:]
        super(MLPv1s, self).__init__()
        self.dim_s, self.dim_v, self.dims_pres2postv, self.actv = dim_s, dim_v, dims_pres2postv, actv
        self.f_s2v = mlp_constructor([dim_s] + dims_pres2postv + [dim_v], actv)

    def v1s(self, s): return self.f_s2v(s)
    def forward(self, s): return self.v1s(s)

def create_discr_from_json(stru_name: str, dim_x: int, dim_y: int, actv: str=None,
        std_v1x_val: float=-1., std_s1vx_val: float=-1., after_actv: bool=True, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPsvy1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPsvy1x(dim_x=dim_x, dim_y=dim_y, std_v1x_val=std_v1x_val, std_s1vx_val=std_s1vx_val,
            after_actv=after_actv, **stru)

def create_gen_from_json(model_type: str="MLPx1sv", discr: MLPsvy1x=None, stru_name: str=None, dim_x: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_x=dim_x, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_x=dim_x, discr=discr, **stru)

