#!/usr/bin/env python3.6
import os, sys
import argparse
from copy import deepcopy
from functools import partial
import tqdm
import torch as tc
import torchvision as tv


from .distr.utils import edic
from .arch import mlp
from .methods import SemVar, SupVAE
from .utils import Averager, unique_filename, boolstr, zip_longer # This imports from 'utils/__init__.py'


from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# tc.autograd.set_detect_anomaly(True)
MODES_OOD_NONGEN = {"discr", "cnbb"}
MODES_OOD_GEN = {"svae", "svgm", "svgm-ind"}
MODES_DA_NONGEN = {"dann", "cdan", "dan", "mdd", "bnm"}
MODES_DA_GEN = {"svae-da", "svgm-da", "svae-da2", "svgm-da2"}

MODES_OOD = MODES_OOD_NONGEN | MODES_OOD_GEN
MODES_DA = MODES_DA_NONGEN | MODES_DA_GEN
MODES_GEN = MODES_OOD_GEN | MODES_DA_GEN
MODES_TWIST = {"svgm-ind", "svae-da", "svgm-da"}

# Init models
def auto_load(dc_vars, names, ckpt):
    if ckpt:
        if type(names) is str: names = [names]
        for name in names:
            model = dc_vars[name]
            model.load_state_dict(ckpt[name+'_state_dict'])
            if hasattr(model, 'eval'): model.eval()

def get_frame(discr, gen, dc_vars, device = None, discr_src = None):

    if type(dc_vars) is not edic: dc_vars = edic(dc_vars)
    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)
    shape_v = discr.shape_v if hasattr(discr, "shape_v") else (dc_vars['dim_v'],)
    std_v1x = discr.std_v1x if hasattr(discr, "std_v1x") else dc_vars['qstd_v']
    std_s1vx = discr.std_s1vx if hasattr(discr, "std_s1vx") else dc_vars['qstd_s']
    std_s1x = discr.std_s1x if hasattr(discr, "std_s1x") else dc_vars['qstd_s']
    mode = dc_vars['mode']

    if mode.startswith("svgm"):
        q_args_stem = (discr.v1x, std_v1x, discr.s1vx, std_s1vx)

    else: return None
    if mode == "svgm-da2" and discr_src is not None:
        q_args = ( discr_src.v1x, discr_src.std_v1x if hasattr(discr_src, "std_v1x") else dc_vars['qstd_v'],
                discr_src.s1vx, discr_src.std_s1vx if hasattr(discr_src, "std_s1vx") else dc_vars['qstd_s'],
            ) + q_args_stem
    elif mode == "svae-da2" and discr_src is not None:
        q_args = ( discr_src.s1x, discr_src.std_s1x if hasattr(discr_src, "std_s1x") else dc_vars['qstd_s'],
            ) + q_args_stem
    elif mode in MODES_TWIST: # svgm-ind, svgm-da
        q_args = (None,)*len(q_args_stem) + q_args_stem
    else: #  svgm
        q_args = q_args_stem + (None,)*len(q_args_stem)

    if mode.startswith("svgm"):
        #(s,v)生成模型
        frame = SemVar( shape_s, shape_v, shape_x, dc_vars['dim_y'],
                gen.x1sv, dc_vars['pstd_x'], discr.y1s, *q_args,
                *dc_vars.sublist(['mu_s', 'sig_s', 'mu_v', 'sig_v', 'corr_sv']),
                mode in MODES_DA, *dc_vars.sublist(['src_mvn_prior', 'tgt_mvn_prior']), device=device )
    elif mode.startswith("svae"):
        frame = SupVAE( shape_s, shape_x, dc_vars['dim_y'],
                gen.x1s, dc_vars['pstd_x'], discr.y1s, *q_args,
                *dc_vars.sublist(['mu_s', 'sig_s']),
                mode in MODES_DA, *dc_vars.sublist(['src_mvn_prior', 'tgt_mvn_prior']), device=device )
    return frame

def get_discr(archtype, dc_vars):
    if archtype == "mlp":
        discr = mlp.create_discr_from_json(
                *dc_vars.sublist([
                    'discrstru', 'dim_x', 'dim_y', 'actv',
                    'qstd_v', 'qstd_s', 'after_actv']),
                jsonfile=dc_vars['mlpstrufile']
            )
    else: raise ValueError(f"unknown `archtype` '{archtype}'")
    return discr

def get_gen(archtype, dc_vars, discr):
    if dc_vars['mode'].startswith("svgm"):
        if archtype == "mlp":
            gen = mlp.create_gen_from_json(
                    "MLPx1sv", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
    elif dc_vars['mode'].startswith("svae"):
        if archtype == "mlp":
            gen = mlp.create_gen_from_json(
                    "MLPx1s", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
    return gen

def get_models(archtype, dc_vars, ckpt = None, device = None):
   
    if type(dc_vars) is not edic:
        print("type(dc_vars) is not edic") 
        dc_vars = edic(dc_vars)
    discr = get_discr(archtype, dc_vars)
    if ckpt is not None: auto_load(locals(), 'discr', ckpt)
    discr.to(device)
    if dc_vars['mode'] in MODES_GEN:
        gen = get_gen(archtype, dc_vars, discr)
        if ckpt is not None: auto_load(locals(), 'gen', ckpt)
        gen.to(device)
        if dc_vars['mode'].endswith("-da2"):
            discr_src = get_discr(archtype, dc_vars)
            if ckpt is not None: auto_load(locals(), 'discr_src', ckpt)
            discr_src.to(device)
            frame = get_frame(discr, gen, dc_vars, device, discr_src)
            if ckpt is not None: auto_load(locals(), 'frame', ckpt)
            return discr, gen, frame, discr_src
        else:
            frame = get_frame(discr, gen, dc_vars, device)
            if ckpt is not None: auto_load(locals(), 'frame', ckpt)
            return discr, gen, frame
    else: return discr, None, None


# Built methods
def get_ce_or_bce_loss(discr, dim_y: int, reduction: str="mean"):
    if dim_y == 1:
       
        celossobj = tc.nn.BCEWithLogitsLoss(reduction=reduction)
        celossfn = lambda x, y: celossobj(x, y.float())   
        
    else:
        celossobj = tc.nn.CrossEntropyLoss(reduction=reduction)
        celossfn = lambda x, y: celossobj(x, y)
        
    relossobj = tc.nn.MSELoss(reduction=reduction)
    relossfn = lambda x, x_c: relossobj(x, x_c)
    
    return celossobj, celossfn, relossobj, relossfn

def add_ce_loss(lossobj, celossfn, relossfn, ag):
    def lossfn(*data_args):
        x, y, y_pre, x_rec = data_args
       
        loss = celossfn(y_pre, y) 
        loss += 50*relossfn(x_rec, x) 
        loss += ag.wgen * lossobj(x,y)
       
        return loss
    return lossfn

def ood_methods(discr, frame, ag, dim_y, cnbb_actv):
    if ag.mode not in MODES_GEN:
        if ag.mode == "discr":
            lossfn = get_ce_or_bce_loss(discr, dim_y, ag.reduction)[1]
        elif ag.mode == "cnbb":
            lossfn = CNBBLoss(discr.s1x, cnbb_actv, discr.forward, dim_y, ag.reg_w, ag.reg_s, ag.lr_w, ag.n_iter_w)
    else: # should be in MODES_GEN
        celossfn = get_ce_or_bce_loss( partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
                if ag.mode in MODES_TWIST and ag.true_sup else discr,
            dim_y, ag.reduction)[1] #交叉熵损失
        relossfn = get_ce_or_bce_loss( partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
                if ag.mode in MODES_TWIST and ag.true_sup else discr,
            dim_y, ag.reduction)[3]
        lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, "defl", wlogpi=ag.wlogpi/ag.wgen) #目标函数最终损失
        lossfn = add_ce_loss(lossobj, celossfn, relossfn, ag)
    return lossfn


def process_continue_run(ag):
    # Process if continue running
    if ag.init_model not in {"rand", "fix"}: # continue running
        ckpt = load_ckpt(ag.init_model, loadmodel=False)
        if ag.mode != ckpt['mode']: raise RuntimeError("mode not match")
        for k in vars(ag):
            if k not in {"testdoms", "n_epk", "gpu"}: # use the new final number of epochs
                setattr(ag, k, ckpt[k])
        ag.testdoms = [ckpt['testdom']] # overwrite the input `testdoms`
    else: ckpt = None
    return ag, ckpt



class ShrinkRatio:
    def __init__(self, w_iter, decay_rate):
        self.w_iter = w_iter
        self.decay_rate = decay_rate

    def __call__(self, n_iter):
        return (1 + self.w_iter * n_iter) ** (-self.decay_rate)
    
