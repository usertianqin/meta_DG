import argparse
from src.model2.utils.utils import boolstr

MODES_OOD_GEN = { "svgm", "svgm-ind"}


MODES_OOD = MODES_OOD_GEN#
MODES_GEN = MODES_OOD_GEN 

MODES_TWIST = {"svgm-ind", "svae-da", "svgm-da"}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type = str, choices = MODES_OOD | MODES_DA)

    # Data
    parser.add_argument("--support_split", type = float, default = .1)
    parser.add_argument("--query_split", type = float, default = .9)

    # Model
    parser.add_argument("--init_model", type = str, default = "rand") # or a model file name to continue running
    parser.add_argument("--discrstru", type = str)
    parser.add_argument("--genstru", type = str)
    

    # Process
    parser.add_argument("--eval_interval", type = int, default = 5)
    parser.add_argument("--avglast", type = int, default = 4)
    parser.add_argument("--test_num", type = int, default = 1)


    # Optimization

    parser.add_argument("--update_lr", type = float, help='task-level inner update learning rate',default=2)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    parser.add_argument("--wl2", type = float)
    parser.add_argument("--reduction", type = str, default = "mean")
    parser.add_argument("--momentum", type = float, default = 0.) # only when "lr" is "SGD"
    parser.add_argument("--nesterov", type = boolstr, default = False) # only when "lr" is "SGD"
    parser.add_argument("--lr_expo", type = float, default = .75) # only when "lr" is "SGD"
    parser.add_argument("--lr_wdatum", type = float, default = 6.25e-6) # only when "lr" is "SGD"

    # For generative models only
    parser.add_argument("--mu_s", type = float, default = 0.)
    parser.add_argument("--sig_s", type = float, default = 1.)
    parser.add_argument("--mu_v", type = float, default = 0.) # for svgm only
    parser.add_argument("--sig_v", type = float, default = 1.) # for svgm only
    parser.add_argument("--corr_sv", type = float, default = .7) # for svgm only
    parser.add_argument("--pstd_x", type = float, default = 3e-2)
    parser.add_argument("--qstd_s", type = float, default = 3e-2)
    parser.add_argument("--qstd_v", type = float, default = 3e-2) # for svgm only
    parser.add_argument("--wgen", type = float, default = 1.)
    parser.add_argument("--wsup", type = float, default = 1.)
    parser.add_argument("--wsup_expo", type = float, default = 0.) # only when "wsup" is not 0
    parser.add_argument("--wsup_wdatum", type = float, default = 6.25e-6) # only when "wsup" and "wsup_expo" are not 0
    parser.add_argument("--wlogpi", type = float, default = None)
    parser.add_argument("--n_mc_q", type = int, default = 0)
    parser.add_argument("--true_sup", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--true_sup_val", type = boolstr, default = True, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--mvn_prior", type = boolstr, default = False)
    


    return parser
