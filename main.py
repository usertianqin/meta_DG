import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from src.core.trans_gnn import trainer, joint_train
from parser import get_parser
from src.model2.utils.utils import boolstr
from src.model2.func import process_continue_run


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config,ag):
    print_config(config)
    print(ag)
    joint_train(config,ag) 


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.safe_load(setting)
    return config


def get_args():
    #parser = argparse.ArgumentParser()
    parser = get_parser()
    parser.add_argument('--config', required=True,
                        type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true',
                        help='flag: multi run')

   
    parser.add_argument("--shuffle", type = boolstr, default = True)

    parser.add_argument("--mlpstrufile", type = str, default = "/src/model2/arch/mlpstru.json")
    parser.add_argument("--actv", type = str, default = "Sigmoid")
    parser.add_argument("--after_actv", type = boolstr, default = True)

    parser.set_defaults(discrstru = "irm", genstru = None,
            mu_s = .5, mu_v = .5,
            pstd_x = 3e-2, qstd_s = 3e-2, qstd_v = 3e-2,
            optim = "RMSprop", lr = 1e-3, wl2 = 1e-5,
            momentum = 0., nesterov = False, lr_expo = .5, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
            wda = 1.
        )
    args = parser.parse_args()
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************\n")


def grid(kwargs):
    '''
    Builds a mesh grid with given keyword arguments for this Config class.
    '''

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        '''
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        '''
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()),
                   dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i]
         for i, k in enumerate(sin)}
    ) for vv in grd]


if __name__ == '__main__':
    cfg = vars(get_args())
    config = get_config(cfg['config']) #加载yaml文件的内容
    ag = get_args()
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    main(config,ag)
