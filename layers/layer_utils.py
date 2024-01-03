"""
Utils for defining model layers
"""
from layers.KPGIN import *


def make_gnn_layer(args):
    """function to construct gnn layer
    Args:
        args (argparser): arguments list
    """
    model_name = args.model_name
    if model_name == "KPGIN":
        gnn_layer = KPGINConv(args.hidden_size, args.hidden_size, args.K, args.eps, args.train_eps, args.num_hop1_edge,
                              args.max_pe_num, args.combine)
    else:
        raise ValueError("Not supported GNN type")

    return gnn_layer
