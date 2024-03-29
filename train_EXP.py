"""
script to train on EXP classification dataset
"""
import argparse
import os
import shutil
import time
from json import dumps

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.seed import seed_everything

import train_utils
from data_utils import extract_multi_hop_neighbors, resistance_distance, post_transform
from datasets.PlanarSATPairsDataset import PlanarSATPairsDataset
from layers.input_encoder import EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.GraphClassification import GraphClassification
from models.model_utils import make_GNN


# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def get_model(args):
    layer = make_gnn_layer(args)
    init_emb = EmbeddingEncoder(args.input_size, args.hidden_size)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob,
        with_kn_configuration=args.with_kn_configuration)

    model = GraphClassification(embedding_model=gnn,
                                pooling_method=args.pooling_method,
                                output_size=args.output_size)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

    return model


def node_feature_transform(data):
    data.x = data.x[:, 0].to(torch.long)
    return data


def train(loader, model, optimizer, device, parallel=False):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        out = model(data)
        predict = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(predict, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val(loader, model, device, parallel=False):
    model.eval()
    loss_all = 0
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        predict = model(data)
        predict = F.log_softmax(predict, dim=-1)
        loss = F.nll_loss(predict, y, reduction='sum').item()
        loss_all += loss

    return loss_all / len(loader.dataset)


@torch.no_grad()
def test(loader, model, device, parallel=False):
    model.eval()
    correct = 0
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        nb_trials = 1  # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(data).max(1)[1]
            successful_trials += pred.eq(y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--seed', type=int, default=224, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_name', type=str, default="EXP", choices=("EXP", "CEXP"), help='Name of dataset')
    parser.add_argument('--drop_prob', type=float, default=0.,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument("--parallel", action="store_true",
                        help="If true, use DataParallel for multi-gpu training")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load as a model checkpoint.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_wd', type=float, default=3e-7, help='L2 weight decay.')
    parser.add_argument("--kernel", type=str, default="spd", choices=("gd", "spd"),
                        help="The kernel used for K-hop neighbors extraction")
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs.')
    parser.add_argument("--hidden_size", type=int, default=48, help="Hidden size of the model")
    parser.add_argument("--model_name", type=str, default="KPGIN",
                        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"), help="Base GNN model")
    parser.add_argument("--K", type=int, default=3, help="Number of hop to consider")
    parser.add_argument("--max_pe_num", type=int, default=1,
                        help="Maximum number of path encoding. Must be equal to or greater than 1")
    parser.add_argument("--max_edge_type", type=int, default=1,
                        help="Maximum number of type of edge to consider in peripheral edge information")
    parser.add_argument("--max_edge_count", type=int, default=1000,
                        help="Maximum count per edge type in peripheral edge information")
    parser.add_argument("--max_hop_num", type=int, default=5,
                        help="Maximum number of hop to consider in peripheral configuration information")
    parser.add_argument("--max_distance_count", type=int, default=1000,
                        help="Maximum count per hop in peripheral configuration information")
    parser.add_argument('--wo_peripheral_edge', action='store_true',
                        help='If true, remove peripheral edge information from model')
    parser.add_argument('--wo_peripheral_configuration', action='store_true',
                        help='If true, remove peripheral node configuration from model')
    parser.add_argument("--wo_path_encoding", action="store_true", help="If true, remove path encoding from model")
    parser.add_argument("--wo_edge_feature", action="store_true", help="If true, remove edge feature from model")
    parser.add_argument("--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1")
    parser.add_argument("--num_layer", type=int, default=3, help="Number of layer for feature encoder")
    parser.add_argument("--JK", type=str, default="last", choices=("sum", "max", "mean", "attention", "last"),
                        help="Jumping knowledge method")
    parser.add_argument("--residual", action="store_true", help="If true, use residual connection between each layer")
    parser.add_argument("--use_rd", action="store_true", help="If true, add resistance distance feature to model")
    parser.add_argument("--virtual_node", action="store_true",
                        help="If true, add virtual node information in each layer")
    parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN")
    parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon is trainable")
    parser.add_argument("--combine", type=str, default="geometric", choices=("attention", "geometric"),
                        help="Combine method in k-hop aggregation")
    parser.add_argument("--pooling_method", type=str, default="sum", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph classification")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
                        help="Normalization method in model")
    parser.add_argument('--aggr', type=str, default="add",
                        help='Aggregation method in GNN layer, only works in GraphSAGE')
    parser.add_argument('--split', type=int, default=10, help='Number of fold in cross validation')
    parser.add_argument('--reprocess', action="store_true", help='If true, reprocess the dataset')
    parser.add_argument('--with_kn_configuration', action='store_true',
                        help='If true, remove add nk node configuration to model')
    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num

    args.name = args.model_name + "_" + args.kernel + "_" + str(args.K) + "_" + str(args.wo_peripheral_edge) + \
                "_" + str(args.wo_peripheral_configuration) + "_" + str(args.wo_path_encoding) + "_" + str(
        args.wo_edge_feature)
    # Set up logging and devices
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, type=args.dataset_name)
    log = train_utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = train_utils.get_available_devices()

    if len(args.gpu_ids) > 1 and args.parallel:
        log.info(f'Using multi-gpu training')
        args.parallel = True
        loader = DataListLoader
        args.batch_size *= max(1, len(args.gpu_ids))
    else:
        log.info(f'Using single-gpu training')
        args.parallel = False
        loader = DataLoader

    # Set random seed
    seed = train_utils.get_seed(args.seed)
    log.info(f'Using random seed {seed}...')
    seed_everything(seed)
    def multihop_transform(g):
        return extract_multi_hop_neighbors(g, args.K, args.max_pe_num, args.max_hop_num, args.max_edge_type,
                                           args.max_edge_count,
                                           args.max_distance_count, args.kernel)

    if args.use_rd:
        rd_feature = resistance_distance
    else:
        def rd_feature(g):
            return g
    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)

    path = "data/" + args.dataset_name
    if os.path.exists(path + '/processed') and args.reprocess:
        shutil.rmtree(path + '/processed')

    dataset = PlanarSATPairsDataset(root=path,
                                    pre_transform=T.Compose([node_feature_transform, multihop_transform, rd_feature]),
                                    transform=transform)

    # dataset = PlanarSATPairsDataset(root=path)

    # additional parameter for EXP dataset and training
    args.input_size = 2
    args.output_size = dataset.num_classes
    args.MODULO = 4
    args.MOD_THRESH = 1

    # output argument to log file
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    tr_accuracies = np.zeros((args.num_epochs, args.split))
    tst_accuracies = np.zeros((args.num_epochs, args.split))
    tst_exp_accuracies = np.zeros((args.num_epochs, args.split))
    tst_lrn_accuracies = np.zeros((args.num_epochs, args.split))
    acc = []
    tr_acc = []

    for i in range(args.split):
        log.info(f"---------------Training on fold {i + 1}------------------------")
        model = get_model(args)
        model.to(device)
        pytorch_total_params = train_utils.count_parameters(model)
        log.info(f'The total parameters of model :{[pytorch_total_params]}')

        optimizer = Adam(model.parameters(), lr=args.lr)

        # K-fold cross validation split
        n = len(dataset) // args.split
        test_mask = torch.zeros(len(dataset))
        test_exp_mask = torch.zeros(len(dataset))
        test_lrn_mask = torch.zeros(len(dataset))

        test_mask[i * n:(i + 1) * n] = 1  # Now set the masks
        learning_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO <= args.MOD_THRESH]
        test_lrn_mask[learning_indices] = 1
        exp_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO > args.MOD_THRESH]
        test_exp_mask[exp_indices] = 1

        # Now load the datasets
        test_dataset = dataset[test_mask.bool()]
        test_exp_dataset = dataset[test_exp_mask.bool()]
        test_lrn_dataset = dataset[test_lrn_mask.bool()]
        train_dataset = dataset[(1 - test_mask).bool()]

        n = len(train_dataset) // args.split
        val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
        val_mask[i * n:(i + 1) * n] = 1
        val_dataset = train_dataset[val_mask]
        train_dataset = train_dataset[~val_mask]

        val_loader = loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_exp_loader = loader(test_exp_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_lrn_loader = loader(test_lrn_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        train_loader = loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        best_val_loss, best_test_acc, best_train_acc = 100, 0, 0
        start_outer = time.time()
        for epoch in range(args.num_epochs):
            start = time.time()
            train_loss = train(train_loader, model, optimizer, device, parallel=args.parallel)
            lr = optimizer.param_groups[0]['lr']
            val_loss = val(val_loader, model, device, parallel=args.parallel)
            train_acc = test(train_loader, model, device, parallel=args.parallel)
            test_acc = test(test_loader, model, device, parallel=args.parallel)
            if best_val_loss >= val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_test_acc = test_acc

            test_exp_acc = test(test_exp_loader, model, device, parallel=args.parallel)
            test_lrn_acc = test(test_lrn_loader, model, device, parallel=args.parallel)
            tr_accuracies[epoch, i] = train_acc
            tst_accuracies[epoch, i] = test_acc
            tst_exp_accuracies[epoch, i] = test_exp_acc
            tst_lrn_accuracies[epoch, i] = test_lrn_acc

            time_per_epoch = time.time() - start

            log.info('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                     'Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}, '
                     'Seconds: {:.4f}'.format(
                epoch + 1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc, time_per_epoch))

            torch.cuda.empty_cache()  # empty test part memory cost

        acc.append(best_test_acc)
        tr_acc.append(best_train_acc)
        time_average_epoch = time.time() - start_outer
        log.info(
            f'Fold {i + 1}, best train: {best_train_acc}, best test: {best_test_acc}, Seconds/epoch: {time_average_epoch / (epoch + 1)}')
    acc = torch.tensor(acc)
    tr_acc = torch.tensor(tr_acc)
    log.info("-------------------Print final result-------------------------")
    log.info(f"Train result: Mean: {tr_acc.mean().item()}, Std :{tr_acc.std().item()}")
    log.info(f"Test result: Mean: {acc.mean().item()}, Std :{acc.std().item()}")


if __name__ == "__main__":
    main()
