import torch
import yaml
import os
import random
import numpy as np


def get_config(args):

    with open(args.config_path, 'rb') as yaml_config:
        cfg = yaml.safe_load(yaml_config)

    cfg['epoch_sel'] = args.epoch if hasattr(args, 'epoch') else None
    cfg['dataset']['scenario'] = args.scenario
    cfg['pointcloud']['downsampling_layer_num'] = len(cfg['pointcloud']['grid_size'])
    cfg['pointcloud']['minimum_point_num'] = cfg['pointcloud']['knn'] # set a heuristic minimum point count to keep KNN search valid.

    # create output dir only if it doesn't exist
    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])
    cfg['output_dir'] = os.path.join(cfg['output_dir'], args.scenario)
    # create scenario-specific output dir only if it doesn't exist
    if not os.path.exists(cfg['output_dir']):
        os.makedirs(cfg['output_dir'])
    
    # for additional data loader used when hard_example_mining is enabled
    cfg['train_hem'] = {}
    cfg['train_hem']['batch_sz'] = cfg['train']['batch_sz']
    cfg['train_hem']['view_num'] = cfg['train']['view_num']
    cfg['train_hem']['num_workers'] = cfg['train']['num_workers']
    cfg['train_hem']['seq_len'] = cfg['train']['seq_len']

    cfg['exp_seed'] = args.seed
    if cfg['exp_seed'] is not None:
        cfg['exp_name_epoch'] = cfg['exp_name'] + f"_seed_{args.seed}" + f"_epoch{cfg['epoch_sel']}" + f"_{args.scenario}" # used only during testing
        cfg['exp_name'] = cfg['exp_name'] + f"_seed_{args.seed}" + f"_{args.scenario}"
    else:
        cfg['exp_name_epoch'] = cfg['exp_name'] + f"_epoch{cfg['epoch_sel']}" + f"_{args.scenario}" # used only during testing
        cfg['exp_name'] = cfg['exp_name'] + f"_{args.scenario}"

    return cfg


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_one_hot_by_majority_vote_numpy_ver(scores):

    assert scores.ndim == 3
    N, C, M = scores.shape

    # Valid cameras: non-zero score vector
    cam_valid = (np.abs(scores).sum(axis=2) > 0)  # (N, C) bool

    # winner object index per camera per point: (N, C)
    cam_winners = scores.argmax(axis=2)

    # One-hot per camera, then mask invalid cameras
    one_hot_cam = np.eye(M, dtype=np.int64)[cam_winners]  # (N, C, M)
    one_hot_cam = one_hot_cam * cam_valid[..., None]  # (N, C, M)

    # Count votes
    counts = one_hot_cam.sum(axis=1)  # (N, M)

    # Sum scores only from valid cameras
    totals = (scores * cam_valid[..., None]).sum(axis=1)  # (N, M)

    # find the max vote count for each point: (N,1)
    max_counts = counts.max(axis=1, keepdims=True)

    # mask for tie candidates
    is_tie = (counts == max_counts)

    # masked totals: only keep totals for tied objects, others = -inf
    masked_totals = np.where(is_tie, totals, -np.inf)

    # final selected object per point
    winners = masked_totals.argmax(axis=1)  # (N,)

    # one-hot
    onehot = np.eye(M)[winners]  # (N, M)
    
    return onehot.astype(scores.dtype)
