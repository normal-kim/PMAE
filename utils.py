# current implementation: only support numerical values
import numpy as np
import torch, os
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import argparse

######################

from configs import *


import math
import numpy as np
import torch

import math
import numpy as np


def split_data(d_numerical, X_init, mask, miss_cols, device):

    ## TEST DATA
    # if X_cat is not None:
    #     if X_num is not None:     
    #         X_init = torch.cat([X_num, X_cat], axis = 1)
    #     else:
    #         X_init = X_cat
    # else:
    #     X_init = X_num
        
            
    X_incomp = X_init.clone()
    X_incomp[mask.bool()] = np.nan
    #X_incomp_num, X_incomp_cat = X_incomp[:, :d_numerical], X_incomp[:, d_numerical:]
    obs_mask = 1 - 1.0*X_incomp.isnan()
    
    #test_data = X_incomp_num, X_incomp_cat, X_init.to(device), obs_mask.to(device), ps.to(device)
    ## TR & VAL DATA
    # split procedures
    tr_idx = int(0.8 * len(X_init))  # we can extend this using cross validation 
    
    X_init_tr, X_init_val = X_init[:tr_idx, :], X_init[tr_idx:, :]
    #X_incomp_tr, # The above code is not doing anything as it is just a variable name `X_incomp_val`
    # followed by `
    X_incomp_tr, X_incomp_val = X_incomp[:tr_idx, :], X_incomp[tr_idx:, :]
    #ps_tr, ps_val = ps[:tr_idx, :], ps[tr_idx:, :]
    
    # relocate for all tr & vals
    relocate_idx =  (1.0*X_incomp_val[:, miss_cols].isnan()).mean(axis = 1) == 1  # 전부 nan인 애들은 val 에서 제외
    
    X_init_tr = torch.cat([X_init_tr, X_init_val[relocate_idx, :]], axis = 0)
    X_incomp_tr = torch.cat([X_incomp_tr, X_incomp_val[relocate_idx, :]], axis = 0)
    #X_incomp_num_tr, X_incomp_cat_tr = X_incomp_tr[:, :d_numerical], X_incomp_tr[:, d_numerical:]
    #ps_tr = torch.cat([ps_tr, ps_val[relocate_idx, :]], axis = 0)
    
    X_init_val = X_init_val[relocate_idx == False, :]
    X_incomp_val = X_incomp_val[relocate_idx == False, :]
    #ps_val = ps_val[relocate_idx == False, :]
    
    ps_val_, X_incomp_val_, X_init_val_, total_mask_, target_mask_ = [], [], [], [], []
    for i in range(5):    
        obs_mask_val = 1.0 - 1.0*X_incomp_val.isnan()
        
        set_all_seeds(i+42)
        
        X_incomp_val_tmp = X_incomp_val.clone()
        mask_idx = (1.0*obs_mask_val[:, miss_cols]  + 1.0*(torch.rand_like(obs_mask_val[:, miss_cols] ) < 0.5) == 2)
        X_incomp_val_tmp[:, miss_cols] = X_incomp_val_tmp[:, miss_cols].where( mask_idx, np.nan)
        
        total_mask = 1 - 1.0*X_incomp_val_tmp.isnan()
        target_mask = obs_mask_val - total_mask
        relocate_idx =  target_mask.mean(axis = 1) != 0
    
        #ps_val_.append(ps_val[relocate_idx, :])
        X_incomp_val_.append(X_incomp_val_tmp[relocate_idx, :])
        X_init_val_.append(X_init_val[relocate_idx, :])
        total_mask_.append(total_mask[relocate_idx, :])
        target_mask_.append(target_mask[relocate_idx, :])
    
    #ps_val_ = torch.cat(ps_val_)
    X_incomp_val_ = torch.cat(X_incomp_val_)
    X_init_val_ = torch.cat(X_init_val_)
    total_mask_ = torch.cat(total_mask_)
    target_mask_ = torch.cat(target_mask_)
    
    #tr_data = X_incomp_num_tr, X_incomp_cat_tr #, ps_tr
    #val_data = X_incomp_val_[:, :d_numerical].nan_to_num().to(device), X_incomp_val_[:, d_numerical:].nan_to_num().to(dtype = torch.int64).to(device), X_init_val_.to(device), total_mask_.to(device), target_mask_.to(device), ps_val_.to(device)
    val_data = X_incomp_val_[:, :d_numerical].to(device), X_incomp_val_[:, d_numerical:].to(dtype = torch.int64).to(device), X_init_val_.to(device), total_mask_.to(device), target_mask_.to(device)#, ps_val_.to(device)
    #test_data = X_incomp_num, X_incomp_cat, X_true_t, total_mask_t, M_t, ps_t
    #X_incomp_num_t, X_incomp_cat_t, X_true_t, obs_mask, ps_t = test_data    

    return X_incomp_val_, X_init_val_, total_mask_.to(device) #val_data#tr_data, val_data, test_data
######################


#######################
def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = (I - unit / n).to(K.device)

    return torch.matmul(torch.matmul(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = X@X.T
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def get_dataset_remasker(dataset_):
    datasets = ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'california', 'diabetes']
    assert dataset_ in datasets
    kwargs = get_configs(dataset_)
    args = get_args_parser().parse_args([])
    args.__dict__.update(kwargs)
    args.dataset = dataset_
    args.path = ''
    X, y = get_dataset(args.dataset, args.path)
    return X, y



class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, rec_len=25, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    #print(tmp_lr)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            
            if clip_grad is not None:
                print('yes')
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)

            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):        
         self.X = X
         self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]
    
    
class MAEDataset2(Dataset):

    def __init__(self, X, X_init_tr, M, ps):        
         self.X = X
         self.X_init_tr = X_init_tr
         self.M = M
         self.ps = ps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.X_init_tr[idx], self.M[idx], self.ps[idx]

class MAEDataset3(Dataset):

    def __init__(self, X_num, X_cat, M, ps):        
         self.X_num = X_num
         self.X_cat = X_cat
         #self.X_init_tr = X_init_tr
         self.M = M
         self.ps = ps

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx: int):
        return self.X_num[idx], self.X_cat[idx], self.M[idx], self.ps[idx]


def get_dataset(dataset : str, path : str):

    if dataset in ['climate', 'compression', 'wine', 'yacht', 'spam', 'letter', 'credit', 'raisin', 'bike', 'obesity', 'airfoil', 'blood', 'yeast', 'health', 'review', 'travel']:
        df = pd.read_csv(os.path.join(path, 'data', dataset + '.csv'), index_col = 0)
        last_col = df.columns[-1]
        y = df[last_col]
        X = df.drop(columns=[last_col])
    elif dataset == 'california':
        from sklearn.datasets import fetch_california_housing
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    elif dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(as_frame=True, return_X_y=True)
    elif dataset == 'iris':
        # only for testing
        from sklearn.datasets import load_iris
        X, y = load_iris(as_frame=True, return_X_y=True)
    

    return X, y


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset', default='california', type=str)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--embed_dim', default=32, type=int, help='embedding dimensions')
    parser.add_argument('--depth', default=6, type=int, help='encoder depth')
    parser.add_argument('--decoder_depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--num_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--mlp_ratio', default=4., type=float, help='mlp ratio')
    parser.add_argument('--encode_func', default='linear', type=str, help='encoding function')

    parser.add_argument('--norm_field_loss', default=False,
                        help='Use (per-patch) normalized field as targets for computing loss')
    parser.set_defaults(norm_field_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    ###### change this path
    parser.add_argument('--path', default='/data/', type=str, help='dataset path')
    parser.add_argument('--exp_name', default='test', type=str, help='experiment name')

    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)

    parser.add_argument('--overwrite', default=True, help='whether to overwrite default config')
    parser.add_argument('--pin_mem', action='store_false')

    # distributed training parameters
    return parser

if __name__ == '__main__':
    
    X = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    X = X.unsqueeze(1)
    mask_embed = ActiveEmbed(4, 6)
    print(mask_embed(X).shape)
