# stdlib
from typing import Any, List, Tuple, Union
import time, sys
from sklearn.preprocessing import MinMaxScaler

# third party
import numpy as np
import math, sys, argparse
import pandas as pd
import torch
from torch import nn
from functools import partial

from utils import NativeScaler, MAEDataset, adjust_learning_rate, get_dataset

import pMAE
from torch.utils.data import DataLoader, RandomSampler
import timm.optim.optim_factory as optim_factory
from utils import get_args_parser


eps = 1e-8


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


########################################################################################

class ProportionalMasker:

    def __init__(self, args):
        self.args = args
        #args = get_args_parser().parse_args([])
        
        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.norm_field_loss = args.norm_field_loss
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = args.warmup_epochs
        self.model = None
        self.norm_parameters = None

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = args.max_epochs
        self.mask_ratio = args.mask_ratio
        self.encode_func = args.encode_func


        # NEW PARTS
        self.new_imp = True
        self.block_mlp = None
        self.check_epochs =600 #300
        
        self.a = 0.05
        self.b = 0.5
        
        self.old_loss = False
        self.device = 'cpu'



    def fit(self, X_raw: pd.DataFrame):
        device = self.device
        print(self.mask_ratio)

        X = torch.tensor(X_raw.values, dtype=torch.float32).clone()

        # Parameters
        no = len(X)
        dim = len(X[0, :])

        X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        
        target_miss = M.float().mean(axis = 0) != 1
        miss_cols = torch.where(target_miss)[0].to(dtype = torch.int64).cpu().numpy()
        
        M = M.float().to(device)

        X = torch.nan_to_num(X)
        X = X.to(device)
        
        print('missing col index:', miss_cols)

        self.model = pMAE.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func, 
            block_mlp = self.block_mlp, 
            old_loss = self.old_loss
        ) 

        self.model.to(device)

        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / self.batch_size

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        self.model.train()

        
        print('p_obs mean: ', M.float().mean(axis = 0))
        
        obs_mean_vec = M.float().mean(axis = 0).float()
            
        start = time.time()

        for epoch in range(self.max_epochs):
            if epoch == 1:
                print(f'1 epoch took: {time.time() - start} sec')

            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            for iter, (samples, masks) in enumerate(dataloader):

                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    tmp_lr = adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)

                samples = samples if self.new_imp else samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    
                    ##########################################
                    # NEW PARTS for PMAE
                    obs_mean_vec = masks.float().mean(axis = 0).float()
                    
                    # a = 0.5, b = 0.05
                    self.mask_ratio = 1.0 * (obs_mean_vec > 0.9999) + 1.0* (obs_mean_vec <=  0.9999) * (self.a * torch.log((obs_mean_vec+1e-20)/(1-(obs_mean_vec)+1e-20)) + self.b)        
                    loss, pred, mask, nask, latent = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    
                    ##########################################

                    loss_value = loss.item()
                    

                if not math.isfinite(loss_value):
                    print("Loss is {},".format(loss_value))
                    optimizer.zero_grad()
                    iter -= 1

                else:
                    total_loss += loss_value
                    
                    loss /= self.accum_iter
                    norm_ = loss_scaler(loss, optimizer, parameters=self.model.parameters(),
                                update_grad=(iter + 1) % self.accum_iter == 0)

                    if (iter + 1) % self.accum_iter == 0:
                        optimizer.zero_grad()


            total_loss = (total_loss / (iter + 1)) ** 0.5

            if epoch % 20 == 0:
                print(f'Loss at epoch {epoch}: {total_loss}')


        # Fitted 
        with torch.no_grad():
            self.model.eval()
            X_imputed = self.transform(torch.tensor(X_raw.values, dtype = torch.float32).clone())
        
        return X_imputed

    def transform(self, X_raw: torch.Tensor):
        #device = self.device

        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape
        X = X.cpu()

        # MinMaxScaler normalization
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(self.device)
        M = M.to(self.device)

        self.model.eval()
        # Imputed data
        with torch.no_grad():
            if self.new_imp:
                loss, imputed_data, mask, nask, latent = self.model(X, M)
                imputed_data = imputed_data.squeeze(dim = 2)

            else:
                latent_lst = []
                for i in range(no):
                    sample = torch.reshape(X[i], (1, 1, -1))
                    mask = torch.reshape(M[i], (1, -1))
                    _, pred, _, _, latent, _ = self.model(sample, mask)

                    pred = pred.squeeze(dim=2)
                    if i == 0:
                        imputed_data = pred
                    else:
                        imputed_data = torch.cat((imputed_data, pred), 0)
        
        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()

        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.
        Args:
            X: np.ndarray
                A dataset with missing values.
        Returns:
            Xhat: The imputed dataset.
        """
        X = torch.tensor(X.values, dtype=torch.float32)
        return self.fit(X).transform(X).detach().cpu().numpy()