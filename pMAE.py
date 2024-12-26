# current implementation: only support numerical values

from functools import partial
from tkinter import E

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from blocks import Block, Block_mlp
from utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed
eps = 1e-6


class MaskedAutoencoder(nn.Module):
    
    """ Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, rec_len=25, embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_field_loss=False, encode_func='linear', 
        block_mlp = None, old_loss = False
        ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)
        
        self.rec_len = rec_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False)  
        self.num_heads = num_heads
        

        ####################################################################
        # ADDED THIS PART 
        # MIXER or TRF
        self.block_mlp = block_mlp

        if self.block_mlp is None:
            self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)       # Transformer
                    for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block_mlp(d_dim = self.rec_len + 1, c_dim = embed_dim, block_config = self.block_mlp)  # MLP-mixer
                for i in range(depth)])

        ####################################################################

        self.norm = norm_layer(embed_dim)        

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if self.block_mlp is None:
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
        else:
            self.decoder_blocks = nn.ModuleList([
                Block_mlp(d_dim = self.rec_len + 1 , c_dim = decoder_embed_dim, block_config = self.block_mlp)
                        for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)  # decoder to patch
        
        # --------------------------------------------------------------------------

        self.norm_field_loss = norm_field_loss
        self.initialize_weights()

        ####################################################################
        # ADDED THIS PART for PMAE
        self.block_mlp = block_mlp        
        self.old_loss = old_loss
        ####################################################################


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.mask_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def random_masking(self, m, mask_ratio = 0.5):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [n, d], sequence
        """

        N, L = m.shape
        noise = torch.rand(N, L, device=m.device) 
        noise[m < eps] = 1 
        mask = torch.where( noise < mask_ratio, 1 - m.clone(), 1)  # 1: miss
        nask = 1 - mask

        return nask

    def forward_encoder(self, x, m, mask_ratio=0.5):
        '''
        x: [n, d]
        m: [n, d]

        '''
        
        if self.training:
        
            d = x.shape[1]
            
            ######################################################
            # ONLY MASK PARTIALLLY OBSERVED COLUMNS

            full_cols = torch.where(m.mean(axis = 0).detach().cpu() == 1)[0].tolist()
            partial_cols = [i for i in range(d) if i not in full_cols] # for this part calculate based on mask (can be easily checked with given m)
            
            m_full, m_partial = m[:, full_cols], m[:, partial_cols]
            m_partial = m[:, partial_cols]
            N = m_full.shape[0]

            ######################################################
            
            nask = self.random_masking(m_partial, mask_ratio = mask_ratio[partial_cols])
            nask = torch.cat([m_full, nask],1).gather(1, torch.argsort(torch.tensor(full_cols + partial_cols)).repeat(N,1).to(x.device))
            
            x_gathered = nask * x
            mask = 1 - nask
            

        else:
            x_gathered = x
            nask = m.float()
            mask = None
            ids_restore = None

        # embed patches
        x = self.mask_embed(x_gathered.unsqueeze(1))   ## tokenizer  --> could be replaced later

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    # to identify which columns attend

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # extra dim cuz of CLS token

        # apply Transformer blocks


        for blk in self.blocks:
            x = blk(x) if self.block_mlp is None else blk(x)

        x = self.norm(x)

        return x, mask, nask


    def forward_decoder(self, x, nask = None):
        
        # embed tokens
        x = self.decoder_embed(x)  ## shared MLP

        x = torch.cat([x[:, :1, :], (x[:, 1:, :] * nask.unsqueeze(2)) + (1 - nask).unsqueeze(2) * self.mask_token], dim = 1)
        x_after_gather = x.clone()

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = torch.tanh(self.decoder_pred(x))/2 + 0.5  # 0 ~ 1

        # remove cls token
        x = x[:, 1:, :]
    
        return x, x_after_gather


    def forward_loss(self, data, pred, mask, nask, observed_mask, loss_flag = 'org'):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        if loss_flag == 'new':
            #print('new loss')
            target_miss = observed_mask.mean(axis = 0) != 1
            target_niss = (target_miss == False)        

            n_observed_miss = observed_mask[:, target_miss].sum()
            n_observed_niss = observed_mask[:, target_niss].sum()

            target = data.squeeze(dim=1)

            loss = ((pred.squeeze(dim=2) - target) ** 2)*observed_mask
            loss1 = 0 if n_observed_miss == 0 else (loss[:, target_miss]).sum() / n_observed_miss
            loss2 = 0 if n_observed_niss == 0 else (loss[:, target_niss]).sum() / n_observed_niss

            loss = loss1 + loss2

        else:
            #print('old loss')
            target = data.squeeze(dim=1)
            loss = (pred.squeeze(dim=2) - target) ** 2
            loss = (loss * mask).sum() / mask.sum()  + (loss * nask).sum() / nask.sum()
            
        return loss


    def forward(self, data, miss_idx, mask_ratio=0.5,):
        latent, mask, nask = self.forward_encoder(data, miss_idx, mask_ratio)
        pred, x_after_gather = self.forward_decoder(latent, nask)   # 0 ~ 1

        if self.training:
            loss_flag = 'new' if self.old_loss == False else 'org'

            loss = self.forward_loss(data, pred, mask, nask, observed_mask = miss_idx, loss_flag = loss_flag)

        else:
            loss = None
        
        latent = x_after_gather.clone()

        return loss, pred, mask, nask, latent




    
