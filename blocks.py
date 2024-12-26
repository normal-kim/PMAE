#from entmax import entmax_bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List

from timm.layers import PatchEmbed, Mlp, DropPath, use_fused_attn

from einops import rearrange, reduce, repeat


# AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
#     trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
#     get_act_layer, get_norm_layer, LayerType


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused #use_fused_attn()
        #self.alpha = nn.Parameter(torch.tensor(2.0))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.save_attn = False

    def forward(self, x: torch.Tensor, attn_mask = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask = attn_mask, 

            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            #attn = entmax_bisect(attn, self.alpha, dim=-1)
            #print(attn)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if (self.save_attn == True):
                self.attn_res = attn 
            x = attn @ v
            

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            fused = True, 
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            fused = fused
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask = None) -> torch.Tensor:
        #x_before_d = x.clone()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask = attn_mask)))
        #print(attn_mask)
        #x_new_d = x.clone()

        #x_before_c = x.clone()
        x = x +self.mlp(self.norm2(x))
        #print('yes')
        #x_new_c = x.clone()
        return x#, x_before_c, x_new_c, x_before_d, x_new_d #x


##############################

class Block_mlp(nn.Module):
    def __init__(
            self,
            d_dim: int,
            c_dim : int,
            block_config: int = 0,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            p_drop : float =0.1, 
    ) -> None:
        super().__init__()
        self.norm_c = norm_layer(c_dim)
        self.norm_d = norm_layer(d_dim)
        
        self.mlp_c = mlp_layer(
            in_features=c_dim,
            hidden_features=int(c_dim * mlp_ratio),
            act_layer=act_layer,
            drop=p_drop,
        )

        self.mlp_d = mlp_layer(
            in_features=d_dim,
            hidden_features=int(d_dim * mlp_ratio),
            act_layer=act_layer,
            drop=p_drop,
        )

        #self.dropout = nn.Dropout(p_drop )

        self.block_config = block_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_config ==0:
            #print(self.block_config)
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x))
            x = rearrange(x, 'n c d -> n d c')
            x = x + self.mlp_c(self.norm_c(x)) 

        elif self.block_config == 1:
            #print(self.block_config)
            x = x + self.mlp_c(self.norm_c(x))
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x)) 
            x = rearrange(x, 'n c d -> n d c')

        elif self.block_config == 2:
            #print(self.block_config)
            x = rearrange(x, 'n d c -> n c d')
            x = x + self.mlp_d(self.norm_d(x))
            x = rearrange(x, 'n c d -> n d c')

        elif self.block_config == 3:
            #print(self.block_config)
            x = x + self.mlp_c(self.norm_c(x))

        else:
            pass

        return x


# class Block_mlp(nn.Module):
#     def __init__(
#             self,
#             d_dim: int,
#             c_dim : int,
#             block_config: int = 0,
#             mlp_ratio: float = 4.,
#             act_layer: nn.Module = nn.GELU,
#             norm_layer: nn.Module = nn.LayerNorm,
#             mlp_layer: nn.Module = Mlp,
#             p_drop : float =0.1, 
#     ) -> None:
#         super().__init__()
#         self.norm_c = norm_layer(c_dim)
#         self.norm_d = norm_layer(d_dim)
        
#         self.mlp_c = mlp_layer(
#             in_features=c_dim,
#             hidden_features=int(c_dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=p_drop,
#         )

#         self.mlp_d = mlp_layer(
#             in_features=d_dim,
#             hidden_features=int(d_dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=p_drop,
#         )

#         #self.dropout = nn.Dropout(p_drop )

#         self.block_config = block_config

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.block_config ==0:
#             #print(self.block_config)
#             x = rearrange(x, 'n d c -> n c d')
#             x_before_d = x.clone()
#             x = x + self.mlp_d(self.norm_d(x))
#             x_new_d = x.clone()
#             x = rearrange(x, 'n c d -> n d c')
#             x_before_c = x.clone()
#             x = x + self.mlp_c(self.norm_c(x)) 
#             x_new_c = x.clone()

#         elif self.block_config == 1:
#             #print(self.block_config)
#             x_before_c = x.clone()
#             x = x + self.mlp_c(self.norm_c(x))
#             x_new_c = x.clone()
#             x = rearrange(x, 'n d c -> n c d')
#             x_before_d = x.clone()
#             x = x + self.mlp_d(self.norm_d(x)) 
#             x_new_d = x.clone()
#             x = rearrange(x, 'n c d -> n d c')

#         elif self.block_config == 2:
#             #print(self.block_config)
#             x = rearrange(x, 'n d c -> n c d')
#             x_before_d = x.clone()
#             x = x + self.mlp_d(self.norm_d(x))
#             x_new_d = x.clone()
#             x = rearrange(x, 'n c d -> n d c')
#             x_before_c = x_new_c = x.clone()

#         elif self.block_config == 3:
#             #print(self.block_config)
#             x = x + self.mlp_c(self.norm_c(x))

#         else:
#             pass



#         return x, x_before_c, x_new_c, x_before_d, x_new_d
