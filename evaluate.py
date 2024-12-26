import torch
from scipy.stats import wasserstein_distance
#from hyperimpute.plugins.utils.metrics import RMSE

import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def scale_data(X: pd.DataFrame) -> pd.DataFrame:
    preproc = MinMaxScaler()
    cols = X.columns
    return pd.DataFrame(preproc.fit_transform(X), columns=cols), preproc#, cols
    #preproc


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def RMSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        RMSE : np.ndarray

    """
    #X = np.asarray(X)
    #X_true = np.asarray(X_true)
    #mask = np.asarray(mask)

    
    mask_ = mask.astype(bool)
    return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())



def evaluate_perf(d_numerical, X_init, M, out):

    # Numerical & categorical
    x_n = torch.tensor(out.iloc[:, :d_numerical].values)
    x_c = torch.tensor(out.iloc[:, d_numerical:].round().values)  # rounding for categorical

    X_init = X_init.to('cpu')

    # R2
    sse = (((1-M[:, :d_numerical])*(x_n - X_init[:, :d_numerical])).pow(2).sum(axis = 0)).detach().cpu()
    mean_gt = ((1-M[:, :d_numerical])*(X_init[:, :d_numerical])).sum(axis = 0) / (1-M[:, :d_numerical]).sum(axis = 0)
    sst = (((1-M[:, :d_numerical])*(X_init[:, :d_numerical] - mean_gt)).pow(2).sum(axis = 0)).detach().cpu()

    rmse_mean =float((sse / (1-M[:, :d_numerical].detach().cpu()).sum(axis = 0)).sqrt().nanmean())

    target_cols = (M.float().mean(axis = 0) != 1).cpu()
    
    r2 = 1 - sse / sst
    r2 = (torch.where(r2 <0, 0, r2)).nan_to_num()
    X_cat = X_init[:, d_numerical:]
    acc_ = (((1 - M[:, d_numerical:])*(x_c == X_cat)).sum(axis = 0) / (1 - M[:, d_numerical:]).sum(axis = 0))[target_cols[d_numerical:]]
    r2_ = r2[target_cols[:d_numerical]]

    total_perf = torch.cat([r2_, acc_]).mean()

    return total_perf.numpy(), r2_.mean().numpy(), acc_.mean().numpy()


def ws_score(X_fitted, X_true, miss_mask, miss_cols):
    res = 0
    for col in miss_cols:
        res += wasserstein_distance(
            np.asarray(X_fitted)[miss_mask[:, col], col], np.asarray(X_true)[miss_mask[:, col], col])
    return res / len(miss_cols)


def evaluator(X_init, out, mask, d, d_numerical, miss_cols, cat_exists):
    '''
    X_init: Ground Truth
    out: Imputed output
    mask: miss (1), obs (0)
    d: entire columns
    d_numerical: numerical columns
    miss_cols: missing columns
    cat_exists: whether there are any categorical columns
    '''

    obs_mask_ = 1 - 1.0*mask  # 1: obs
    
    X_init_scaled, preproc = scale_data(pd.DataFrame(X_init))
    out_scaled = preproc.fit_transform(pd.DataFrame(out))
    
    # (1) imputation accuracy
    total_perf, r2, acc = evaluate_perf(d_numerical, X_init, obs_mask_, pd.DataFrame(out)) 
    
    # (2) WS distance (after min-max)
    mask = np.asarray(mask)
    try:
        ws = ws_score(out_scaled, X_init_scaled, mask, miss_cols)
        
    except:
        print('error')
        ws = None
    
    # (3) RMSE score
    rmse_score = RMSE(np.asarray(out_scaled), np.asarray(X_init_scaled), (1.0*mask))
    
    rmse_score_cat = RMSE(np.asarray(out_scaled)[:, d_numerical:], 
                              np.asarray(X_init_scaled)[:, d_numerical:], 
                              (1.0*mask[:, d_numerical:]))  if cat_exists else np.nan
                          
    rmse_score_num = RMSE(np.asarray(out_scaled)[:, :d_numerical], 
                              np.asarray(X_init_scaled)[:, :d_numerical], 
                              (1.0*mask[:, :d_numerical])) if d_numerical != 0 else np.nan

    score_dict = {'imp_acc': float(total_perf), 
                  'R2(numerical)': float(r2), 
                  'Acc(categorical)': float(acc), 
                  'WD': float(ws), 
                  'RMSE': float(rmse_score), 
                  'RMSE(num)': float(rmse_score_num), 
                  'RMSE(cat)': float(rmse_score_cat)}
    
    return score_dict


