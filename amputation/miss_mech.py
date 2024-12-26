import torch
import numpy as np
import pandas as pd
#import seaborn as sns
import torch
import os
import random
from scipy import optimize
from sklearn.preprocessing import OneHotEncoder
import itertools


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())

##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    #ber = 0.4 + 0.2 * ber
    mask[:, idxs_nas] = ber < ps

    return mask, 1 - ps, idxs_nas

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    #ber = 0.4 + 0.2 * ber
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask, 1 - ps, idxs_nas

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    #ber = 0.4 + 0.2 * ber
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask, ps


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    #ber = 0.4 + 0.2 * ber
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    n,d  = X.shape
    mask = torch.zeros(n, d).bool()

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        #mask, ps = MAR_mask(X, p_miss, p_obs).double()
        ps = torch.ones_like(X)
        mask, ps_nas, idxs_nas = MAR_mask(X, p_miss, p_obs)
        ps[:, idxs_nas] = ps_nas
        miss_col_idx = np.array([d_ in idxs_nas for d_ in range(d)])
        #mask.double(), ps.double()
    elif mecha == "MNAR": #and opt == "logistic":
        #mask, ps = MNAR_mask_logistic(X, p_miss, p_obs, exclude_inputs=False).double()
        ps = torch.ones_like(X)
        d_na = d - int(d * p_obs)
        idxs_nas = np.random.choice(d, d_na, replace=False)
        X_in = X[:, idxs_nas]
        mask_nas, ps_nas, _ = MNAR_mask_logistic(X_in, p_miss, exclude_inputs=False)
        ps[:, idxs_nas] = ps_nas
        mask[:, idxs_nas] = mask_nas
        miss_col_idx = np.array([d_ in idxs_nas for d_ in range(d)])

    # elif mecha == "MNAR" and opt == "quantile":
    #     mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    # elif mecha == "MNAR" and opt == "selfmasked":
    #     mask, ps = MNAR_self_mask_logistic(X, p_miss).double()
    
    else:  # MCAR
        #mask = (torch.rand(X.shape) < p_miss).double()
        d_na = d - int(d * p_obs)
        idxs_nas = np.random.choice(d, d_na, replace=False)
        ber = torch.rand(n, d_na)
        #ber = 0.4 + 0.2 * ber
        ps = torch.ones_like(X)
        ps[:, idxs_nas] = 1 - p_miss * ps[:, idxs_nas]
        ps_tar = ps[:, idxs_nas].clone()
        mask[:, idxs_nas] = ber > ps_tar
        miss_col_idx = np.array([d_ in idxs_nas for d_ in range(d)])
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask, 'ps': ps, 'miss_col_idx': miss_col_idx}


##############################################################################################################

def generate_missing(X_num, X_cat, p, idxs_nas = None, idxs_obs = None, mechanism = 'MAR', seed = 42, 
                        c = 0., v = 1., power = 1, var_select = False, standardize = True, p_obs_col = 0.3, 
                        nmar_all = False, abs_adj = False):
    set_all_seeds(seed)
    # input: X_num, X_cat mechanism, p, p_obs, idxs_nas, seed

    # categorical input process
    if X_cat is not None:
        X_cats_raw = X_cat.numpy()
        onehot = OneHotEncoder(handle_unknown='ignore')
        onehot.fit(X_cats_raw)
        X_cats_new = onehot.transform(X_cats_raw).toarray()
        cat_inds = onehot.categories_
        categories = [len(x) for x in cat_inds]
        if X_num is not None:
            X_tmp = np.concatenate([X_num.numpy(), X_cats_new], axis = 1)## miss mech input
        else:
            X_tmp = X_cats_new
    else:
        categories = None
        X_tmp = X_num.numpy()
    #np.concatenate([X_num.numpy(), X_cats_new], axis = 1)## miss mech input
    
    n = X_tmp.shape[0]
    
    d_numerical = X_num.shape[1] if X_num is not None else 0
    d_categorical = X_cat.shape[1] if X_cat is not None else 0
    d = d_numerical + d_categorical
    
    to_torch = torch.is_tensor(X_num) or torch.is_tensor(X_cat) ## output a pytorch tensor, or a numpy array
    
    if not to_torch:
        X_num = torch.from_numpy(X_num) if X_num is not None else None
        X_cat = torch.from_numpy(X_cat) if X_cat is not None else None

    # Case1. na idx & obs idx given, Case2. na idx given, Case3. no idx given -> sampling
    
    d_obs_num = max(int(p_obs_col * d_numerical), 0) if d_numerical != 0 else 0 ## number of variables that will have no missing values (at least one variable)
    d_obs_cat = max(int(p_obs_col * d_categorical), 0) if d_categorical != 0 else 0
    
    d_na_num = d_numerical - d_obs_num
    d_na_cat = d_categorical - d_obs_cat
    
    
    if idxs_nas is None:
        idxs_nas_num = np.random.choice(d_numerical, d_na_num, replace=False)
        idxs_nas_cat = np.random.choice(list(range(d_numerical, d)), d_na_cat, replace=False)   
        
        idxs_nas = np.concatenate([idxs_nas_num, idxs_nas_cat])
                
        if (mechanism == 'MNAR'):
            if (nmar_all == False):  # self logistics
                idxs_obs = np.sort(np.array([i for i in range(d) if i in idxs_nas])) #np.concatenate([idxs_obs_num, idxs_nas_cat])
            else:
                idxs_obs = np.array([i for i in range(d)])
        else:
            idxs_obs = np.sort(np.array([i for i in range(d) if i not in idxs_nas]))
        
            # idxs_obs = np.random.choice(d, d_obs, replace=False)
            # idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
            
    else:
        pass
    
    # get expanded for all variables after ohe
    
    num_col_inds = list(range(d_numerical))+[d_numerical] if d_numerical != 0 else []
    cat_col_inds = np.cumsum([[d_numerical] + categories]).tolist()[1:] if categories is not None else []
    if d_numerical != 0:
        if categories is not None:
            col_inds = num_col_inds + cat_col_inds
        else:
            col_inds = num_col_inds 
    else:
        col_inds = [0] + cat_col_inds
        
    
    # print(idxs_nas)
    # print(idxs_obs)
    # print(col_inds)
    
    obs_ids_new = list(itertools.chain(*[list(range(col_inds[ids], col_inds[ids+1])) for ids in np.sort(idxs_obs)]))

    d_obs = len(obs_ids_new)
    d_na = len(idxs_nas) #len(na_ids_new)
    X_tmp = torch.tensor(X_tmp).float()
    
    
    # Could make the mechanism nonlinear / harder
    if (var_select == True):
        coeffs = c +  v * (torch.randn(d_obs, d_na) ** power) *  (torch.rand(d_obs, d_na) > 0.5)
    else: 
        coeffs = c +  v * (torch.randn(d_obs, d_na) ** power)
        
    coeffs = abs(coeffs) if abs_adj else coeffs

    if standardize:
        Wx = X_tmp[:, obs_ids_new].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)

    if mechanism == 'MCAR':
        coeffs = 0 * coeffs
        
    intercepts = fit_intercepts(X_tmp[:, obs_ids_new], coeffs, 1 - p)
    
    
    # propensity
    ps_tmp = torch.sigmoid(X_tmp[:, obs_ids_new].mm(coeffs) + intercepts)
    
    # miss_mask
    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
    mask[:, idxs_nas] = (torch.bernoulli(ps_tmp) == 0)

    # ps adjustment
    ps = 1.0* torch.ones_like(mask)
    ps[:, idxs_nas] = ((ps_tmp.float()) + 1e-6) / (1+1e-6)
    
    return ps, mask, coeffs