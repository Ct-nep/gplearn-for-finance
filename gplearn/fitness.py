"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers
from typing import *

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    # if not isinstance(function(np.array([1, 1]),
    #                   np.array([2, 2]),
    #                   np.array([1, 1])), numbers.Number):
    #     raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)

# Add below helper functions for backtest.

def winsorize(weight: np.ndarray, left: float = 0., right: float = 1.):
    '''Winsorize weight matrix so each day, weight percentiles lower
    than left bound / higher than right bound will be clipped to
    boundary values. WARNING: There is no validation on parameters for
    performance.
    
    Parameters
    ----------
    weight : array-like, shape = (n_days, n_firms)
        Raw portfolio weight.
        
    left : float, default = 0.
        Left boundary (minimum) percentile for each day in range (0., 1.). 
        
    right : float, default = 1.
        Right boundary (maximum) percentile for each day in range (0., 1.). 
        
    Returns
    -------
    weight_winsorized : array-like of shape (n_days, n_firms)
        The winsorized weight array.
    '''
    minimum, maximum = np.nanquantile(weight, (left, right), axis=1)
    weight = np.clip(weight, minimum[..., None], maximum[..., None])
    return weight

def neutralize(weight: np.ndarray, industry: Union[None, np.ndarray] = None):
    '''Neutralize weight matrix so each day, positive and negative
    weights sum up to 0.
    
    Parameters
    ----------
    weight : array-like, shape = (n_days, n_firms)
        Raw portfolio weight.
        
    industry : Union[None, array-like], default = None
        If provided, must be int-like matrix of shape (n_days, n_firms).
        Industry classification for neutralization within each group, if
        not provided, neutralize all intruments within the market.
        
    Returns
    -------
    weight_neutralized : array-like of shape (n_days, n_firms)
        The neutralized weight array, so at each day long and short
        sides sum up to 0.
    '''
    # Market-wise neutralization if no industry info.
    if industry is None:
        return weight - np.nanmean(weight, axis=1, keepdims=True)
    if industry.shape != weight.shape:
        raise ValueError(f'Weight and industry have different shape:'
                         f'{weight.shape}, {industry.shape}')
    ind_ls = np.unique(industry)
    # Use loop as matrix representation costs too much memory.
    for i in ind_ls:
        if not np.isfinite(i): continue
        mask = (industry == i)
        mean = np.nanmean(np.where(mask, weight, 0.), axis=1, keepdims=True)
        weight -= np.where(mask, mean, 0.)
    return weight

def rebalance(weight: np.ndarray, min_obs: int = 1):
    '''Rebalance long and short side so they sum up to +- 1. Do not
    require the weight to be neutralized in advance. If either side has
    valid stocks lower than min_obs, treat the whole side as 0.
    
    Parameters
    ----------
    weight : array-like of shape (n_days, n_firms)
        Raw portfolio weight.
    
    min_obs: int, default=1
        At each day if long / short side have valid weight count lower
        than this value, all weights on that side are set to 0.
        
    Returns
    -------
    weight_balanced : array-like of shape (n_days, n_firms)
        The balanced weight array, so at each day weights on long side
        sum up to 1, and short side to -1.
    '''
    pos_mask = weight > 0.
    pos_count = pos_mask.sum(axis=1, keepdims=True)
    pos_sum = np.where(pos_mask, weight, 0.).sum(axis=1, keepdims=True)
    pos_sum = np.where(pos_count < min_obs, np.inf, pos_sum)
    neg_mask = weight < 0.
    neg_count = neg_mask.sum(axis=1, keepdims=True)
    neg_sum = -np.where(neg_mask, weight, 0.).sum(axis=1, keepdims=True)
    neg_sum = np.where(neg_count < min_obs, np.inf, neg_sum)
    weight = np.where(pos_mask, weight / pos_sum, weight)
    weight = np.where(neg_mask, weight / neg_sum, weight)
    return weight

def clip_limit(weight: np.ndarray, permit_long: np.ndarray, 
               permit_short: np.ndarray):
    '''Clip weight based on price limit info, so firm-day weight cannot
    be higher than previous value unless permit_long is True, and vice
    versa. 
    
    Parameters
    ----------
    weight : array-like of shape (n_days, n_firms)
        Raw portfolio weight.
    
    permit_long: array-like of shape (n_days, n_firms)
        A firm-day's weight will be clipped to previous value if higher
        and not permitted to long.
        
    permit_short: array-like of shape (n_days, n_firms)
        A firm-day's weight will be clipped to previous value if lower
        and not permitted to short.
        
    Returns
    -------
    weight_clipped : array-like of shape (n_samples, n_days, n_firms)
        The clipped weight array, so unless a firm-day is permitted to
        be furtherly long / short-ed, its weight cannot be hihger /
        lower than previous value.
    '''
    # Fillna as NaN cannot be compared.
    mask = np.isfinite(weight)
    weight = np.where(mask, weight, 0.)
    greater = weight > np.roll(weight, 1, axis=0)
    greater[0] = weight[0] > 0.
    valid = (permit_long | ~greater) & (permit_short | greater)
    # Similar to pandas's ffill.
    idx = np.tile(np.arange(valid.shape[0]), (valid.shape[1], 1)).T
    idx = np.where(valid, idx, 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    weight[~valid] = weight[idx[~valid], np.nonzero(~valid)[1]]
    weight = np.where(mask, weight, np.nan)
    return weight

def df_stats(y, y_pred, start_size=1.):
    '''Used for 2D DataFrame weight summary. Return annually IC,
    non-compounding returns, sharpe ratio etc. in DataFrame, plus
    cumulative PNL.
    
    Parameters
    ----------
    y : DataFrame, shape = (n_samples, n_firms)
        Real daily return of each firm-day.
        
    y_pred : DataFrame, shape = (n_samples, n_firms)
        Valid weight of each firm-day. "Valid" means weights must be long-short
        neutralized, each side sum up to 1 (-1) at each day.
        
    start_size : float, optional, default = 1.
        Set the starting size of portfolio. Only affect how cumulative PNL is
        presented.
        
    Returns
    -------
    stats : DataFrame
        Annual and all-sample stats.
        
    pnl : Series
        Daily series of cumulative pnl.
    '''
    def get_stats(df):
        res_dict = {
            'days': df.shape[0],
            'ic': df['corr'].mean(),
            'ret': df['return'].mean() * 250,
            'sharpe': df['return'].mean() / df['return'].std() * np.sqrt(250),
            'ret_long': df['return_long'].mean() * 250,
            'sharpe_long': df['return_long'].mean() / df['return_long'].std() \
                * np.sqrt(250),
            'long_size': df['long_size'].mean(),
            'short_size': df['short_size'].mean(),
            'min_coverage': df['coverage'].min(),
            'coverage': df['coverage'].mean(),
            'maxwt': df['max_weight'].max(),
            'avg_maxwt': df['max_weight'].mean(),
            'max_tvr': df['turnover'].max(),
            'avg_tvr': df['turnover'].mean(),
            'max_dd': (df['cumret'] - df['cumret'].cummax()).min(),
        }
        return pd.Series(res_dict)
        
    if y.shape != y_pred.shape:
        raise ValueError(f'y and y_pred have different shape: '
                         f'{y.shape}, {y_pred.shape}')
    if len(y.shape) != 2:
        raise ValueError(f'Expect 2d input (date, firm), '
                         f'got {y.shape}')
    # Drop initial all-NaN period
    mask = pd.isna(y_pred).all(axis=1)
    idx = 0
    while mask.iat[idx] and (idx < mask.shape[0]):
        idx += 1
    y, y_pred = y.iloc[idx:], y_pred.iloc[idx:]
    mask = np.isfinite(y) & np.isfinite(y_pred)
    wt = y_pred.where(mask, 0.)
    wt_center = (wt - np.nanmean(wt, axis=1, keepdims=True)).where(mask, 0.)
    y_center = (y - np.nanmean(y, axis=1, keepdims=True)).where(mask, 0.)
    tvr = wt.diff(1)
    tvr.iloc[0] = wt.iloc[0]
    ret = y_pred * y
    res = (mask.sum(axis=1) / np.isfinite(y).sum(axis=1)).to_frame('coverage')
    res['corr'] = (wt_center * y_center).sum(axis=1)
    res['corr'] /= np.sqrt(wt_center.pow(2).sum(axis=1) \
        * y_center.pow(2).sum(axis=1))
    res['long_size'] = wt.where(wt > 0., 0.).sum(axis=1)
    res['short_size'] = wt.where(wt < 0., 0.).sum(axis=1)
    res['turnover'] = tvr.abs().sum(axis=1)
    res['return'] = ret.sum(axis=1)
    res['return'] -= 5e-4 * res['turnover']
    res['return'] /= 2.
    res['return_long'] = ret.where(wt > 0., 0.).sum(axis=1)
    res['return_long'] -= 5e-4 * tvr.where(tvr > 0., 0.).sum(axis=1)
    res['cumret'] = res['return'].cumsum() + start_size
    res['max_weight'] = wt.abs().max(axis=1)
    grouper = pd.Grouper(freq='Y')
    stats = res.groupby(grouper).apply(get_stats)
    stats.index = stats.index.year
    full = get_stats(res).to_frame('ALL').T
    stats = pd.concat([stats, full], axis=0)
    # Filter out short periods.
    stats = stats[stats['days'] > 10]
    return stats, res['cumret']


weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss}
