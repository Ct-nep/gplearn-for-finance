"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers
from collections.abc import Iterable

import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.utils.validation import check_array


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

def _check_input(X, y, sample_weight, trans_args):
    '''Private function to make sure inputs are ndarray.'''
    idx, col = None, None
    if y is None and sample_weight is not None:
        raise ValueError('Input y is missing.')
    # Convert X DataFrames to ndarray
    if isinstance(X, Iterable) \
        and all([isinstance(i, pd.DataFrame) for i in X]):
        # warn(f'Received DataFrames as X, try converting.')
        if isinstance(y, pd.DataFrame):
            size, idx, col = y.shape, y.index, y.columns
        else:
            size, idx, col = X[0].shape, X[0].index, X[0].columns
        if not all([i.shape==size and (i.index==idx).all() \
                    and (i.columns==col).all() for i in X]):
            raise ValueError('All DataFrames must have same '
                                'shape, indices and columns')
        X = np.stack(
            [check_array(i, dtype=float, force_all_finite='allow-nan') \
                for i in X], 
            axis=1,
        )
    # Call check_array on y and other arguments.
    if y is not None:
        y = check_array(y, force_all_finite='allow-nan', 
                        ensure_2d=True, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=float)
    else:
        # sample_weight could be 1d, boradcast it to 2d
        sample_weight = check_array(sample_weight, dtype=float, 
                                    force_all_finite='allow-nan',
                                    ensure_2d=False)
        if len(sample_weight.shape) > 2:
            raise ValueError(f'Invalid sample_weight shape '
                                f'{sample_weight.shape}')
        if len(sample_weight) == 1:
            sample_weight = np.broadcast_to(sample_weight[..., None], 
                                            y.shape)
    if trans_args is None:
        trans_args = dict()
    for k, v in trans_args.items():
        if v is None:
            del trans_args[k]
        # Ensure 2d but no dtype check.
        else:
            try:
                trans_args[k] = check_array(v, dtype=None, 
                                            force_all_finite='allow-nan')
            except Exception as e:
                raise ValueError(f'Cannot coerce additional data {k} '
                                f'to compatible structure: {e}')
    X = check_array(X, force_all_finite='allow-nan', 
                    ensure_2d=False, allow_nd=True, dtype=float)
    # Check data shape, cannot rely on sklearn for 3d data.
    if len(X.shape) != 3:
        raise ValueError(f'Invalid X shape {X.shape}')
    X_shape = (X.shape[0], X.shape[2])
    if y is not None and X_shape != y.shape:
        raise ValueError(f'X, y must have same shape except on '
                            f'n_feature dim, received {X.shape}, {y.shape}')
    for k, v in trans_args.items():
        if X_shape != v.shape:
            raise ValueError(f'Any additional data must have shape '
                                f'(n_samples, n_firms), received '
                                f'{v.shape} for key {k}')
    return X, y, sample_weight, trans_args, idx, col