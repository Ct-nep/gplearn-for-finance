"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import bottleneck as bn
from joblib import wrap_non_picklable_objects
from collections.abc import Iterable

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity
        self.valid_range = (None,)

    def __call__(self, param, *args):
        return self.function(*args)
    
    
class _TSFunction(_Function):
    """A subclass of _Function which require a possibly varying
    parameter of lookback window to be executed. Generally used for
    time-series calculation e.g. 20 day moving average.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.
        
    valid_range : list-like
        A Iterable where each element is a valid option of lookback
        window length. During evolution, phenotypes will call this
        function with lookback initialized / mutated / passed by parents
        from this list.
        If multiple params required, they could be organized as tuples
        of set product.
    """
    def __init__(self, function, name, arity, valid_range):
        super().__init__(function, name, arity)
        assert isinstance(valid_range, Iterable) and len(valid_range), \
            f'Invalid range: {valid_range}'
        self.valid_range = valid_range
    
    def __call__(self, param, *args):
        return self.function(param, *args)


def make_function(*, function, name, arity, wrap=True, skip_check=False):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.
        
    skip_check : bool, optional, default = False
        Wether or not to validate inputs before wrapping the function.

    Returns
    -------
    fn : _TSFunction
        Wrapped and pickled function that can be registered and used during
        evolution.
    """
    if not skip_check:
        if not isinstance(arity, int):
            raise ValueError('arity must be an int, got %s' % type(arity))
        if not isinstance(function, np.ufunc):
            if function.__code__.co_argcount != arity:
                raise ValueError('arity %d does not match required number of '
                                'function arguments of %d.'
                                % (arity, function.__code__.co_argcount))
        if not isinstance(name, str):
            raise ValueError('name must be a string, got %s' % type(name))
        if not isinstance(wrap, bool):
            raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)
    
def make_ts_function(*, function, name, arity, valid_range, 
                     wrap=True, skip_check=False) -> object:
    """Make a time-series variant of function, _TSFunction instance.

    Parameters
    ----------
    function : callable
        A function with signature `function(params, x1, ...)` that returns a
        Numpy array of the same shape as one of its input data array.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.
        
    valid_range : list-like
        A Iterable where each element is a valid option of lookback
        window length. During evolution, phenotypes will call this
        function with lookback initialized / mutated / passed by parents
        from this list.

    wrap : bool, optional, default = True
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.
        
    skip_check : bool, optional, default = False
        Wether or not to validate inputs before wrapping the function.

    Returns
    -------
    fn : _TSFunction
        Wrapped and pickled function that can be registered and used during
        evolution.
    """
    if not skip_check:
        if not isinstance(arity, int):
            raise ValueError('arity must be an int, got %s' % type(arity))
        if not isinstance(function, np.ufunc):
            if function.__code__.co_argcount-1 != arity:
                raise ValueError('arity %d does not match required number of '
                                'function arguments of %d.'
                                % (arity, function.__code__.co_argcount-1))
        if not isinstance(name, str):
            raise ValueError('name must be a string, got %s' % type(name))
        if not isinstance(wrap, bool):
            raise ValueError('wrap must be an bool, got %s' % type(wrap))
        if not isinstance(valid_range, Iterable) \
            or not all([np.issubdtype(type(i), int) for i in valid_range]):
            raise ValueError('date_range must be Iterable with integers.')
    if wrap:
        return _TSFunction(function=wrap_non_picklable_objects(function), 
                           name=name, arity=arity, valid_range=valid_range)
    return _TSFunction(function=function, name=name, arity=arity, 
                       valid_range=valid_range)

def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(x1, x2)
        return np.where(np.isfinite(res), res, np.nan)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

########################
# Customized functions #
########################

# Timing functions, which return scalar each day and broadcasted to 2d.
def _csmax(x1):
    '''Cross-sectional (daily) max. Bottleneck doesn't support quantile
    which may works better.'''
    res = bn.nanmax(x1, axis=1)
    return np.where(np.isfinite(x1), res[:, None], np.nan)

def _csmin(x1):
    '''Cross-sectional (daily) min.'''
    res = bn.nanmin(x1, axis=1)
    return np.where(np.isfinite(x1), res[:, None], np.nan)

def _csmean(x1):
    '''Cross-sectional (daily) mean.'''
    res = bn.nanmean(x1, axis=1)
    return np.where(np.isfinite(x1), res[:, None], np.nan)

def _csmedian(x1):
    '''Cross-sectional (daily) median.'''
    res = bn.nanmedian(x1, axis=1)
    return np.where(np.isfinite(x1), res[:, None], np.nan)
    
def _csstd(x1):
    '''Cross-sectional (daily) std'''
    with np.errstate(over='ignore', under='ignore'):
        res = bn.nanstd(x1, axis=1)
    return np.where(np.isfinite(x1), res[:, None], np.nan)
    
# Cross-sectional transform functions, which return array each day using
# only daily info.
def _csdemean(x1):
    '''Shortcut for x1 - csmean(x1), substitute of industry
    neutralization which is too costly to implement as op.'''
    with np.errstate(over='ignore', under='ignore'):
        return x1 - bn.nanmean(x1, axis=1)[:, None]

def _csrank(x1):
    '''Rank daily data as pct'''
    res = bn.nanrankdata(x1, axis=1)
    with np.errstate(over='ignore', under='ignore'):
        res = (res - 1.) / (bn.nanmax(res, axis=1)[:, None] - 1.)
        return res

add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

# Customized operators
rank1 = make_function(function=_csrank, name='rank', arity=1, skip_check=True)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}
