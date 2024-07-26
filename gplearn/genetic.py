"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import itertools
import pathlib
import pickle
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Callable
from copy import copy
from operator import neg
from time import time
from typing import *
from warnings import warn

import numpy as np
import pandas as pd
import cloudpickle
from joblib import Parallel, delayed
# from scipy.stats import rankdata
from scipy.optimize import minimize, Bounds, LinearConstraint
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from ._program import _Program
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators, check_random_state, _get_n_jobs

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


def _parallel_evolve(n_programs, parents, X, y, sample_weight, 
                     trans_args, seeds, params):
    """Private function used to build a batch of programs within a job.
    
    It seems joblib has some problem to serialize ``parents`` properly, which
    takes lengthy time for large list of ancestors. This could be mitigated by
    mannually pickling it before call this function, then unpickle it on the
    run. Use cloudpickle here since joblib already relies on it.
    """
    n_samples, n_features = X.shape[:2]
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    p_grow_terminal = params['p_grow_terminal']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    is_split = params['is_split']
    feature_names = params['feature_names']
    
    # We are sending serialized program, decode it.
    if isinstance(parents, bytes):
        parents = cloudpickle.loads(parents)
    is_split = int(is_split * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_grow_terminal=p_grow_terminal,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            # curr_sample_weight = np.ones((n_samples,))
            curr_sample_weight = np.ones_like(y, dtype=float)
        else:
            curr_sample_weight = sample_weight.copy()
        # # Broadcast weight matrix to be compatible with y.
        # if y.shape != curr_sample_weight.shape:
        #     err_msg = f'Cannot cast sample weight to the shape of y:'\
        #         f'{curr_sample_weight.shape} -> {y.shape}'
        #     diff = len(y.shape) - len(curr_sample_weight.shape)
        #     if diff > 0:
        #         # Assume later dims are missing.
        #         curr_sample_weight = np.expand_dims(
        #             curr_sample_weight,
        #             axis=np.arange(diff)+len(y.shape),
        #         )
        #         try:
        #             curr_sample_weight = np.broadcast_to(curr_sample_weight, 
        #                                                 y.shape).copy()
        #         except Exception:
        #             raise RuntimeError(err_msg)
        #     else:
        #         raise RuntimeError(err_msg)
        os_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples, is_split)

        curr_sample_weight[not_indices] = 0
        os_sample_weight[indices] = 0

        res = program.raw_fitness(X, y, curr_sample_weight, 
                                  trans_args, return_pnl=True)
        program.raw_fitness_, program._pnl = res
        if is_split < n_samples:
            # Calculate OS fitness
            program.os_fitness_ = program.raw_fitness(X, y, os_sample_weight,
                                                      trans_args)

        programs.append(program)

    return cloudpickle.dumps(programs)


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_grow_terminal=0.1,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 elitism=1,
                 variety='corr',
                 is_split=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 callbacks=None,
                 random_state=None):
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_grow_terminal = p_grow_terminal
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.elitism = elitism
        self.variety = variety
        self.is_split = is_split
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.callbacks = callbacks
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^26}|{:^26}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 26 + ' ' + '-' * 26 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}'
            print(line_format.format('Gen', 'Length', 'Variety', 'Fitness', 
                                     'Length', 'Fitness', 'OS Fitness', 
                                     'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            os_fitness = 'N/A'
            line_format = \
                '{:4d} {:8.2f} {:8.3f} {:8.4f} {:8d} {:8.4f} {:>8} {:>10}'
            if self.is_split < 1.0:
                os_fitness = run_details['best_os_fitness'][-1]
                line_format =  '{:4d} {:8.2f} {:8.2f} {:8.4f} {:8d} '\
                    '{:8.4f} {:8.4f} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['variety'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     os_fitness,
                                     remaining_time))
    
    def validate_data(self, X, y, sample_weight, trans_args, fit=False):
        '''Patch parent's _validate_data() method for 3-dim situation. Make it 
        public so building blocks can rely on processed ndarray input only.
        Due to drastic change of behavior, it is no long suitable to use 
        sklearn's validate method.
        
        Support inputs of multiple DataFrame as X and single DataFrame as y. If
        this is the case, convert them to ndarray and preserve index / columns
        info.
        
        Parameters
        ----------
        X : List[pd.DataFrame] or array-like, shape = (n_samples, n_features,
        n_firms)
            See self.fit().

        y : None or pd.DataFrame or array-like, shape = (n_samples, n_firms)
            See self.fit().

        sample_weight : None or pd.DataFrame or array-like, shape =
        (n_samples,) or (n_samples, n_firms)
            See self.fit().
            
        trans_args : None or dict of DataFrame / array-like
            See self.fit().
            
        fit : bool, default=False
            If set to True, overwrite regressor with inputs' feature number and
            names, otherwise only check if they match with previous call of
            fit.

        Returns
        -------
        out : tuple
            All input data converted to correct dtype and shape, plus index and
            columns info.
        '''
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
            # Ensure float unless for classification task.
            y_dtype = None if isinstance(self, ClassifierMixin) else float
            y = check_array(y, force_all_finite='allow-nan', dtype=y_dtype)
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
        X = super()._validate_data(X, force_all_finite='allow-nan', 
                                   reset=fit, ensure_2d=False, 
                                   allow_nd=True, dtype=float)
        super()._check_n_features(X, reset=fit)
        # Check data shape, cannot rely on sklearn for 3d data.
        if len(X.shape) != 3:
            raise ValueError(f'Invalid X shape {X.shape}')
        X_shape = (X.shape[0], X.shape[2])
        if y is not None and X_shape != y.shape:
            raise ValueError(f'X, y must have same shape except on '
                                f'n_feature dim, received {X.shape}, '
                                f'{y.shape}')
        for k, v in trans_args.items():
            if X_shape != v.shape:
                raise ValueError(f'Any additional data must have shape '
                                 f'(n_samples, n_firms), received '
                                 f'{v.shape} for key {k}')
        # Similar to sklearn's check on classifier label.
        if y is not None and isinstance(self, ClassifierMixin):
            if np.issubdtype(y.dtype, np.floating):
                mask = np.isfinite(y)
                if not (y[mask] == y[mask].astype(int)).all():
                    raise ValueError('y cannot be continuous float.')
            elif np.issubdtype(y.dtype, np.integer):
                y = y.astype(float)
            elif not np.issubdtype(y.dtype, np.str_):
                raise ValueError(f'Require label y to be int, string or '
                                    f'int-like float, received {y.dtype}.')
        return (X, y, sample_weight, trans_args, idx, col)
    
    def get_params(self, deep=True, raw=False):
        '''Override BaseEstimator function to correctly validate & pass
        arguments required in ``fit``.
        
        Parameters
        --------
        deep : bool, optional, default = True
            If True, make deep copy of sub-Estimator object in the param dict.
            
        raw : bool, optional, default = False
            If True, return params in ``__init__`` signature as-is, which is
            expected by callbacks.
            
        Returns
        --------
        params : dict
            Param dict with critical params readily usable in evolution.
        '''
        params = super().get_params(deep)
        if raw:
            return params
        
        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')
        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)
        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)
        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')
        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            else:
                raise ValueError('Invalid `transformer`. Expected _Function '
                                 'object, got %s' % type(self.transformer))
            if self._transformer.arity != 2:
                raise ValueError('Invalid arity for `transformer`. Expected 2,'
                                 ' got %d.' % (self._transformer.arity))
        if isinstance(self.elitism, float):
            if not 0. < self.elitism < 1.:
                raise ValueError(f'Elitism ratio must in range (0., 1.), '
                                 f'got {self.elitism}')
            self.elitism = int(self.elitism * self.population_size)
        elif self.elitism is None:
            self.elitism = 0
        if not 0 <= self.elitism < self.population_size:
            raise ValueError(f'Elitism number of genotypes must in range '
                             f'[0, population_size), got {self.elitism}')
        
        params['_metric'] = self._metric
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        return params
    
    def fit(
        self, 
        X: Union[List[pd.DataFrame], np.ndarray], 
        y: np.ndarray, 
        sample_weight: Union[None, np.ndarray] = None, 
        trans_args: Union[None, Dict[str, np.ndarray]] = None,
    ) -> object:
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like, shape = (n_samples, n_features,
        n_firms)
            Training vectors, where n_samples is the number of trading days as
            samples, n_features is the number of features, n_firms is number of
            firms / stocks.  If provided as DataFrames, must have same index /
            columns, with shape (n_samples, n_firms).

        y : None, pd.DataFrame or array-like, shape = (n_samples, n_firms)
            Target values, if provided as DataFrame, must have same index /
            columns with all X DataFrame, with shape (n_samples, n_firms).

        sample_weight : None or pd.DataFrame or array-like, shape =
        (n_samples,) or (n_samples, n_firms), optional, default = None
            Weights applied to individual samples. Will be broadcast to 2d if
            supplied as 1d array / Series.
            
        trans_args : None or dict, optional, default = None
            Additional information like instrument universe, industry
            classification etc. If provided, will be passed to fitness
            evaluation function. The method only check shape of these data,
            since it does not know what more to be expected.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.variety not in [None, 'corr', 'fitness']:
            raise ValueError(f'Expect variety in [None, "corr", "fitness"], '
                             f'got {self.variety}')
        if not 0 < self.p_grow_terminal < 1:
            raise ValueError(f'p_grow_terminal must in range (0., 1.), '
                             f'got {self.p_grow_terminal}')
        if not 0 < self.is_split <= 1:
            raise ValueError(f'is_split must in range (0., 1.], '
                             f'got {self.is_split}')
        if self.callbacks is not None:
            if isinstance(self.callbacks, Callable):
                self.callbacks = [self.callbacks]
            elif isinstance(self.callbacks, Iterable):
                if not all([isinstance(i, Callable) for i in self.callbacks]):
                    raise ValueError('callbacks sequence has '
                                     'non-callable object.')
            else:
                raise ValueError('callbacks must be callable, '
                                     'or list of callable.')

        random_state = check_random_state(self.random_state)
        X, y, sample_weight, trans_args, _, _ = \
            self.validate_data(X, y, sample_weight, trans_args, fit=True)
        if isinstance(self, ClassifierMixin):
            n_samples, n_firms = y.shape
            if self.class_weight:
                # Modify the sample weights with the corresponding class
                # weight.
                sample_weight = sample_weight.reshape(-1,)
                sample_weight = (sample_weight *
                                 compute_sample_weight(self.class_weight, 
                                                       y.reshape(-1,)))
                sample_weight = sample_weight.reshape(n_samples, n_firms)

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            y = y.reshape(n_samples, n_firms).astype(float)
            self.n_classes_ = len(self.classes_)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        # if isinstance(self.metric, _Fitness):
        #     self._metric = self.metric
        # elif isinstance(self, RegressorMixin):
        #     if self.metric not in ('mean absolute error', 'mse', 'rmse',
        #                            'pearson', 'spearman'):
        #         raise ValueError('Unsupported metric: %s' % self.metric)
        #     self._metric = _fitness_map[self.metric]
        # elif isinstance(self, ClassifierMixin):
        #     if self.metric != 'log loss':
        #         raise ValueError('Unsupported metric: %s' % self.metric)
        #     self._metric = _fitness_map[self.metric]
        # elif isinstance(self, TransformerMixin):
        #     if self.metric not in ('pearson', 'spearman'):
        #         raise ValueError('Unsupported metric: %s' % self.metric)
        #     self._metric = _fitness_map[self.metric]

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not((isinstance(self.const_range, tuple) and
                len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {
                'generation': [],
                'average_length': [],
                'average_fitness': [],
                'best_length': [],
                'best_fitness': [],
                'best_os_fitness': [],
                'generation_time': [],
                'variety': [],
            }

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()
            params = self.get_params()
            
            if gen == 0:
                parents = None
            else:
                # Parallel has serialization issue ONLY on this list
                # esp. with high n_jobs. If sent as-is, this incurs
                # significant performance costs since second generation.
                # Dump it and send bytes magically saves time.
                parents = cloudpickle.dumps(self._programs[gen - 1])

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          trans_args,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable([
                cloudpickle.loads(i) for i in population
            ]))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]
            
            # Apply elitism to preserve (near unique) best performers in
            # previous generation, after parsimony penalty.
            if self.elitism and gen > 0 and self._programs[gen-1]:
                fit_old = [program.fitness_ 
                           for program in self._programs[gen-1]]
                # Identical genotypes will be dropped by duplicative
                # fitness (after parsimony penalty).
                fit_unique, idx = np.unique(fit_old, return_index=True)
                left = min(self.elitism, len(fit_unique))
                right = len(fit_unique) - left
                if self._metric.greater_is_better:
                    best = fit_unique.argpartition(right)[right:]
                    worst = np.argpartition(fitness, left)[:left]
                else:
                    best = fit_unique.argpartition(left)[:left]
                    worst = np.argpartition(fitness, right)[right:]
                best = idx[best]
                for i in range(len(best)):
                    population[worst[i]] = copy(self._programs[gen-1][best[i]])
                    population[worst[i]].parents = {
                        'method': 'Reproduction',
                        'parent_idx': best[i],
                        'parent_nodes': [],
                    }
                    fitness[worst[i]] = population[worst[i]].raw_fitness_
                    length[worst[i]] = population[worst[i]].length_
                    
            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)
            
            self._programs.append(population)

            # Always remove cached old PNL.
            if gen > 0 and self._programs[gen-1]:
                for program in self._programs[gen-1]:
                        program._pnl = None
            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None
                
            # Calculate population variety:
            #  - Method 'fitness' uses a non-parametric estimation of
            #    binned phenotype (raw fitness) distribution entropy,
            #    which is derived from -sum(p_i * log(p_i)) / log(10),
            #    where p_i is prob of raw fitness falling into i-th
            #    equally spaced bin out of 10. 
            #  - Method 'corr' requires pnl corr matrix (additional
            #    memory cost for caching) and is derived from 1.0 -
            #    norm(corr) / sqrt(corr.size).
            #  - Method None leaves variety as NaN.
            variety = np.nan
            if self.variety == 'fitness':
                temp = np.unique(fitness)
                if len(temp) < 10:
                    variety = np.nan
                else:
                    minimum, maximum = temp[1], temp[-2]
                    temp = np.clip(fitness, minimum, maximum)
                    bins = np.linspace(minimum, maximum+1e-9, 10)
                    bins = np.digitize(fitness, bins)
                    _, count = np.unique(bins, return_counts=True)
                    count = count / len(fitness)
                    variety = -np.nansum(count * np.log(count)) / np.log(10.)
            elif self.variety == 'corr':
                corr = np.stack([program._pnl for program in population], 
                               axis=1)
                # Remove NaNs.
                corr = np.corrcoef(corr, rowvar=False)
                mask = ~np.isnan(corr).all(axis=0)
                mask = np.nonzero(mask)[0] 
                corr = corr[mask][:, mask]
                variety = 1. - np.linalg.norm(corr) / np.sqrt(corr.size)
            self.run_details_['variety'].append(variety)

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            os_fitness = np.nan
            if self.is_split < 1.0:
                os_fitness = best_program.os_fitness_
            self.run_details_['best_os_fitness'].append(os_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break
                
            if self.callbacks is not None:
                for fn in self.callbacks:
                    fn(self)

        # if isinstance(self, TransformerMixin):
        #     # Find the best individuals in the final generation
        #     fitness = np.array(fitness)
        #     if self._metric.greater_is_better:
        #         hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
        #     else:
        #         hall_of_fame = fitness.argsort()[:self.hall_of_fame]
        #     evaluation = np.array([gp.execute(X) for gp in
        #                            [self._programs[-1][i] for
        #                             i in hall_of_fame]])
        #     if self.metric == 'spearman':
        #         evaluation = np.apply_along_axis(rankdata, 1, evaluation)

        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         correlations = np.abs(np.corrcoef(evaluation))
        #     np.fill_diagonal(correlations, 0.)
        #     components = list(range(self.hall_of_fame))
        #     indices = list(range(self.hall_of_fame))
        #     # Iteratively remove least fit individual of most correlated pair
        #     while len(components) > self.n_components:
        #         most_correlated = np.unravel_index(np.argmax(correlations),
        #                                            correlations.shape)
        #         # The correlation matrix is sorted by fitness, so identifying
        #         # the least fit of the pair is simply getting the higher index
        #         worst = max(most_correlated)
        #         components.pop(worst)
        #         indices.remove(worst)
        #         correlations = correlations[:, indices][indices, :]
        #         indices = list(range(len(components)))
        #     self._best_programs = [self._programs[-1][i] for i in
        #                            hall_of_fame[components]]

        # else:
        # Find the best individual in the final generation
        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            self._program = self._programs[-1][np.argmin(fitness)]

        return self
    
    def make_portfolio(
        self, 
        min_programs: int = 5, 
        max_programs: Union[int, None] = None, 
        fitness_threshold: Union[float, None] = None, 
        corr_threshold: float = 0.9, 
        use_raw_fitness: bool = True,
        is_split: Union[float, None] = None, 
        n_jobs: int = -1,
    ) -> object:
        '''From last generation, pick some best individuals by fitness and
        correlation standard to form a portfolio. 
        
        The weight of programs are not fitted, which can be computed using
        different methods by calling ``fit`` on returned object.
        
        Parameters
        --------
        min_programs : int, optional, default = 5
            Minimum number of selected programs, raise RuntimeError if
            cannot find enough programs.
        
        max_programs : int or None, optional, default = None
            Maximum number of programs. Not setting means no limit.
        
        fitness_threshold : float or None, optional, default = None
            Minimum (if greater is better, vice versa) individual fitness
            required for adding it into the portfolio.
        
        corr_threshold : float, optional, default = 0.9
            Maximum correlation threshold. Reject individual if it has higher
            correlation with any previously picked program.
            
        use_raw_fitness : bool, optional, default = True
            If True, use fitness before parsimony penalty when sorting
            individuals and checking for threshold.
            
        is_split : None or float, optional, default = None
            Same as main regressor's ``is_split``, proportion / numbers of
            in-sample days to calculate mean / variance stats and perform
            optimization. If not set, use main regressor's value.
            
        n_jobs : int, optional, default = -1
            Number of subprocess used by portfolio instance in ``fit`` (only if
            re-computation of PNL required) and ``predict``.  Passed to its
            init args.
            
        Returns
        --------
        portfolio : SymbolicPortfolio
            Unfitted portfolio.
        '''
        if not hasattr(self, '_metric'):
            raise ValueError('Model must have a valid metric.')
        if not hasattr(self, '_programs') or not len(self._programs) \
            or not len(self._programs[-1]):
                raise NotFittedError('SymbolicRegressor not fitted.')
        if min_programs < 1:
            raise ValueError('Invalid min_programs.')
        if max_programs is not None and max_programs < min_programs:
            raise ValueError('max_programs must be no less than min_programs.')
        elif max_programs is None:
            max_programs = np.inf
        n_samples = self._programs[-1][0]._pnl.shape[0]
        if is_split is None:
            is_split = self.is_split
        if isinstance(is_split, float) and 0. < is_split < 1.:
            split = int(is_split * n_samples)
        else:
            raise ValueError('Invalid is_split parameter.')
        
        greater = self._metric.greater_is_better
        if fitness_threshold is None:
            fitness_threshold = -np.inf if greater else np.inf
        key_fn = lambda x: x.raw_fitness_ if use_raw_fitness else x.fitness_
        population = list(sorted(self._programs[-1], key=key_fn, reverse=greater))
        fitness = list(map(key_fn, population))
        pnl = np.stack([program._pnl for program in population], axis=1)
        # Current fitness evalutaion returns all sample pnl regardless of
        # is_split, so we can select different split here.
        pnl = pnl[:split]
        corr = np.corrcoef(pnl, rowvar=False)
        port_idx = [0]
        if greater:
            fitness_threshold *= -1
            fitness = list(map(neg, fitness))
        # From best single program, iteratively add new program to portfolio
        # if it:
        #  1) Has better fitness than given threshold, and;
        #  2) Pairwise correlation with all currently picked programs are
        #     lower than given threshold.
        for i in range(len(population)):
            if fitness[i] > fitness_threshold or len(port_idx) >= max_programs:
                break
            # Compare signed instead of absolute value since portfolio
            # weights are restricted to be positive later.
            if corr[i, port_idx].max() < corr_threshold:
                port_idx.append(i)
        if len(port_idx) < min_programs:
            raise RuntimeError(f'Cannot find enough programs: Found'
                            f'{len(port_idx)} / {min_programs}')
        portfolio = SymbolicPortfolio(
            programs=[population[i] for i in port_idx],
            is_split=is_split,
            n_jobs=n_jobs,
        )
        return portfolio
    
    def save(self, path='./saved_model.pkl'):
        '''Serialize the whole model and dump it to given path.
        
        Under the hood this just calls pickle to do the work. It is not
        suitable for long-term storage, and only guaranteed to work when
        de-serialized in the same python environment.
        
        Parameters
        --------
        path : str or pathlike, optional, default = './saved_model.pkl'
            Path to save the model. Expected to end with filename in extension
            type of .pkl, will be created if not exist.
        '''
        path = pathlib.Path(path)
        if not str(path).endswith('.pkl'):
            raise ValueError('path must be filename with .pkl extension.')
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path, program_only=False):
        '''Load the model file dumped by ``save``.
        
        Parameters
        --------
        path : str or pathlike
            Path to the saved model.
            
        program_only : bool, optional, default = False
            If True, only load back serialized programs in the file. May cause
            unexpected behavior if other parameters are different.
            
        Returns
        --------
        self : object
            Loaded model.
        '''
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, self.__class__):
            raise ValueError('Invalid model file.')
        if program_only:
            if hasattr(model, '_programs'):
                self._programs = model._programs
            if hasattr(model, '_program'):
                self._program = model._program
        else:
            self.__dict__.update(model.__dict__)
        return self  


class SymbolicRegressor(BaseSymbolic, RegressorMixin):

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional, default = 1000
        The number of programs in each generation.

    generations : integer, optional, default = 20
        The number of generations to evolve.

    tournament_size : integer, optional, default = 20
        Size of randomly pitted (with replacement) programs to compete for
        reproduction. Given population P and tournament size T, K-smallest
        fitness has a winning chance of Prob(K) = ((P-K+1)/P) ** T - Prob(K-1)

    stopping_criteria : float, optional, default = 0.0
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional, default = (-1., 1.)
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional, default = (2, 6)
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional, default = 'half and half'
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional, default = ('add', 'sub', 'mul', 'div')
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional, default='mean absolute error'
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.
        
    transformer : None or _Function, optional, default = None
        A transformer function applied to raw weight after calculation, which
        may neutralize weight and apply restriction of tradable instruments at
        each day.  If provided, must be a _Function object created from
        ``make_function`` factory with args (y_pred, trans_args), where
        ``y_pred`` is raw_weight, and ``trans_args`` is a dict passed by
        calling ``fit`` or ``predict`` to store other useful info. See example
        for how could it be utilized.

    parsimony_coefficient : float or "auto", optional, default = 0.001
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.
        
    p_grow_terminal : float, optional, default = 0.1
        The probability that, when growing a tree, choose a real value
        or feature for a node as terminal of the subtree, instead of
        keeping it growing with functions. The max tree depth is still
        capped by ``init_depth``, but it is unlikely to grow a deep tree
        with high terminal prob.

    p_crossover : float, optional, default = 0.9
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional, default = 0.01
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional, default = 0.01
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional, default = 0.01
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced.  Terminals are replaced by other
        terminals and functions are replaced by other functions that require
        the same number of arguments as the original node. The resulting tree
        forms an offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional, default = 0.05
        For point mutation only, the probability that any given node will be
        mutated. 
        Half of the time, only parameters of the function is replaced
        within its valid param range, so the user may want to increase
        this prob more than vanila gplearn version.
        
    elitism : int or float, optional, default = 1
        Number or fraction of best performing samples in previous generation
        that will be carried to next interation by reproduction.
        Ensure best results will never be discarded by randomness, generally
        also eliminate need for reproduction prob.
        NOTE: Elitism preserve highest fitness after parsimony penalty, while
        ``fit`` report best raw fitness before penalty, the reported "best" may
        be discarded due to lower real fitness.
        
    variety : None or "fitness" or "corr", optional, default = None
        Measure of variety of population. If selected, will be report
        alongside other metrics in ``fit``. Two metrics are supported:
        
        - "fitness" : Use 10 equally spaced bins between min / max fitness of
          current population, and estimate non-parametric distribution of
          sample fitness. Calculate entropy of prob density:
          Variety = -sum(p_i * log(p_i)) / log(10)
        - "corr" : Cache each genotype's PNL when evaluating fitness, then
          calculate correlation from stacked PNL matrix: 
          Variety = 1.0 - norm(corr_matrix) / sqrt(corr_matrix.size)

    is_split : float, optional, default = 1.0
        The fraction of samples from X to evaluate each program on. If smaller
        than 1.0, split by ratio so first (n_samples * is_split) entries are
        in-the-sample and others are out-of-sample. This is suitable for
        time-series in ascending order.

    feature_names : list, optional, default = None
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional, default = False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional, default = False
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional, default = 1
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional, default = 0
        Controls the verbosity of the evolution building process.
        
    callbacks : None, callable or Iterable[callable], optional, default = None
        Function or list of functions takes self as only argument and
        return None. If provided, all functions are called in fitting at
        the end of each generation.
        Callbacks can be used to adjust population variety, implement
        dynamic population size, or save current models.

    random_state : int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'variety' : Measure of population variety of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_os_fitness' : The out of sample fitness of the best
          program in the generation (requires `is_split` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 transformer=None,
                 parsimony_coefficient=0.001,
                 p_grow_terminal=0.1,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 elitism=1,
                 variety='corr',
                 is_split=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 callbacks=None,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            transformer=transformer,
            parsimony_coefficient=parsimony_coefficient,
            p_grow_terminal=p_grow_terminal,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            elitism=elitism,
            variety=variety,
            is_split=is_split,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            callbacks=callbacks,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(
        self, 
        X: Union[List[pd.DataFrame], np.ndarray], 
        trans_args: Union[None, Dict[str, np.ndarray]] = None,
        negative: bool = False,
        transform: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Calculate portfolio weight using the best single program
        given a set of inputs.

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like, shape = (n_samples, n_features,
        n_firms)
            Training vectors, where n_samples is the number of trading days as
            samples, n_features is the number of features, n_firms is number of
            firms / stocks.  If provided as DataFrames, must have same index /
            columns, with shape (n_samples, n_firms).
            
        trans_args : optional, None or dict, default = None
            Additional information like instrument universe, industry
            classification etc. If provided, will be passed to weight transform
            function. The method only check shape of these data, since it does
            not know what more to be expected.
            
        negative : bool, default = False
            Take the negative raw weight. This happens before transform (if
            any) to ensure the negative weight is achievable given restrictions
            in transformer,
            
        transform : bool, default = True
            If set to True, would transform raw weight with self._transform
            before returning it.

        Returns
        -------
        weight : pd.DataFrame or array-like, shape = (n_samples, n_firms)
            Returns weight. If X is list of DataFrames, return weight as
            DataFrame of same shape, index and columns.
        """
        if not hasattr(self, '_program'):
            # Snapshot during fit() may not have this attribute, set it
            # as current best.
            if hasattr(self, '_metric') and hasattr(self, '_programs') \
                and len(self._programs) and len(self._programs[-1]):
                fitness = [program.fitness_ for program in self._programs[-1]]
                idx = np.argmax(fitness) \
                    if self._metric.greater_is_better else np.argmin(fitness)
                self._program = self._programs[-1][idx]
            else:
                raise NotFittedError('SymbolicRegressor not fitted.')

        X, _, _, trans_args, ind, col = \
            self.validate_data(X, None, None, trans_args, fit=False)
        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)
        if negative: y = -y
        if transform:
            if hasattr(self, '_transformer'):
                # _Function has null parameter.
                y = self._transformer(None, y, trans_args)
            else:
                raise ValueError('Regressor has no transformer.')
        if ind is not None:
            y = pd.DataFrame(y, index=ind, columns=col)
        return y


class SymbolicPortfolio(BaseEstimator):
    '''Helper class to formulate portfolio from programs. 
    
    Parameters
    --------
    programs : None or List[_Program]
        Candidate programs to form the portfolio.
        
    method : 'equal' | 'sharpe', optional, default = 'sharpe'
        Optimization method.
        - 'equal' forms a simple equal-weight portfolio.
        - 'sharpe' performs mean-varaince optimization to maximize 
            in-sample mean return scaled by standard deviation. Due to possible
            limitation on buy & sale, only positive weights are allowed.
        
    is_split : float, optional, default = 1.0
        The fraction of samples from X in optimization. If smaller than 1.0,
        split by ratio so first (n_samples * is_split) entries are
        in-the-sample and others are out-of-sample. This is suitable for
        time-series in ascending order.
        
    n_jobs : integer, optional, default = 1
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.
        
    verbose : int, optional, default = 0
        Verbosity of parallel evaluation. If > 0, output log from
        ``joblib.Parallel`` in ``fit`` and ``predict``.
        
    Attributes
    -------- 
    weights : array-like
        Weights for each program. Has same shape with ``programs``.
    '''
    def __init__(
        self, 
        programs: Union[None, List[_Program]] = None, 
        method: str = 'sharpe',
        is_split: float = 1.0,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> None:
        self.programs = programs
        self.method = method
        self.is_split = is_split
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.weights = None
        
    def __repr__(self):
        return f'{self.__class__.__name__}(method={self.method}, '\
            f'fitted={self.weights is not None}, '\
            f'n_programs={len(self.programs)})'
    
    def validate_data(self, X, y, sample_weight, trans_args, fit=False):
        '''Patch parent's _validate_data() method for 3-dim situation. Make it 
        public so building blocks can rely on processed ndarray input only.
        Due to drastic change of behavior, it is no long suitable to use 
        sklearn's validate method.
        
        Support inputs of multiple DataFrame as X and single DataFrame as y. If
        this is the case, convert them to ndarray and preserve index / columns
        info.
        
        Basically a replication of ``BaseSymbolic``'s method.
        
        Parameters
        ----------
        X : List[pd.DataFrame] or array-like, shape = (n_samples, n_features,
        n_firms)
            See self.fit().

        y : None or pd.DataFrame or array-like, shape = (n_samples, n_firms)
            See self.fit().

        sample_weight : Any
            Weight is not supported by either method and not used here,
            thus not validated.
            
        trans_args : None or dict of DataFrame / array-like
            See self.fit().
            
        fit : bool, default=False
            If set to True, overwrite regressor with inputs' feature number and
            names, otherwise only check if they match with previous call of
            fit.

        Returns
        -------
        out : tuple
            All input data converted to correct dtype and shape, plus index and
            columns info.
        '''
        idx, col = None, None
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
            # Ensure float unless for classification task.
            y = check_array(y, force_all_finite='allow-nan', dtype=float)
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
        X = super()._validate_data(X, force_all_finite='allow-nan', 
                                   reset=fit, ensure_2d=False, 
                                   allow_nd=True, dtype=float)
        super()._check_n_features(X, reset=fit)
        # Check data shape, cannot rely on sklearn for 3d data.
        if len(X.shape) != 3:
            raise ValueError(f'Invalid X shape {X.shape}')
        X_shape = (X.shape[0], X.shape[2])
        if y is not None and X_shape != y.shape:
            raise ValueError(f'X, y must have same shape except on '
                                f'n_feature dim, received {X.shape}, '
                                f'{y.shape}')
        for k, v in trans_args.items():
            if X_shape != v.shape:
                raise ValueError(f'Any additional data must have shape '
                                 f'(n_samples, n_firms), received '
                                 f'{v.shape} for key {k}')
        # If there is notable all-NaN period at start of X or y, warn.
        mask = np.isnan(X).all(axis=2).any(axis=1)
        mask |= np.isnan(y).all(axis=1)
        idx = 0
        while mask[idx] and (idx < len(mask)):
            idx += 1
        if idx > 0.1 * len(mask):
            warn(f'There is initial NaN period in X, y for up to {idx+1} '
                 f'days. It is recommended to drop initial NaNs to obtain '
                 f'better performance and accurate metrics.')
        return (X, y, sample_weight, trans_args, idx, col)
        
    def fit(
        self,
        X: Union[None, List[pd.DataFrame], np.ndarray] = None, 
        y: Union[None, np.ndarray] = None, 
        sample_weight: Any = None, 
        trans_args: Union[None, Dict[str, np.ndarray]] = None,
        method: Union[None, str] = None,
    ) -> object:
        '''Fit portfolio weight. Using given method.
        
        If only ``method`` is specified, either a equal-weight portfolio is
        formed, or all programs must have pre-calculated ``_pnl`` attribute to
        perform optimization. Otherwise input data is used to perform
        optimization.
        
        Parameters
        ----------
        X : None or List[pd.DataFrame] or array-like, shape = (n_samples,
        n_features, n_firms), optional, default = None
            Training vectors, where n_samples is the number of trading days as
            samples, n_features is the number of features, n_firms is number of
            firms / stocks.  If provided as DataFrames, must have same index /
            columns, with shape (n_samples, n_firms).

        y : None, pd.DataFrame or array-like, shape = (n_samples, n_firms)
            Target values, if provided as DataFrame, must have same index /
            columns with all X DataFrame, with shape (n_samples, n_firms).

        sample_weight : Any
            Weight is not supported by either method and not used here, retains
            signature for sklearn compatibility.
            
        trans_args : None or dict, optional, default = None
            Additional information like instrument universe, industry
            classification etc. If provided, will be passed to fitness
            evaluation function. The method only check shape of these data,
            since it does not know what more to be expected.

        method : None or 'equal' or 'sharpe', optional, default = None
            Specifying method will override ``init`` args.
            - 'equal' forms a simple equal-weight portfolio.
            - 'sharpe' performs mean-varaince optimization to maximize 
              in-sample mean return scaled by standard deviation. Due to
              possible limitation on buy & sale, only positive weights are
              allowed.
              
        Returns
        -------
        self : object
            Returns self.
        '''
        def _parallel_wrapper(program, X, y, trans_args):
            program = cloudpickle.loads(program)
            pred = program.execute_transform(X, trans_args)
            program._pnl = np.nansum(pred * y, axis=1)
            return cloudpickle.dumps(program)
        
        def sharpe_target(w, ret, cov):
            return -w @ ret / np.sqrt(w @ cov @ w)
        
        if not isinstance(self.programs, Iterable) \
            or not all([isinstance(i, _Program) for i in self.programs]):
            raise ValueError('Invalid candidate program list.')
        method = method if method is not None else self.method
        if len(self.programs) == 1 or method == 'equal':
            self.weights = np.ones(len(self.programs), dtype=float) \
                / len(self.programs)
            return self
        if method != 'sharpe':
            raise ValueError('Only support method "equal" or "sharpe".')
        if not 0 < self.is_split <= 1:
            raise ValueError(f'is_split must in range (0., 1.], '
                             f'got {self.is_split}')
        has_pnl = all([hasattr(i, '_pnl') for i in self.programs])
        if not has_pnl and X is None:
            raise ValueError('Programs have no pre-computed PNL, '
                             'and X is not provided.')
            
        n_jobs = min(_get_n_jobs(self.n_jobs), len(self.programs))
        # Get programs with PNL data.
        if X is not None:
            X, y, _, trans_args, _, _ = \
                self.validate_data(X, y, None, trans_args, fit=True)
            programs = [cloudpickle.dumps(i) for i in self.programs]
            programs = Parallel(n_jobs=n_jobs, 
                                verbose=int(self.verbose>0) * 11)(
                delayed(_parallel_wrapper)(i, X, y, trans_args) 
                    for i in programs
            )
            programs = [cloudpickle.loads(i) for i in programs]
        else:
            programs = self.programs
        
        # Split IS / OS PNL and calculate IS stats accordingly.
        pnl = np.stack([i._pnl for i in programs], axis=1)
        is_split = int(self.is_split * pnl.shape[0])
        pnl_is = pnl[:is_split]
        ret = np.nanmean(pnl_is, axis=0)
        # Pandas implementation handles NaN efficiently.
        cov = pd.DataFrame(pnl_is).cov().to_numpy()
        # Set up boundary and constant limit, init starting guess.
        bounds = Bounds(0., 1.)
        constraints = LinearConstraint(np.ones_like(ret), 1., 1.)
        w0 = np.ones_like(ret) / len(ret)
        res = minimize(sharpe_target, w0, args=(ret, cov), 
                       constraints=constraints, bounds=bounds, 
                       method='trust-constr')
        self.weights = res.x
        return self
    
    def predict(
        self,
        X: Union[None, List[pd.DataFrame], np.ndarray], 
        trans_args: Union[None, Dict[str, np.ndarray]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Combine predictions from each program based on program weights.

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like, shape = (n_samples, n_features,
        n_firms)
            Training vectors, where n_samples is the number of trading days as
            samples, n_features is the number of features, n_firms is number of
            firms / stocks.  If provided as DataFrames, must have same index /
            columns, with shape (n_samples, n_firms).
            
        trans_args : optional, None or dict, default = None
            Additional information like instrument universe, industry
            classification etc. If provided, will be passed to weight transform
            function. The method only check shape of these data, since it does
            not know what more to be expected.

        Returns
        -------
        weight : pd.DataFrame or array-like, shape = (n_samples, n_firms)
            Returns weight. If X is list of DataFrames, return weight as
            DataFrame of same shape, index and columns.
        """
        def _parallel_wrapper(program, X, trans_args):
            program = cloudpickle.loads(program)
            weight = program.execute_transform(X, trans_args, force=True)
            return weight
        
        if not isinstance(self.programs, Iterable) \
            or not all([isinstance(i, _Program) for i in self.programs]):
            raise ValueError('Invalid candidate program list.')
        if self.weights is None:
            raise NotFittedError(f'{self.__class__.__name__} not fitted.')
        
        n_jobs = min(_get_n_jobs(self.n_jobs), len(self.programs))
        X, _, _, trans_args, idx, col = \
            self.validate_data(X, None, None, trans_args, fit=True)
        programs = [cloudpickle.dumps(i) for i in self.programs]
        programs = Parallel(n_jobs=n_jobs, 
                            verbose=int(self.verbose>0) * 11)(
            delayed(_parallel_wrapper)(i, X, trans_args) 
                for i in programs
        )
        stock_weights = sum([pos * wt 
                             for pos, wt in zip(programs, self.weights)])
        if idx is not None:
            stock_weights = pd.DataFrame(stock_weights, index=idx, columns=col)
        return stock_weights
    
    def _predcit_single(self, X, trans_args):
        if not isinstance(self.programs, Iterable) \
            or not all([isinstance(i, _Program) for i in self.programs]):
            raise ValueError('Invalid candidate program list.')
        if self.weights is None:
            raise NotFittedError(f'{self.__class__.__name__} not fitted.')
        
        X, _, _, trans_args, idx, col = \
            self.validate_data(X, None, None, trans_args, fit=True)
        stock_weights = np.zeros_like(X[:, 0], dtype=float)
        for program, wt in zip(self.programs, self.weights):
            pos = program.execute_transform(X, trans_args, force=True)
            stock_weights += pos * wt
        if idx is not None:
            stock_weights = pd.DataFrame(stock_weights, index=idx, columns=col)
        return stock_weights

    def fit_predict(
        self,
        method: str ='sharpe',
        X: Union[None, List[pd.DataFrame], np.ndarray] = None, 
        y: Union[None, np.ndarray] = None, 
        sample_weight: Any = None, 
        trans_args: Union[None, Dict[str, np.ndarray]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        '''Shortcut to fit and predict on same input. Not optimized. See
        ``fit`` for parameter definition.
        '''
        _ = self.fit(X, y, None, trans_args, method)
        return self.predict(X, trans_args)
    
    def save(self, path='./saved_portfolio.pkl'):
        '''Serialize the whole model and dump it to given path.
        
        Under the hood this just calls pickle to do the work. It is not
        suitable for long-term storage, and only guaranteed to work when
        de-serialized in the same python environment.
        
        Parameters
        --------
        path : str or pathlike, optional, default = './saved_model.pkl'
            Path to save the model. Expected to end with filename in extension
            type of .pkl, will be created if not exist.
        '''
        path = pathlib.Path(path)
        if not str(path).endswith('.pkl'):
            raise ValueError('path must be filename with .pkl extension.')
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path):
        '''Load the model file dumped by ``save``.
        
        Parameters
        --------
        path : str or pathlike
            Path to the saved model.

        Returns
        --------
        self : object
            Loaded model.
        '''
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, self.__class__):
            raise ValueError('Invalid model file.')
        self.__dict__.update(model.__dict__)
        return self  