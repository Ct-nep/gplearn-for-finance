"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import itertools
from abc import ABCMeta, abstractmethod
from collections import Iterable
from time import time
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from ._program import _Program
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators
from .utils import check_random_state

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape[:2]
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    max_samples = int(max_samples * n_samples)

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
        # Broadcast weight matrix to be compatible with y.
        if y.shape != curr_sample_weight.shape:
            err_msg = f'Cannot cast sample weight to the shape of y:'\
                f'{curr_sample_weight.shape} -> {y.shape}'
            diff = len(y.shape) - len(curr_sample_weight.shape)
            if diff > 0:
                # Assume later dims are missing.
                curr_sample_weight = np.expand_dims(
                    curr_sample_weight,
                    axis=np.arange(diff)+len(y.shape),
                )
                try:
                    curr_sample_weight = np.broadcast_to(curr_sample_weight, 
                                                        y.shape).copy()
                except Exception:
                    raise RuntimeError(err_msg)
            else:
                raise RuntimeError(err_msg)
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs


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
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 sample_days=None,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        if sample_days is not None and \
            (not isinstance(sample_days, int) or sample_days <= 0):
            raise ValueError('sample_days must be positive integer.')
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
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.sample_days = sample_days
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))
    
    def _validate_data(self, X, y, sample_weight=None, fit=False):
        '''Patch parent's _validate_data() method for n-dim situation.
        Due to drastic change of behavior, it is no long suitable to
        rely on sklearn's validate method. By default, if inputs are
        missing n_samples dim, a newaxis will be added.
        
        Support inputs of multiple DataFrame as X and single DataFrame
        as y. If this is the case, convert them to ndarray and preserve
        indices.
        
        If fit is set to True and self.sample_days is a positive
        integer, the input will be padded and reshaped so n_days ==
        sample_days. This is recommend for calling fit() method but
        unnecessary for predict().
        
        Parameters
        ----------
        X : List[pd.DataFrame] or array-like of shape (n_features,
        n_days, n_firms) or (n_samples, n_features, n_days, n_firms)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features, n_days is sequence
            length of each sample, n_firms is number of firms / stocks.
            If provided as DataFrames, must have same index / columns,
            with shape (n_days, n_firms).

        y : None, pd.DataFrame or array-like of shape (n_days, n_firms)
        or (n_samples, n_days, n_firms)
            Target values, if provided as DataFrame, must have same
            index / columns with all X DataFrame, with shape (n_days,
            n_firms).

        sample_weight : optional, None or pd.DataFrame or array-like of
        shape (n_samples,) or (n_samples, n_days, n_firms) default=None
            Weights applied to individual samples.
            
        fit : bool, default=False
            If set to True and self.sample_days is a positive integer,
            input data will be padded and reshaped to formulate
            minibatchs so n_samples >= 1 and n_days == sample_days.

        Returns
        -------
        X : array-like of shape (n_samples, n_features, n_days, n_firms)
            X converted to compatible shape.
            
        y : None or array-like of shape (n_samples, n_days, n_firms)
            y converted to compatible shape.
            
        sample_weight : array-like of shape (n_samples, n_days, n_firms)
            sample_weight converted to compatible shape. If not
            provided, will be np.ones_like(y, dtype=float).
            
        index: Union[int, pd.Index]
            If X, y are DataFrames, preserve index, otherwise is
            length of original panel data.
            
        columns: Union[int, pd.Index]
            If X, y are DataFrames, preserve columns, otherwise is
            width of original panel data.
        '''
        idx, col = None, None
        if y is None and sample_weight is not None:
            raise ValueError('Input y is missing.')
        # Convert DataFrames to ndarray
        if (isinstance(y, pd.DataFrame) or y is None) \
            and isinstance(X, Iterable) \
            and all([isinstance(i, pd.DataFrame) for i in X]):
            warn(f'Received DataFrames as input, try converting.')
            if y is not None:
                size, idx, col = y.shape, y.index, y.columns
            else:
                size, idx, col = X[0].shape, X[0].index, X[0].columns
            if not all([i.shape==size and (i.index==idx).all() \
                        and (i.columns==col).all() for i in X]):
                raise ValueError('All X DataFrames and y must have same '
                                 'shape, indices and columns')
            X = np.stack([i.to_numpy() for i in X], axis=0)[None, ...]
            if y is not None:
                y = y.to_numpy()[None, ...]
            if sample_weight is not None:
                if not isinstance(sample_weight, pd.DataFrame) \
                    or sample_weight.shape != size \
                    or sample_weight.index != idx \
                    or sample_weight.columns != col:
                    raise ValueError('Sample weight must also be DataFrame and'
                                     ' with same shape, index and columns.')
                sample_weight = sample_weight.to_numpy()[None, ...]
        if not isinstance(X, np.ndarray) \
            or not (isinstance(y, np.ndarray) or y is None):
            raise ValueError(f'Cannot coerce X, y with type {type(X)}, '
                            f'{type(y)} to compatible data structure.')
        if idx is None:
            idx, col = X.shape[2], X.shape[3]
        # Ensure numeric inputs except for classifier's y
        X = super()._validate_data(X, force_all_finite='allow-nan', 
                                   reset=fit, ensure_2d=False, 
                                   allow_nd=True, dtype='numeric')
        super()._check_n_features(X, reset=fit)
        if y is not None:
            y_dtype = None if isinstance(y, ClassifierMixin) else 'numeric'
            y = check_array(y, force_all_finite='allow-nan',
                            ensure_2d=False, allow_nd=True, dtype=y_dtype)
        msg = f'Received X, y with shape {X.shape}, '\
        f'{"" if y is None else y.shape}, assume n_samples dim is missing.'
        if len(X) == 3 or (y is not None and len(y) == 2):
            warn(msg)
            if len(X) == 3:  X = X[None, ...]
        if y is not None:
            if len(y) == 2: y = y[None, ...]
            X_shape, y_shape = list(X.shape), list(y.shape)
            msg = f'X, y have incorrect shape {X.shape}, {y.shape}'
            if len(X_shape) > 1: 
                del X_shape[1]
            else:
                raise ValueError(msg)
            if (not len(X_shape) == len(y_shape) == 3) \
                or X_shape != y_shape:
                raise ValueError(msg)
        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                raise ValueError('Sample weight must also be ndarray.')
            if len(sample_weight.shape) == 1 \
                and sample_weight.shape[0] == y.shape[0]:
                sample_weight = sample_weight[:, None, None]  
            elif len(sample_weight.shape) == 2:
                sample_weight = sample_weight[None, ...]
            try:
                sample_weight = np.broadcast_to(sample_weight[:, None, None], 
                                                y.shape)
            except Exception as e:
                raise ValueError(f'Cannot coerce shape of sample ' 
                                 f'weight with y: {e}')
        # Check class label, convert all inputs to float if possible
        # since it is expected to have nans.
        if y is not None:
            if isinstance(self, ClassifierMixin):
                if np.issubdtype(y.dtype, float):
                    mask = np.isfinite(y)
                    if not (y[mask] == y[mask].astype(int)).all():
                        raise ValueError('y cannot be continuous float.')
                elif np.issubdtype(y.dtype, int):
                    y = y.astype(float)
                elif not np.issubdtype(y.dtype, str):
                    raise ValueError(f'Require label y to be int, string or '
                                     f'int-like float, received {y.dtype}.')
            elif not np.issubdtype(y.dtype, float):
                y = y.astype(float)
        if not np.issubdtype(X.dtype, float): 
            X = X.astype(float)
        if sample_weight is None:
            if y is not None:
                sample_weight = np.ones_like(y, dtype=float)
        elif not np.issubdtype(sample_weight.dtype, float):
            sample_weight = sample_weight.astype(float)
        # Reshape inputs to satisfy self.sample_days
        if fit:
            X, y, sample_weight = self._pad_and_reshape(X, y, sample_weight)
        return X, y, sample_weight, idx, col
    
    def _pad_and_reshape(self, X, y, sample_weight):
        '''If self.sample_days is specified, this method will pad and
        reshape inputs so n_days == sample_days.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_days, n_firms)

        y : None or array-like of shape (n_samples, n_days, n_firms)

        sample_weight : None or array-like of shape 
        (n_samples, n_days, n_firms)

        Returns
        -------
        X : array-like of shape (n_samples, n_features, sample_days, n_firms)

        y : None or array-like of shape (n_samples, sample_days, n_firms)

        sample_weight : None or array-like of shape 
        (n_samples, sample_days, n_firms)
        '''
        if self.sample_days is not None and self.sample_days != X.shape[2]:
            n_samples, n_days, n_firms = X.shape[0], X.shape[2], X.shape[3]
            pad = (n_samples*n_days) % self.sample_days
            fn = lambda arr: \
                np.pad(
                    arr.reshape(-1, n_firms), 
                    ((0, pad), (0, 0)), 
                    'constant', 
                    constant_values=np.str_(np.nan) \
                        if np.issubdtype(arr.dtype, str) else np.nan
                ).reshape(-1, self.sample_days, n_firms)
            if y is not None: y = fn(y)
            if sample_weight is not None: sample_weight = fn(sample_weight)
            X = np.stack([fn(X[:, i]) for i in range(X.shape[1])], axis=1)
        return X, y, sample_weight

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        X, y, sample_weight, _, _ = \
            self._validate_data(X, y, sample_weight, fit=True)
        if isinstance(self, ClassifierMixin):
            n_samples, n_days, n_firms = y.shape
            if self.class_weight:
                # modify the sample weights with the corresponding class weight
                sample_weight = sample_weight.reshape(-1,)
                sample_weight = (sample_weight *
                                 compute_sample_weight(self.class_weight, 
                                                       y.reshape(-1,)))
                sample_weight = sample_weight.reshape(n_samples, n_days, n_firms)

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            y = y.reshape(n_samples, n_days, n_firms).astype(float)
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

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

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

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

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

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

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
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

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
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
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

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

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
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
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

    metric : str, optional (default='mean absolute error')
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

    parsimony_coefficient : float or "auto", optional (default=0.001)
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

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
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
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

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
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 sample_days=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
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
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            sample_days=sample_days,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X, transform=True, return_2d=False):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like of shape (n_features,
        n_days, n_firms) or (n_samples, n_features, n_days, n_firms)
            Input vectors, where n_samples is the number of samples,
            n_features is the number of features, n_days is sequence
            length of each sample, n_firms is number of firms / stocks.
            If provided as DataFrames, must have same index / columns, 
            with shape (n_days, n_firms).

        transform : bool, default=True
            If set to True and have attribute self._transformer, will
            use it to transform result.
            
        return_2d : bool, default=False
            If set to True, will flatten n_samples dim and reshape to
            (n_samples*n_days, n_firms) with padded rows removed. If X
            is provided as DataFrames, also keep original index and
            columns.

        Returns
        -------
        y : pd.DataFrame or array-like of shape (n_days, n_firms)
        or (n_samples, n_days, n_firms)
            Target values, shape depends on return_2d argument.
        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        # X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        X, _, _, ind, col = self._validate_data(X, None, None)
        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)
        if transform and getattr(self, '_transformer', None) is not None:
            y = self._transformer(y)
        if return_2d:
            if isinstance(ind, int):
                y = y[0]
            elif isinstance(ind, pd.Index):
                y = pd.DataFrame(y[0], index=ind, columns=col)
            else:
                raise ValueError(f'Unsurpported index type ' 
                                 f'{type(ind)}, {type(col)}')
        return y


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):

    """A Genetic Programming symbolic classifier.

    A symbolic classifier is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=500)
        The number of programs in each generation.

    generations : integer, optional (default=10)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
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

    transformer : str, optional (default='sigmoid')
        The name of the function through which the raw decision function is
        passed. This function will transform the raw decision function into
        probabilities of each class.

        This can also be replaced by your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

    metric : str, optional (default='log loss')
        The name of the raw fitness metric. Available options include:

        - 'log loss' aka binary cross-entropy loss.

    parsimony_coefficient : float or "auto", optional (default=0.001)
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

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    class_weight : dict, 'balanced' or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
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
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

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
                 transformer='sigmoid',
                 metric='log loss',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicClassifier, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            class_weight=class_weight,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def _more_tags(self):
        return {'binary_only': True}

    def predict_proba(self, X):
        """Predict probabilities on vectors X. 

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like of shape (n_features,
        n_days, n_firms) or (n_samples, n_features, n_days, n_firms)
            Input vectors, where n_samples is the number of samples,
            n_features is the number of features, n_days is sequence
            length of each sample, n_firms is number of firms / stocks.
            If provided as DataFrames, must have same index / columns, 
            with shape (n_days, n_firms).

        Returns
        -------
        y_prob : array-like of shape (n_samples, n_days, n_firms, n_classes)
            Target values, shape depends on return_2d argument.
        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X, _, _, _, _ = self._validate_data(X, None, None)
        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)
        if getattr(self, '_transformer', None) is None:
            warn('No transformer for result, use raw predict as-is.')
        else:
            y = self._transformer(y)
        y = np.stack([1.-y, y], axis=-1)
        return y

    def predict(self, X, return_2d=False):
        """Predict classes on test vectors X. If return_2d is set
        to True, return predict result in 2d array or DataFrame.

        Parameters
        ----------
        X : List[pd.DataFrame] or array-like of shape (n_features,
        n_days, n_firms) or (n_samples, n_features, n_days, n_firms)
            Input vectors, where n_samples is the number of samples,
            n_features is the number of features, n_days is sequence
            length of each sample, n_firms is number of firms / stocks.
            If provided as DataFrames, must have same index / columns, 
            with shape (n_days, n_firms).
            
        return_2d : bool, default=False
            If set to True, will flatten n_samples dim and reshape to
            (n_samples*n_days, n_firms) with padded rows removed. If X
            is provided as DataFrames, also keep original index and
            columns. Finally, only probability for positive class is
            returned.

        Returns
        -------
        y : pd.DataFrame or array-like of shape (n_days, n_firms)
        or (n_samples, n_days, n_firms)
            Target values, shape depends on return_2d argument.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X, _, _, ind, col = self._validate_data(X, None, None)
        n_features = X.shape[1]
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)
        if getattr(self, '_transformer', None) is None:
            raise ValueError('Cannot find transformer for raw predict.')
        y = self._transformer(y)
        mask = np.isfinite(y)
        y = np.where(mask, y, 0.)
        y = np.stack([1.-y, y], axis=-1)
        y = self.classes_.take(np.argmax(y, axis=-1))
        missing = np.str_(np.nan) \
            if np.issubdtype(self.classes_.dtype, str) else np.nan
        y = np.where(mask, y, missing)
        if return_2d:
            if isinstance(ind, int):
                y = y[0]
            elif isinstance(ind, pd.Index):
                y = pd.DataFrame(y[0], index=ind, columns=col)
            else:
                raise ValueError(f'Unsurpported index type '
                                   f'{type(ind)}, {type(col)}')
        return y


# class SymbolicTransformer(BaseSymbolic, TransformerMixin):

#     """A Genetic Programming symbolic transformer.

#     A symbolic transformer is a supervised transformer that begins by building
#     a population of naive random formulas to represent a relationship. The
#     formulas are represented as tree-like structures with mathematical
#     functions being recursively applied to variables and constants. Each
#     successive generation of programs is then evolved from the one that came
#     before it by selecting the fittest individuals from the population to
#     undergo genetic operations such as crossover, mutation or reproduction.
#     The final population is searched for the fittest individuals with the least
#     correlation to one another.

#     Parameters
#     ----------
#     population_size : integer, optional (default=1000)
#         The number of programs in each generation.

#     hall_of_fame : integer, or None, optional (default=100)
#         The number of fittest programs to compare from when finding the
#         least-correlated individuals for the n_components. If `None`, the
#         entire final generation will be used.

#     n_components : integer, or None, optional (default=10)
#         The number of best programs to return after searching the hall_of_fame
#         for the least-correlated individuals. If `None`, the entire
#         hall_of_fame will be used.

#     generations : integer, optional (default=20)
#         The number of generations to evolve.

#     tournament_size : integer, optional (default=20)
#         The number of programs that will compete to become part of the next
#         generation.

#     stopping_criteria : float, optional (default=1.0)
#         The required metric value required in order to stop evolution early.

#     const_range : tuple of two floats, or None, optional (default=(-1., 1.))
#         The range of constants to include in the formulas. If None then no
#         constants will be included in the candidate programs.

#     init_depth : tuple of two ints, optional (default=(2, 6))
#         The range of tree depths for the initial population of naive formulas.
#         Individual trees will randomly choose a maximum depth from this range.
#         When combined with `init_method='half and half'` this yields the well-
#         known 'ramped half and half' initialization method.

#     init_method : str, optional (default='half and half')
#         - 'grow' : Nodes are chosen at random from both functions and
#           terminals, allowing for smaller trees than `init_depth` allows. Tends
#           to grow asymmetrical trees.
#         - 'full' : Functions are chosen until the `init_depth` is reached, and
#           then terminals are selected. Tends to grow 'bushy' trees.
#         - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
#           'grow', making for a mix of tree shapes in the initial population.

#     function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
#         The functions to use when building and evolving programs. This iterable
#         can include strings to indicate either individual functions as outlined
#         below, or you can also include your own functions as built using the
#         ``make_function`` factory from the ``functions`` module.

#         Available individual functions are:

#         - 'add' : addition, arity=2.
#         - 'sub' : subtraction, arity=2.
#         - 'mul' : multiplication, arity=2.
#         - 'div' : protected division where a denominator near-zero returns 1.,
#           arity=2.
#         - 'sqrt' : protected square root where the absolute value of the
#           argument is used, arity=1.
#         - 'log' : protected log where the absolute value of the argument is
#           used and a near-zero argument returns 0., arity=1.
#         - 'abs' : absolute value, arity=1.
#         - 'neg' : negative, arity=1.
#         - 'inv' : protected inverse where a near-zero argument returns 0.,
#           arity=1.
#         - 'max' : maximum, arity=2.
#         - 'min' : minimum, arity=2.
#         - 'sin' : sine (radians), arity=1.
#         - 'cos' : cosine (radians), arity=1.
#         - 'tan' : tangent (radians), arity=1.

#     metric : str, optional (default='pearson')
#         The name of the raw fitness metric. Available options include:

#         - 'pearson', for Pearson's product-moment correlation coefficient.
#         - 'spearman' for Spearman's rank-order correlation coefficient.

#     parsimony_coefficient : float or "auto", optional (default=0.001)
#         This constant penalizes large programs by adjusting their fitness to
#         be less favorable for selection. Larger values penalize the program
#         more which can control the phenomenon known as 'bloat'. Bloat is when
#         evolution is increasing the size of programs without a significant
#         increase in fitness, which is costly for computation time and makes for
#         a less understandable final result. This parameter may need to be tuned
#         over successive runs.

#         If "auto" the parsimony coefficient is recalculated for each generation
#         using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
#         program size l and program fitness f in the population, and Var(l) is
#         the variance of program sizes.

#     p_crossover : float, optional (default=0.9)
#         The probability of performing crossover on a tournament winner.
#         Crossover takes the winner of a tournament and selects a random subtree
#         from it to be replaced. A second tournament is performed to find a
#         donor. The donor also has a subtree selected at random and this is
#         inserted into the original parent to form an offspring in the next
#         generation.

#     p_subtree_mutation : float, optional (default=0.01)
#         The probability of performing subtree mutation on a tournament winner.
#         Subtree mutation takes the winner of a tournament and selects a random
#         subtree from it to be replaced. A donor subtree is generated at random
#         and this is inserted into the original parent to form an offspring in
#         the next generation.

#     p_hoist_mutation : float, optional (default=0.01)
#         The probability of performing hoist mutation on a tournament winner.
#         Hoist mutation takes the winner of a tournament and selects a random
#         subtree from it. A random subtree of that subtree is then selected
#         and this is 'hoisted' into the original subtrees location to form an
#         offspring in the next generation. This method helps to control bloat.

#     p_point_mutation : float, optional (default=0.01)
#         The probability of performing point mutation on a tournament winner.
#         Point mutation takes the winner of a tournament and selects random
#         nodes from it to be replaced. Terminals are replaced by other terminals
#         and functions are replaced by other functions that require the same
#         number of arguments as the original node. The resulting tree forms an
#         offspring in the next generation.

#         Note : The above genetic operation probabilities must sum to less than
#         one. The balance of probability is assigned to 'reproduction', where a
#         tournament winner is cloned and enters the next generation unmodified.

#     p_point_replace : float, optional (default=0.05)
#         For point mutation only, the probability that any given node will be
#         mutated.

#     max_samples : float, optional (default=1.0)
#         The fraction of samples to draw from X to evaluate each program on.

#     feature_names : list, optional (default=None)
#         Optional list of feature names, used purely for representations in
#         the `print` operation or `export_graphviz`. If None, then X0, X1, etc
#         will be used for representations.

#     warm_start : bool, optional (default=False)
#         When set to ``True``, reuse the solution of the previous call to fit
#         and add more generations to the evolution, otherwise, just fit a new
#         evolution.

#     low_memory : bool, optional (default=False)
#         When set to ``True``, only the current generation is retained. Parent
#         information is discarded. For very large populations or runs with many
#         generations, this can result in substantial memory use reduction.

#     n_jobs : integer, optional (default=1)
#         The number of jobs to run in parallel for `fit`. If -1, then the number
#         of jobs is set to the number of cores.

#     verbose : int, optional (default=0)
#         Controls the verbosity of the evolution building process.

#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.

#     Attributes
#     ----------
#     run_details_ : dict
#         Details of the evolution process. Includes the following elements:

#         - 'generation' : The generation index.
#         - 'average_length' : The average program length of the generation.
#         - 'average_fitness' : The average program fitness of the generation.
#         - 'best_length' : The length of the best program in the generation.
#         - 'best_fitness' : The fitness of the best program in the generation.
#         - 'best_oob_fitness' : The out of bag fitness of the best program in
#           the generation (requires `max_samples` < 1.0).
#         - 'generation_time' : The time it took for the generation to evolve.

#     See Also
#     --------
#     SymbolicRegressor

#     References
#     ----------
#     .. [1] J. Koza, "Genetic Programming", 1992.

#     .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

#     """

#     def __init__(self,
#                  *,
#                  population_size=1000,
#                  hall_of_fame=100,
#                  n_components=10,
#                  generations=20,
#                  tournament_size=20,
#                  stopping_criteria=1.0,
#                  const_range=(-1., 1.),
#                  init_depth=(2, 6),
#                  init_method='half and half',
#                  function_set=('add', 'sub', 'mul', 'div'),
#                  metric='pearson',
#                  parsimony_coefficient=0.001,
#                  p_crossover=0.9,
#                  p_subtree_mutation=0.01,
#                  p_hoist_mutation=0.01,
#                  p_point_mutation=0.01,
#                  p_point_replace=0.05,
#                  max_samples=1.0,
#                  feature_names=None,
#                  warm_start=False,
#                  low_memory=False,
#                  n_jobs=1,
#                  verbose=0,
#                  random_state=None):
#         super(SymbolicTransformer, self).__init__(
#             population_size=population_size,
#             hall_of_fame=hall_of_fame,
#             n_components=n_components,
#             generations=generations,
#             tournament_size=tournament_size,
#             stopping_criteria=stopping_criteria,
#             const_range=const_range,
#             init_depth=init_depth,
#             init_method=init_method,
#             function_set=function_set,
#             metric=metric,
#             parsimony_coefficient=parsimony_coefficient,
#             p_crossover=p_crossover,
#             p_subtree_mutation=p_subtree_mutation,
#             p_hoist_mutation=p_hoist_mutation,
#             p_point_mutation=p_point_mutation,
#             p_point_replace=p_point_replace,
#             max_samples=max_samples,
#             feature_names=feature_names,
#             warm_start=warm_start,
#             low_memory=low_memory,
#             n_jobs=n_jobs,
#             verbose=verbose,
#             random_state=random_state)

#     def __len__(self):
#         """Overloads `len` output to be the number of fitted components."""
#         if not hasattr(self, '_best_programs'):
#             return 0
#         return self.n_components

#     def __getitem__(self, item):
#         """Return the ith item of the fitted components."""
#         if item >= len(self):
#             raise IndexError
#         return self._best_programs[item]

#     def __str__(self):
#         """Overloads `print` output of the object to resemble LISP trees."""
#         if not hasattr(self, '_best_programs'):
#             return self.__repr__()
#         output = str([gp.__str__() for gp in self])
#         return output.replace("',", ",\n").replace("'", "")

#     def _more_tags(self):
#         return {
#             "_xfail_checks": {
#                 "check_sample_weights_invariance": (
#                     "zero sample_weight is not equivalent to removing samples"
#                 ),
#             }
#         }

#     def transform(self, X):
#         """Transform X according to the fitted transformer.

#         Parameters
#         ----------
#         X : array-like, shape = [n_samples, n_features]
#             Input vectors, where n_samples is the number of samples
#             and n_features is the number of features.

#         Returns
#         -------
#         X_new : array-like, shape = [n_samples, n_components]
#             Transformed array.

#         """
#         if not hasattr(self, '_best_programs'):
#             raise NotFittedError('SymbolicTransformer not fitted.')
#         X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
#         _, n_features = X.shape[:2]
#         if self.n_features_in_ != n_features:
#             raise ValueError('Number of features of the model must match the '
#                              'input. Model n_features is %s and input '
#                              'n_features is %s.'
#                              % (self.n_features_in_, n_features))

#         X_new = np.array([gp.execute(X) for gp in self._best_programs]).T

#         return X_new

#     def fit_transform(self, X, y, sample_weight=None):
#         """Fit to data, then transform it.

#         Parameters
#         ----------
#         X : array-like, shape = [n_samples, n_features]
#             Training vectors, where n_samples is the number of samples and
#             n_features is the number of features.

#         y : array-like, shape = [n_samples]
#             Target values.

#         sample_weight : array-like, shape = [n_samples], optional
#             Weights applied to individual samples.

#         Returns
#         -------
#         X_new : array-like, shape = [n_samples, n_components]
#             Transformed array.

#         """
#         return self.fit(X, y, sample_weight).transform(X)
