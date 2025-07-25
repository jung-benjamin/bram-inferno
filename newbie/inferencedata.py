#! /usr/bin/env python3
"""Classes for managing inference data

Mostly contains subclasses of arviz.InferenceData.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path

import arviz as az
import xarray as xr

from .estimators import EstimatorFactory


class InferenceData(az.InferenceData):
    """Extend functionality of arviz.InferenceData"""

    def __init__(self, **kwargs):
        """Initialize the class and its variables."""
        super().__init__(**kwargs)

    @classmethod
    def config_logger(cls,
                      loglevel='INFO',
                      logpath=None,
                      formatstr='%(levelname)s:%(name)s:%(message)s'):
        """Configure the logger."""
        log = logging.getLogger(cls.__name__)
        log.setLevel(getattr(logging, loglevel.upper()))
        log.handlers.clear()
        fmt = logging.Formatter(formatstr)
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logpath:
            fh = logging.FileHandler(logpath)
            fh.setLevel(getattr(logging, loglevel.upper()))
            fh.setFormatter(fmt)
            log.addHandler(fh)

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_json(cls, filepath):
        """Initialize the inference data from a json file."""
        data = az.from_json(filepath)
        instance = cls()
        instance.__dict__.update(data.__dict__.copy())
        return instance

    @classmethod
    def from_inferencedata(cls, inference_data):
        """Create class from a plain inference data object."""
        instance = cls()
        instance.__dict__.update(inference_data.__dict__.copy())
        return instance

    def _get_estimator(self, estimator_type):
        """Wrapper for EstimatorFactory.create_estimator"""
        return EstimatorFactory.create_estimator(estimator_type=estimator_type,
                                                 inference_data=self)

    def calculate_estimator(self, estimator_type, data_vars=None, **kwargs):
        """Calculate the estimator for all inference data."""
        estimator = self._get_estimator(estimator_type)
        self.estimator_values = estimator.calculate_estimator(
            data_vars=data_vars, **kwargs)
        return self.estimator_values

    def normalize_posterior_vars(self, data_vars):
        """Normalize a set of variables to sum to 1.
        
        Updates the posterior group of the InferenceData.
        """
        pos = self.posterior
        norm = sum([pos[n] for n in data_vars])
        for v in data_vars:
            pos[v] = pos[v] / norm


class ClassificationResults(InferenceData):
    """Results of Bayesian Inference for Reactor Type Classification"""

    def __init__(self, class_var, **kwargs):
        """Instantiate the classification results class."""
        super().__init__(**kwargs)
        self.reactor_map = {}
        self.class_var = class_var
        self.class_results = None
        self.class_posterior = []
        self.batch_posteriors = {}
        self._batch_map = {}

    @classmethod
    def from_json(cls, class_var, filepath):
        """Read classification inference data from a json file."""
        data = az.from_json(filepath)
        instance = cls(class_var)
        instance.__dict__.update(data.__dict__.copy())
        return instance

    @classmethod
    def from_inferencedata(cls, class_var, inference_data):
        """Create class from a plain inference data object."""
        instance = cls(class_var)
        instance.__dict__.update(inference_data.__dict__.copy())
        return instance

    @property
    def batch_map(self):
        """Return a mapping of batch IDs to sampled classes (integers)"""
        if not self._batch_map:
            return self._batch_map_from_mixing()
        else:
            return self._batch_map

    @property
    def inv_batch_map(self):
        """Invert map of batch IDs to sampled classes."""
        return {it: n for n, it in self.batch_map.items()}

    @property
    def reactor_map(self):
        """Return mapping of reactor name to number"""
        return self._reactor_map

    @reactor_map.setter
    def reactor_map(self, d):
        """Set the reactor map."""
        self._reactor_map = d

    @property
    def inv_reactor_map(self):
        """Mapping of number to reactor name"""
        return {it: n for n, it in self.reactor_map.items()}

    def _batch_map_from_mixing(self):
        """Map batch IDs to integers"""
        alpha_variables = sorted(
            set([v for v in self['posterior'].data_vars if 'alpha' in v]))
        # To-Do: find neater way to access the reactor ID
        self._batch_map = {v[-1]: i for i, v in enumerate(alpha_variables)}
        return self._batch_map

    def _calc_class_posterior(self, class_var=None):
        """Translate class posteriors."""
        if not class_var:
            class_var = self.class_var
        if self.reactor_map:
            use_map = self.inv_reactor_map
        else:
            use_map = self.inv_batch_map
        self.class_posterior = [
            use_map[i] for i in self['posterior'][class_var].values.flatten()
        ]
        return self.class_posterior

    def get_class_results(self, class_var=None, threshold=0.7):
        """Determine results of the classification."""
        if not class_var:
            class_var = self.class_var
        self._calc_class_posterior(class_var=class_var)
        counts = Counter(self.class_posterior)
        best = counts.most_common(1)
        if not ((best[0][1] / sum(counts.values())) > threshold):
            results = 'inconclusive'
        else:
            results = best[0][0]
        self.class_results = results
        return self.class_results

    def sort_posteriors_by_batch(self):
        """Sort posterior chains by their batch ID."""
        self.batch_posteriors = defaultdict(dict)
        for n in self.batch_map:
            for v, it in self['posterior'].data_vars.items():
                if 'alpha' in v:
                    pass
                elif v[-1] == n:
                    self.batch_posteriors[n][v] = it
        return self.batch_posteriors

    def hide_non_posteriors(self, **kwargs):
        """Drop posteriors not beloning to class result.
        
        Posteriors that do not belong the the label that is
        determined as the result of the classification part
        of the inference are dropped from the posterior and
        moved to hidden_posteriors.
        """
        if not self.batch_posteriors:
            self.sort_posteriors_by_batch()
        if not self.class_results:
            self.get_class_results(**kwargs)
        keep_vars = list(self.batch_posteriors[self.class_results])
        all_vars = self.posterior.data_vars
        self.hidden_posterior = self.posterior.copy()
        drop_vars = list(set(all_vars) - set(keep_vars))
        self.posterior = self.posterior.drop_vars(drop_vars)
        self.posterior = self.posterior.rename(
            dict(
                zip(keep_vars,
                    [k.strip(f'_{self.class_results}') for k in keep_vars])))


class InferenceDataSet:
    """Inference data that are related through some context."""
    name_format = 'drop_drop_reactor_id'

    def __init__(self, data):
        """Initialize the inference data set.
        
        Creates a dictionary of inference data and ids. If
        arviz.InferenceData are passed, these are converted
        to the subclass from this module.

        To-Do: Option to select subset of data or add keys
        for listed data via an optional variable.

        Parameters
        ----------
        data : dict, list or tuple
            A container of inference data. If the container does
            not provide keys, keys are created by enumeration.
        """
        try:
            self.data = dict(data)
        except ValueError:
            self.data = dict(enumerate(data))
        if not all([
                isinstance(d, (InferenceData, ClassificationResults))
                for d in self.data.values()
        ]):
            self.data = {
                n: InferenceData.from_inferencedata(it)
                for n, it in self.data.items()
            }
        self.posteriors = {n: it.posterior for n, it in self.data.items()}
        self.estimators = None

    @classmethod
    def config_logger(cls,
                      loglevel='INFO',
                      logpath=None,
                      formatstr='%(levelname)s:%(name)s:%(message)s'):
        """Configure the logger."""
        log = logging.getLogger(cls.__name__)
        log.setLevel(getattr(logging, loglevel.upper()))
        log.handlers.clear()
        fmt = logging.Formatter(formatstr)
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logpath:
            fh = logging.FileHandler(logpath)
            fh.setLevel(getattr(logging, loglevel.upper()))
            fh.setFormatter(fmt)
            log.addHandler(fh)

    @classmethod
    def from_json(cls, file_paths, fmt=None, class_var=None):
        """Load the inference data from a list of json files."""
        if class_var:
            idata = [
                ClassificationResults.from_json(class_var=class_var,
                                                filepath=f) for f in file_paths
            ]
        else:
            idata = [InferenceData.from_json(f) for f in file_paths]
        if callable(fmt):
            ids = [fmt(Path(f).stem) for f in file_paths]
        elif fmt:
            cls.set_name_format(fmt)
            ids = [cls.key_from_filename(Path(f).stem) for f in file_paths]
        else:
            ids = [cls.key_from_filename(Path(f).stem) for f in file_paths]
        return cls(dict(zip(ids, idata)))

    @classmethod
    def parse_filename(cls, name):
        """Extract data from the filenames given a format.

        An underscore is used to separate the format and the filename.
        """
        info = {}
        if cls.name_format == '':
            return {'filename': name}
        for i, j in zip(cls.name_format.split('_'), name.split('_')):
            if i == 'drop':
                pass
            else:
                info[i] = j
        return info

    @classmethod
    def key_from_filename(cls, fname):
        """Turn filename into a key."""
        info = cls.parse_filename(fname)
        return '_'.join(list(info.values()))

    @classmethod
    def set_name_format(cls, fmt):
        """Change the name format."""
        cls.name_format = fmt

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    def get_variables(self, var_name):
        """Return inference variables from all posteriors."""
        var_dict = {n: it[var_name] for n, it in self.posteriors.items()}
        return var_dict

    def calculate_estimator(self, estimator_type, data_vars=None, **kwargs):
        """Calculate estimator for all inference data in the set.
        
        Returns the values of the specified estimator type and sets the
        estimator values as the new self.estimators variable. If the
        variable exists, combines the datasets, overwriting previously
        calculated values of the same estimator.
        If the data are ClassificationResults, any elements with an
        'inconclusive' prediction are skipped.

        Warning: The exact behaviour of the concatenation if data_vars is
        specified is unkown.

        Parameters
        ----------
        estimator_type : str {'mean', 'mode', 'peak'}
            Choose which type of estimator use.
        data_vars: str or list of str
            Select a subset of data_variables of the inference_data.
        
        """
        est_dict = {}
        for n, idata in self.data.items():
            if isinstance(idata, ClassificationResults
                          ) and idata.class_results == 'inconclusive':
                continue
            est = idata.calculate_estimator(estimator_type=estimator_type,
                                            data_vars=data_vars,
                                            **kwargs)
            est_dict[n] = est.expand_dims({'ID': [n]})
        est_ds = xr.concat(est_dict.values(), dim='ID').expand_dims(
            {'Estimator': [estimator_type]})
        if not self.estimators:
            self.estimators = est_ds.copy()
        elif estimator_type in list(self.estimators['Estimator']):
            # Workaround, because update does not work as I expected.
            self.estimators = xr.concat([
                self.estimators.drop_sel(Estimator=estimator_type),
                est_ds.copy()
            ],
                                        dim='Estimator')
            # self.estimators.update(est_ds)
        else:
            self.estimators = xr.combine_by_coords(
                [self.estimators, est_ds.copy()])
        return est_ds

    def get_data_attributes_dict(self, attribute, *args, **kwargs):
        """Access methods and varables from data.
        
        Get a dictionary of attributes or methods evaluated at
        the args and kwargs for each item in the data.
        """
        attribute_dict = {}
        for n, idata in self.data.items():
            attribute_dict[n] = getattr(idata, attribute)
            try:
                attribute_dict[n] = attribute_dict[n](*args, **kwargs)
            except TypeError:
                pass
        return attribute_dict

    def apply_func(self, func, *args, **kwargs):
        """Apply a function to each element in the data set.
        
        Parameters
        ----------
        func: callable
            Function that accepts an InferenceData object as its
            first argument.
        args, kwargs
            Further arguments passed to the function.

        Returns
        -------
        results: dict
            Results of each function call associated with the respective
            ID of the inference data.
        """
        results = {
            n: func(idata, *args, **kwargs)
            for n, idata in self.data.items()
        }
        return results

    def drop_data_items(self, key, store=True):
        """Drop items from the data dictionary
        
        Specify which items to drop via the key. The
        dropped items can be stored to a new dictionary.

        Parameters
        ----------
        key: callable
            Returns True or False when applied to an item of
            the data dictionary. Items that return True are
            dropped.
        store: bool
            Set to true to store the dropped items as a
            dictionary.
        """
        drop = {}
        for n, it in self.data.items():
            if key(it):
                drop[n] = it
        for n in drop:
            self.data.pop(n)
        if store:
            self.dropped = drop
