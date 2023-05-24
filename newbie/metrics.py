#! /usr/bin/env python3
"""Metrics for comparing the posteriors to the (synthetic) true values.

Requirements
- Read synthetic truth from file
- Get estimator
- Different handling of categorical inference of reactor type
"""

import logging
from functools import cache

import numpy as np
import xarray as xr

from .estimators import EstimatorFactory


class Metric:

    def __init__(self, estimator, inference_data, truth, data_vars=None):
        self.estimator = EstimatorFactory.create_estimator(
            estimator, inference_data)
        self.data_vars = data_vars
        if self.data_vars:
            self.truth = truth[self.data_vars]
        else:
            self.truth = truth
        self.logger.debug(f'Truth vars: {list(self.truth)}')
        self.logger.debug(
            f'Idata vars: {list(inference_data["posterior"].data_vars)}')

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

    @cache
    def calculate_estimator(self, **kwargs):
        """Wrapper for estimator.calculate_estimator."""
        return self.estimator.calculate_estimator(self.data_vars, **kwargs)

    def calculate_distance(self, normalize=None, absolute=False, **kwargs):
        """Calculate distance between estimator and truth."""
        est = self.calculate_estimator(**kwargs)
        self.logger.debug(f'Estimator shape: {est.variables}')
        self.logger.debug(f'Truth shape: {self.truth}')
        dist = self.truth - est
        if absolute:
            dist = np.abs(dist)
        if normalize == 'truth':
            dist /= self.truth
        elif normalize == 'estimator':
            dist /= est
        elif normalize == 'max':
            dist /= max([self.truth, est])
        elif normalize == 'abssum':
            dist = 2 * dist / (np.abs(self.truth) + np.abs(est))
        elif normalize == 'range':
            if self.data_vars:
                range = (self.estimator.inference_data['posterior'][
                    self.data_vars].max() -
                         self.estimator.inference_data['posterior'][
                             self.data_vars].min())
            else:
                range = (self.estimator.inference_data.posterior.max() -
                         self.estimator.inference_data.posterior.min())
            dist /= np.abs(range)
        return dist

    def calculate_metric_norm(self, unit=True, **kwargs):
        """Calculate the euclidian norm of the distances."""
        dist = self.calculate_distance(**kwargs)
        d = dist.to_array(dim='data_vars')
        if unit:
            return np.linalg.norm(d) / len(d)
        return np.linalg.norm(d)


class MetricSet:

    def __init__(self,
                 inference_data,
                 truth,
                 estimators=['peak', 'mean', 'mode'],
                 data_vars=None):
        """Set inference data and truth objects."""
        self.metrics = {
            e: Metric(e, inference_data, truth, data_vars)
            for e in estimators
        }
        self.truth = truth

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

    def calculate_distances(self, **kwargs):
        """Calculate distances for selected estimators."""
        m = [it.calculate_distance(**kwargs) for n, it in self.metrics.items()]
        return xr.concat(m,
                         dim='Metric').assign_coords(Metric=list(self.metrics))

    def calculate_metric_norms(self, unit=True, **kwargs):
        """Calculate euclidian norms of the distances."""
        dist = self.calculate_distances(**kwargs)
        d = dist.to_array(dim='data_vars')
        if unit:
            metr = np.linalg.norm(d, axis=0) / d.shape[1]
        else:
            metr = np.linalg.norm(d, axis=1)
        return dict(zip(self.metrics, metr))