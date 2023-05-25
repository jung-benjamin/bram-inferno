#! /usr/bin/env python3
"""Metrics for comparing the posteriors to the (synthetic) true values.

Requirements
- Read synthetic truth from file
- Get estimator
- Different handling of categorical inference of reactor type
"""

import logging
from abc import ABC, abstractmethod
from functools import cache

import numpy as np
import pandas as pd
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


class AccuracyMeasure(ABC):
    """Statistical measures for the accuracy of predictions."""

    @abstractmethod
    def calculate_accuracy(self, truth, prediction):
        pass

    def __call__(self, truth, prediction):
        return self.calculate_accuracy(truth, prediction)


class RSquaredMeasure(AccuracyMeasure):
    """R-squared as measure for prediction accuracy."""

    def calculate_accuracy(self, truth, prediction, dim='ID'):
        mean_truth = truth.mean(dim=dim)
        ss_total = ((truth - mean_truth)**2).sum(dim=dim)
        ss_residual = ((truth - prediction)**2).sum(dim=dim)
        r_squared = 1.0 - (ss_residual / ss_total)
        return r_squared


class MAPEMeasure(AccuracyMeasure):
    """Mean absolute percentage error"""

    def calculate_accuracy(self, truth, prediction, dim='ID'):
        absolute_errors = np.abs(truth - prediction)
        relative_errors = absolute_errors / np.maximum(np.abs(truth),
                                                       np.finfo(float).eps)
        mape = relative_errors.mean(dim=dim) * 100.0
        return mape


class RMSEMeasure(AccuracyMeasure):
    """Root mean squared error"""

    def calculate_accuracy(self, truth, prediction, dim='ID'):
        mse = ((truth - prediction)**2).mean(dim=dim)
        rmse = np.sqrt(mse)
        return rmse


class AccuracyMeasureFactory:
    """A factory class for accuracy measures."""

    def create_measure(measure_type):
        if measure_type == "r_squared":
            return RSquaredMeasure()
        elif measure_type == "mape":
            return MAPEMeasure()
        elif measure_type == "rmse":
            return RMSEMeasure()
        else:
            raise ValueError("Invalid measure type provided.")


class MetricDataSet:
    """Calculate Metrics for a set of related inference data."""

    def __init__(self, data_set, truth):
        """Initialize the class with inference data and true values.

        To-Do: add some checks so that the IDs in truth and data_set
        match.

        Parameters
        ----------
        data_set : InferenceDataSet
            Data set containing related inference data with associated
            IDs.
        truth : DataFrame or Dataset
            True parameter values associated with each item in the
            inference data set. If a dataframe is passed, the index
            name is set to 'ID'.
        """
        self.data = data_set
        if isinstance(truth, pd.DataFrame):
            truth.index.name = 'ID'
            self.truth = xr.Dataset(truth)
        self.truth = truth

    def get_measure(self, measure_type):
        """Wrapper for AccuracyMeasureFactory.create_measure"""
        return AccuracyMeasureFactory.create_measure(measure_type=measure_type)

    def calc_measure(self, measure_type, estimator_type, data_vars=None):
        """Calculate an accuracy measure for the inference data."""
        measure = self.get_measure(measure_type=measure_type)
        if data_vars:
            truth = self.truth[data_vars]
        else:
            truth = self.truth
        predicted = self.data.calculate_estimator(estimator_type,
                                                  data_vars=data_vars)
        return measure(truth=truth, prediction=predicted)
