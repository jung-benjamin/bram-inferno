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

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from .estimators import EstimatorFactory
from .inferencedata import ClassificationResults


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

    def calc_hdi(self, *args, **kwargs):
        """Calculate highest density intervals for the data set.
        
        Multimodal HDIs are not supported. Calculating them throws
        an error that I am currentlly unable to solve. Without the
        ability to calculate multimodal HDIs, there is no need to
        support such cases.

        Parameters
        ----------
        args, kwargs
            Positional and keword arguments passed to arviz.hdi
        
        Returns
        -------
        hdi: pd.DataFrame
            DataFrame of HDI
        """
        hdi_dict = self.data.apply_func(az.hdi, *args, **kwargs)
        hdi = xr.concat(hdi_dict.values(),
                        dim=xr.DataArray(list(hdi_dict.keys()))).rename(
                            {'dim_0': 'ID'})
        return hdi

    def compare_hdi_prior(self, priors, *args, **kwargs):
        """Compare width of HDI to width of uniform priors.
        
        Parameters
        ----------
        priors : dict
            Dictionary mapping parameter labels to dictionaries
            that specify upper and lower bounds of uniform prior
            distribtions.
        args, kwargs
            Arguments for the `calc_hdi` method.

        Returns
        -------
        relative_span : xr.Dataset
            Length of the HDI vs the length of the uniform prior
            for each parameter in each element of the dataset.
        """
        hdi = self.calc_hdi(*args, **kwargs)
        if isinstance(priors, xr.Dataset):
            prior_lens = priors.sel(bound='upper') - priors.sel(bound='lower')
        else:
            prior_span = {
                n: it['upper'] - it['lower']
                for n, it in priors.items()
            }
            if all(
                    isinstance(d, ClassificationResults)
                    for d in self.data.data.values()):
                prior_lens = {}
                for n, idata in self.data.data.items():
                    p_l = {
                        k[:-1]: prior_span[k]
                        for k in list(idata.batch_posteriors[
                            idata.class_results])
                    }
                    prior_lens[n] = p_l
                prior_lens = xr.Dataset.from_dataframe(
                    pd.DataFrame(prior_lens).T).rename({'index': 'ID'})
            else:
                prior_lens = {n: prior_span for n in self.data.data}
                prior_lens = xr.Dataset.from_dataframe(
                    pd.DataFrame(prior_lens).T).rename({'index': 'ID'})
        hdi_span = hdi.sel(hdi='higher') - hdi.sel(hdi='lower')
        relative_span = hdi_span / prior_lens
        return relative_span

    def truth_in_hdi(self, *args, **kwargs):
        """Check if true parameters lie inside the HDI."""
        hdi = self.calc_hdi(*args, **kwargs)
        gt_lower = self.truth > hdi.sel(hdi='lower')
        lt_higher = self.truth < hdi.sel(hdi='higher')
        return gt_lower * lt_higher

    def calculate_distance(self,
                           estimator_type,
                           data_vars=None,
                           normalize=None,
                           absolute=False,
                           **kwargs):
        """Calculate distance between an estimator and the truth.
        
        Calculates absolute or relative distances between the prediction
        determined by an estimator and the true parameter. For a relative
        distance, the several normalization options are available.
        
        Parameters
        ----------
        estimator_type: str
            Select which type of estimator to use. Choose from 'mean',
            'mode' and 'peak'.
        data_vars: str or list of str (optional, default None)
            Select a subset of the data variables to operate on.
        normalize: str (optional, default None)
            Select how to normalize the distance. Choose from `None`,
            'truth', 'predicted', 'max', 'abssum'.
        absolute: bool (optional, default False)
            Set to true to calcuate the absolute value of the distance.
        kwargs
            Keyword arguments for the `calculate_estimator` method of the
            InferenceDataSet.
            
        Returns
        dist: xr.Dataset
            Dataset containing the relative distances.
        """
        if data_vars:
            truth = self.truth[data_vars]
        else:
            truth = self.truth
        predicted = self.data.calculate_estimator(estimator_type,
                                                  data_vars=data_vars,
                                                  **kwargs)
        dist = truth - predicted
        if absolute:
            dist = np.abs(dist)
        if normalize == 'truth':
            dist /= truth
        elif normalize == 'predicted':
            dist /= predicted
        elif normalize == 'max':
            dist /= max([truth, predicted])
        elif normalize == 'abssum':
            dist = 2 * dist / (np.abs(truth) + np.abs(predicted))
        return dist.expand_dims({'Norm': [normalize], 'Absolute': [absolute]})

    def distance_scan(self, estimator_types, normalize, absolute=[False]):
        """Calculate several different distance metrics
        
        Calculates the distance between prediction and truth for all
        combinations of the specified estimator types, normalization
        methods, etc.
        """
        est_list = []
        for est in estimator_types:
            norm_list = []
            for norm in normalize:
                abs_list = []
                for abs in sorted(set(absolute)):
                    d = self.calculate_distance(estimator_type=est,
                                                normalize=norm,
                                                absolute=abs)
                    abs_list.append(d)
                norm_list.append(xr.concat(abs_list, dim='Absolute'))
            est_list.append(xr.concat(norm_list, dim='Norm'))
        return xr.concat(est_list, dim='Estimator')
