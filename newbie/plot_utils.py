#! /usr/bin/env python3
"""Utility functions for plotting Bayesian inference results"""

import logging
import re

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .estimators import EstimatorFactory
from .inferencedata import ClassificationResults

PARAM_REGEX = re.compile('([a-zA-Z])([a-z]+)(\d|[A-Z]?)')


class DistancePlot:

    def __init__(self, distances):
        """Instantiate class with xarray dataset of distances."""
        self.distances = distances
    
    def get_subplots(self, metrics='all', **kwargs):
        """Create figure and axes with subplots."""
        if metrics == 'all':
            cols = len(self.distances.Metric)
        elif isinstance(metrics, list):
            cols = len(metrics)
        else:
            msg = 'Metrics must be "all" or a list.'
        rows = len(self.distances.data_vars)
        fig, ax = plt.subplots(nrows=rows, ncols=cols, **kwargs)
        return fig, ax
    
    def plot_iterator(self, axes):
        """Iterate over the plot axes and the data."""
        for var, row in zip(self.distances.data_vars, axes):
            data = self.distances[var].to_pandas()
            for m, col in zip(data.columns, row):
                yield data, m, col
    
    def plot_distances(self):
        """Plot distances between estimators and true values."""
        raise NotImplementedError(
            'Subclasses of DistancePlot must implement a plot_distances method'
        )


class DistanceHistogram(DistancePlot):

    def plot_distances(self, **kwargs):
        """Histogram the distances between inference and truth."""
        fig, axes = self.get_subplots(**kwargs)
        for data, m, col in self.plot_iterator(axes):
            sns.histplot(data=data, x=m, ax=col)
        return fig, axes


class DistanceScatter(DistancePlot):

    def plot_distances(self, **kwargs):
        """Create a scatterplot of the distances."""
        fig, axes = self.get_subplots(**kwargs)
        for data, m, col in self.plot_iterator(axes):
            sns.scatterplot(data=data, x='ID', y=m, ax=col)
        return fig, axes


def get_param_batch_id(data_var):
    """Extract the batch id of an inference variable."""
    c = PARAM_REGEX.fullmatch(data_var)
    if c:
        batch_id = c.group(3)
        return batch_id


def strip_param_batch_id(data_var):
    """Remove the batch ID from parameter name."""
    c = PARAM_REGEX.fullmatch(data_var)
    if c:
        return ''.join([c.group(1), c.group(2)])
    else:
        return data_var


def group_batch_parameters(data_variables):
    """Group inference variables by their batch id."""
    batch_params = {}
    skipped_params = []
    for n in data_variables:
        batch_id = get_param_batch_id(n)
        if batch_id:
            try:
                batch_params[batch_id].extend([n])
            except KeyError:
                batch_params[batch_id] = [n]
        else:
            skipped_params.append(n)
    return {n: sorted(it) for n, it in batch_params.items()}, skipped_params


class PosteriorPlot:
    """Plot Bayesian inference posteriors."""

    def __init__(self, inference_data):
        """Instantiate the plotting class with inference results."""
        self.inference_results = inference_data
        self.figure = None
        self.ax_dict = None
        self.truth = {}
        self.colors = plt.cm.tab10
        self.estimators = {}

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

    def plot_posteriors(self, var_names=None, **plot_kw):
        """Plot posterior distributions."""
        if not var_names:
            var_names = list(self.inference_results.posterior.data_vars)
        batch_params, others = group_batch_parameters(var_names)
        self.logger.debug(f'Batch parameters: {batch_params}')
        self.logger.debug(f'Other parameters: {others}')
        nrows = len(batch_params)
        ncols = max([len(it) for n, it in batch_params.items()])
        self.fig, axes = plt.subplots(ncols=ncols, nrows=nrows, **plot_kw)
        if len(axes.shape) == 1:
            axes = axes[np.newaxis, :]
        self.ax_dict = {}
        for i, (n, it) in enumerate(batch_params.items()):
            az.plot_posterior(self.inference_results,
                              var_names=it,
                              ax=axes[i, :],
                              point_estimate=None,
                              hdi_prob='hide')
            for i, a in zip(it, axes[i, :]):
                self.ax_dict[i] = a
        return self.fig, self.ax_dict

    def plot_truth(self, truth=None):
        """Add vertical lines indicating the expected parameter values."""
        if truth is None:
            truth = self.truth
        elif not dict(truth):
            truth = self.truth
        if not dict(truth):
            msg = f'No true parameter values to plot.'
            self.logger.warn(msg)
            return self.ax_dict
        for n, ax in self.ax_dict.items():
            try:
                ax.axvline(truth[n], ls='dashed')
            except KeyError:
                try:
                    ax.axvline(truth[strip_param_batch_id(n)], ls='dashed')
                except KeyError:
                    msg = f'No synthetic truth found for {n}'
                    self.logger.warn(msg)
        return self.ax_dict

    def calc_estimators(self, estimators):
        """Calculate estimators for the posteriors."""
        if isinstance(estimators, str):
            estimators = [estimators]
        est_dict = {}
        for e in estimators:
            est = EstimatorFactory.create_estimator(e, self.inference_results)
            est_dict[e] = est.calculate_estimator()
        self.estimators.update(est_dict)
        return est_dict

    def _plot_estimator(self, estimator, color, label):
        """Add vertical lines of a posterior estimator to the graphs."""
        for n, ax in self.ax_dict.items():
            e = estimator[n]
            ax.axvline(e, ls='dotted', color=color, label=label)

    def plot_estimators(self, est_dict=None):
        """Add vertical lines indicating posterior estimators.
        
        To-Do: Add customizability for colors and labels
        """
        if not est_dict:
            est_dict = self.estimators
        if not est_dict:
            msg = f'No estimators to plot.'
            self.logger.warn(msg)
            return self.ax_dict
        for i, (n, e) in enumerate(est_dict.items()):
            self._plot_estimator(e, color=self.colors(i), label=n)
        return self.ax_dict

    def plot(self,
             var_names=None,
             truth=None,
             estimators=None,
             save=None,
             show=True,
             **plot_kw):
        """Plot posterior distributions in the inference data.
        
        To-Do: 
            Proper legend creation. Consider forcing a very recent
            version of matplotlib to leverage the new legend
            syntax for figures.
        """
        self.plot_posteriors(var_names, **plot_kw)
        self.plot_truth(truth=truth)
        if isinstance(estimators, (list, str)):
            self.calc_estimators(estimators=estimators)
            self.plot_estimators()
        else:
            self.plot_estimators(estimators)
        if save:
            plt.savefig(save)
        if show:
            plt.show()
        else:
            plt.close()

