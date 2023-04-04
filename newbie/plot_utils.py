#! /usr/bin/env python3
"""Utility functions for plotting Bayesian inference results"""

import matplotlib.pyplot as plt
import seaborn as sns


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