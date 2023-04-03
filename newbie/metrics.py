#! /usr/bin/env python3
"""Metrics for comparing the posteriors to the (synthetic) true values.

Requirements
- Read synthetic truth from file
- Get estimator
- Different handling of categorical inference of reactor type
"""

import numpy as np
import xarray as xr
import arviz as az
import pandas as pd

from . import evidence
from . import estimators


class Metric:

    def __init__(self, estimator, inference_data, truth) -> None:
        self.estimator = estimators.EstimatorFactory.create_estimator(
            estimator, inference_data)
        self.truth = truth

    def calculate_distance(self, normalize=None, absolute=False, **kwargs):
        """Calculate distance between estimator and truth."""
        est = self.estimator.calculate_estimator(**kwargs)
        dist = self.truth - est
        if absolute:
            dist = np.abs(dist)
        if normalize == 'truth':
            dist /= self.truth
        elif normalize == 'max':
            range = (self.estimator.inference_data.posterior.max() -
                     self.estimator.inference_data.posterior.min())
            dist /= np.abs(range)
        return dist

    def calculate_metric(self, unit=True, **kwargs):
        """Calculate the euclidian norm of the distances."""
        dist = self.calculate_distance(**kwargs)
        d = dist.to_array(dim='data_vars')
        if unit:
            return np.linalg.norm(d) / len(d)
        return np.linalg.norm(d)
