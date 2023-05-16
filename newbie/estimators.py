#!/usr/bin/env python3
"""
A module containing a factory class for producing subclasses of an estimator base class.

The estimator base class provides a method for calculating an estimator based on an arviz InferenceData object.
The factory class provides three subclasses of the estimator: ModeEstimator, MeanEstimator, and PeakEstimator.

The ModeEstimator calculates the mode of the posterior variables in the inference data.
The MeanEstimator calculates the mean of the posterior variables in the inference data.
The PeakEstimator finds the highest peak or all peaks that have a prominence above a threshold set by the user.
The user can modify other keyword arguments of the find_peaks method.

This module uses the scipy style guide for docstrings.
"""

import arviz as az
import numpy as np
import xarray as xr
from scipy import signal, stats


class Estimator:

    def __init__(self, inference_data):
        self.inference_data = inference_data

    def _estimator_func(self, posterior, **kwargs):
        """Warpper for functions implemented by subclasses."""
        msg = 'Subclasses of Estimator must implement _estimator_func().'
        raise NotImplementedError(msg)

    def calculate_estimator(self, data_vars=None, **kwargs):
        """Calculate the estimator for posteriors in the inference data."""
        estimator = {}
        if isinstance(data_vars, str):
            pos = self.inference_data['posterior'][data_vars]
            estimator[data_vars] = self._estimator_func(pos, **kwargs)
        elif data_vars:
            for v in data_vars:
                pos = self.inference_data['posterior'][v]
                estimator[v] = self._estimator_func(pos, **kwargs)
        else:
            for v, it in self.inference_data.posterior.items():
                estimator[v] = self._estimator_func(it, **kwargs)
        self.estimator = xr.Dataset(estimator)
        return self.estimator


class ModeEstimator(Estimator):

    def _estimator_func(self, posterior, **kwargs):
        """Calculate the mode of the data samples."""
        pe = az.plots.plot_utils.calculate_point_estimate
        return pe('mode', np.concatenate(posterior), **kwargs)


class MeanEstimator(Estimator):

    def _estimator_func(self, posterior, **kwargs):
        """Calculate the mean of the data samples."""
        return posterior.mean()


class PeakEstimator(Estimator):

    def _get_kde_samples(self, posterior, backend='arviz'):
        """Resample from the posterior with a gaussian kde.
        
        Parameters
        ----------
        posterior : array
            Posterior as stored in inference data.
        backend : str (optional, {'scipy', 'arviz'})
            Select backend for calculating the kde."""
        arr = np.concatenate(posterior)
        if backend == 'scipy':
            kde = stats.gaussian_kde(arr)
            samples = np.linspace(min(arr), max(arr), 1000)
            probs = kde.evaluate(samples)
            return samples, probs
        elif backend == 'arviz':
            return az.kde(arr)

    def _assess_peaks(self,
                      posterior,
                      kde_lib='arviz',
                      height=None,
                      wlen=None,
                      **kwargs):
        """Find peaks in the posterior distribution.
        
        Uses a kernel density estimate to obtain a smoothed sample
        of the posterior and determines the peaks of the sample.
        """
        samples, probs = self._get_kde_samples(posterior, backend=kde_lib)
        if not height:
            height = max(probs) * 0.5
        peak_idx, properties = signal.find_peaks(probs,
                                                 height=height,
                                                 wlen=wlen,
                                                 **kwargs)
        if len(peak_idx) == 0:
            msg = f'Number of peaks is {len(peak_idx)}'
            print('Warning:', msg)
            return
        if not wlen:
            wlen = len(samples) / len(peak_idx)
        peak_prominence = signal.peak_prominences(probs, peak_idx, wlen=wlen)
        peaks = {
            'loc': samples[peak_idx],
            'probability': probs[peak_idx],
            'height': properties['peak_heights'],
            'prominence': peak_prominence[0]
        }
        return peaks

    def _estimator_func(self, posterior, kind='height', **kwargs):
        """Calculate the peak estimator
        
        Returns either the highest or the most prominent peak
        as a characteristic of the posterior.
        """
        peaks = self._assess_peaks(posterior, **kwargs)
        try:
            return peaks['loc'][np.argmax(peaks[kind])]
        except TypeError:
            return np.inf


class EstimatorFactory:

    @classmethod
    def create_estimator(cls, estimator_type, inference_data):
        if estimator_type == "mode":
            return ModeEstimator(inference_data)
        elif estimator_type == "mean":
            return MeanEstimator(inference_data)
        elif estimator_type == "peak":
            return PeakEstimator(inference_data)
        else:
            raise ValueError("Invalid estimator type")
