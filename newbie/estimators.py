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
import xarray as xr
import numpy as np
# import scipy.signal
from scipy import signal, stats


class Estimator:

    def __init__(self, inference_data):
        self.inference_data = inference_data

    def calculate_estimator(self):
        raise NotImplementedError(
            "Subclasses of Estimator must implement calculate_estimator() method"
        )


class ModeEstimator(Estimator):

    def mode(self, x, **kwargs):
        pe = az.plots.plot_utils.calculate_point_estimate
        return pe('mode', np.concatenate(x), **kwargs)

    def calculate_estimator(self, **kwargs):
        estimator = {
            n: self.mode(it, **kwargs)
            for n, it in self.inference_data.posterior.items()
        }
        return xr.Dataset(estimator)


class MeanEstimator(Estimator):

    def calculate_estimator(self):
        return self.inference_data.posterior.mean()


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

    def peak(self, posterior, kind='height', **kwargs):
        """Calculate the peak estimator
        
        Returns either the highest or the most prominent peak
        as a characteristic of the posterior.
        """
        peaks = self._assess_peaks(posterior, **kwargs)
        return peaks['loc'][np.argmax(peaks[kind])]

    def calculate_estimator(self, kind='height', **kwargs):
        estimators = {
            n: self.peak(it)
            for n, it in self.inference_data.posterior.items()
        }
        return xr.Dataset(estimators)


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
