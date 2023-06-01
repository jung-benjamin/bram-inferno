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

import logging

import arviz as az
import numpy as np
import xarray as xr
from scipy import signal, stats


class Estimator:

    def __init__(self, inference_data=None):
        self.inference_data = inference_data

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

    def _estimator_func(self, posterior, **kwargs):
        """Warpper for functions implemented by subclasses."""
        msg = 'Subclasses of Estimator must implement _estimator_func().'
        raise NotImplementedError(msg)

    def calculate_estimator(self,
                            data_vars=None,
                            inference_data=None,
                            **kwargs):
        """Calculate the estimator for posteriors in the inference data."""
        if inference_data is None:
            inference_data = self.inference_data
            if inference_data is None:
                msg = ('Inference data has not been specified during' +
                       ' initalization.')
                raise ValueError(msg)
        estimator = {}
        if isinstance(data_vars, str):
            pos = inference_data['posterior'][data_vars]
            estimator[data_vars] = self._estimator_func(pos, **kwargs)
        elif data_vars:
            for v in data_vars:
                pos = inference_data['posterior'][v]
                estimator[v] = self._estimator_func(pos, **kwargs)
        else:
            for v, it in inference_data.posterior.items():
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
            self.logger.warn('Number of peaks is 0.')
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
    def create_estimator(cls, estimator_type, inference_data=None):
        if estimator_type == "mode":
            return ModeEstimator(inference_data)
        elif estimator_type == "mean":
            return MeanEstimator(inference_data)
        elif estimator_type == "peak":
            return PeakEstimator(inference_data)
        else:
            raise ValueError("Invalid estimator type")
