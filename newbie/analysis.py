#! /usr/bin/env python3
"""A module to analyze posterior distributions.

Contains functions to facilitate analyzing the Bayesian posteriors
and automatically assess a large number if inference results and
evaluate the performance of the inference model.
"""
import logging

import arviz as az
import pandas as pd
import xarray as xr

from newbie import metrics


class PosteriorAnalysis:

    def __init__(self,
                 *args,
                 estimators=['peak', 'mean', 'mode'],
                 loglevel='INFO'):
        """Instantiate the class with input arguments
        
        Parameters
        args : (id, inference_data, truth)
        """
        self.analyses = {}
        for i, d, t in args:
            self.analyses[i] = metrics.MetricSet(d, t, estimators)
        self.logger = self.get_logger(loglevel=loglevel)

    @classmethod
    def get_logger(cls, loglevel='INFO'):
        """Configure a logger for the class."""
        log = logging.getLogger(cls.__name__)
        log.setLevel(getattr(logging, loglevel.upper()))
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, loglevel.upper()))
        log.addHandler(ch)
        return log

    @classmethod
    def from_filepaths(cls,
                       inference_files,
                       truth_file,
                       ids=None,
                       loglevel='INFO'):
        """Instantiate class with data stored in files."""
        log = cls.get_logger(loglevel)
        truth = pd.read_csv(truth_file, index_col=0)
        if not ids:
            ids = list(truth.index)
        log.info('Creating PosteriorAnalysis...')
        log.info(f'IDs: {ids}')
        idata = {}
        if isinstance(inference_files, dict):
            for i in ids:
                try:
                    idata[i] = az.from_json(inference_files[i])
                except KeyError:
                    msg = f'Warning: {i} has no corresponding inference file.'
                    log.warning(msg)
        elif isinstance(inference_files, list):
            for i in ids:
                for f in inference_files:
                    if i in f.name:
                        idata[i] = az.from_json(f)
        tr = [xr.Dataset(truth.loc[n].to_dict()) for n in idata]
        args = list(zip(*zip(*idata.items()), tr))
        return cls(*args, loglevel=loglevel)

    def calculate_distances(self, **kwargs):
        """Calculate distances between estimators and true parameters."""
        dist = []
        for n, it in self.analyses.items():
            d = it.calculate_distances(**kwargs)
            dist.append(d)
        ds = xr.concat(dist, dim='ID')
        return ds.assign_coords(ID=list(self.analyses))
