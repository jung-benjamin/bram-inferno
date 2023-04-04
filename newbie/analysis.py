#! /usr/bin/env python3
"""A module to analyze posterior distributions.

Contains functions to facilitate analyzing the Bayesian posteriors
and automatically assess a large number if inference results and
evaluate the performance of the inference model.
"""

import arviz as az
import pandas as pd
import xarray as xr

from newbie import metrics


class PosteriorAnalysis:

    def __init__(self, *args, estimators=['peak', 'mean', 'mode']):
        """Instantiate the class with input arguments
        
        Parameters
        args : (id, inference_data, truth)
        """
        self.analyses = {}
        for i, d, t in args:
            self.analyses[i] = metrics.MetricSet(d, t, estimators)

    @classmethod
    def from_filepaths(cls, inference_files, truth_file, ids=None):
        """Instantiate class with data stored in files."""
        truth = pd.read_csv(truth_file, index_col=0)
        if ids:
            ids = list(truth.index)
        idata = {}
        if isinstance(inference_files, dict):
            for i in ids:
                try:
                    idata[i] = az.from_json(inference_files[i])
                except KeyError:
                    msg = f'Warning: {i} has no corresponding inference file.'
        elif isinstance(inference_files, list):
            for i in ids:
                for f in inference_files:
                    if i in f.name:
                        idata[i] = az.from_json(f)
        tr = [xr.Dataset(truth.loc[n].to_dict()) for n in idata]
        args = list(zip(*zip(*idata.items()), tr))
        return cls(*args)

    def calculate_distances(self, **kwargs):
        """Calculate distances between estimators and true parameters."""
        dist = []
        for n, it in self.analyses.items():
            d = it.calculate_distances(**kwargs)
            dist.append(d)
        ds = xr.concat(dist, dim='ID')
        return ds.assign_coords(ID=list(self.analyses))
