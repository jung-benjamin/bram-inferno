#! /usr/bin/env python3
"""A module to analyze posterior distributions.

Contains functions to facilitate analyzing the Bayesian posteriors
and automatically assess a large number if inference results and
evaluate the performance of the inference model.
"""
import logging
from collections import Counter
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
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
        self.logger.setLevel(getattr(logging, loglevel.upper()))

    @classmethod
    def get_logger(cls, loglevel='INFO'):
        """Configure a logger for the class."""
        log = logging.getLogger(cls.__name__)
        log.setLevel(getattr(logging, loglevel.upper()))
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, loglevel.upper()))
        log.addHandler(ch)
        return log

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

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
        log.info(f'Number of inference files: {len(inference_files)}')
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
        log.debug(f'Number of init *args: {len(args)}')
        return cls(*args, loglevel=loglevel)

    def calculate_distances(self, **kwargs):
        """Calculate distances between estimators and true parameters."""
        self.logger.info(
            f'Calculating distances for {len(self.analyses)} items.')
        dist = []
        for n, it in self.analyses.items():
            self.logger.info(f'ID: {n}...')
            d = it.calculate_distances(**kwargs)
            dist.append(d)
            self.logger.info(f'Done.')
        ds = xr.concat(dist, dim='ID')
        return ds.assign_coords(ID=list(self.analyses))


class ConfusionAnalyzer:
    """Analyze confusion matrix of reactor discrimination."""

    def __init__(self, idata, ids=None, class_var=None):
        """Instantiate the class with a list of inference data objects."""
        if isinstance(idata, (list, tuple)):
            if ids:
                self.idata = dict(zip(ids, idata))
            else:
                msg = 'Inference data has no ids.'
                self.idata = dict(zip(range(len(idata)), idata))
        elif isinstance(idata, dict):
            self.idata = idata
        else:
            raise Exception(
                'Inference data must be `list`, `tuple` or `dict`.')
        self.info = {}
        if ids:
            for i in ids:
                self.info[i] = self.parse_name(i)
        self.class_var = class_var
        self.reactor_map = {}

    name_format = 'drop_drop_reactor_id'

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

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

    @classmethod
    def from_json(cls, file_paths, fmt=None, class_var=None):
        """Load the inference data from a list of json files."""
        idata = [az.from_json(f) for f in file_paths]
        ids = [Path(f).stem for f in file_paths]
        if fmt:
            cls.set_name_format(fmt)
        return cls(idata, ids, class_var)

    @classmethod
    def parse_name(cls, name):
        """Extract data from the filenames given a format.

        An underscore is used to separate the format and the filename.
        """
        info = {}
        for i, j in zip(cls.name_format.split('_'), name.split('_')):
            if i == 'drop':
                pass
            else:
                info[i] = j
        return info

    @classmethod
    def set_name_format(cls, fmt):
        """Change the name format."""
        cls.name_format = fmt

    @property
    def reactor_map(self):
        """Return mapping of reactor name to number"""
        return self._reactor_map

    @reactor_map.setter
    def reactor_map(self, d):
        """Set the reactor map."""
        self._reactor_map = d

    @property
    def inv_reactor_map(self):
        """Mapping of number to reactor name"""
        return {it: n for n, it in self.reactor_map.items()}

    def _calc_class_posteriors(self, varname):
        """Translate class posteriors."""
        self.class_posteriors = {}
        for n, it in self.idata.items():
            self.class_posteriors[n] = [
                self.inv_reactor_map[i]
                for i in it['posterior'][varname].values.flatten()
            ]
        return self.class_posteriors

    def get_class_results(self, varname=None, threshold=0.7):
        """Determine results of the classification."""
        if not varname:
            varname = self.class_var
        self._calc_class_posteriors(varname=varname)
        self.class_results = {}
        for n, it in self.class_posteriors.items():
            counts = Counter(it)
            best = counts.most_common(1)
            self.logger.debug(f'Class results {n}: {best}')
            if not ((best[0][1] / sum(counts.values())) > threshold):
                results = 'inconclusive'
            else:
                results = best[0][0]
            self.class_results[n] = results
        return self.class_results

    def calc_confusion_matrix(self):
        """Calculate the confusion matrix."""
        try:
            predicted_labels = list(self.class_results.values())
        except AttributeError:
            self.get_class_results()
            predicted_labels = list(self.class_results.values())
        true_labels = [self.info[n]['reactor'] for n in self.class_results]

        unique_labels = set(true_labels + predicted_labels)
        num_classes = len(unique_labels)
        label_to_index = {
            label: index
            for index, label in enumerate(unique_labels)
        }

        matrix = [[0] * num_classes for _ in range(num_classes)]
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            true_index = label_to_index[true_label]
            predicted_index = label_to_index[predicted_label]
            matrix[true_index][predicted_index] += 1

        self.matrix_labels = list(label_to_index)
        self.confusion_matrix = np.array(matrix)
        return self.confusion_matrix

    def plot_confusion_matrix(self, show=False, save=None):
        """Plot the confusion matrix"""
        try:
            matrix = self.confusion_matrix
        except AttributeError:
            matrix = self.calc_confusion_matrix()
        labels = self.matrix_labels
        fig, ax = plt.subplots(constrained_layout=True)
        im = ax.imshow(matrix, cmap='Blues')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Rotate the tick labels and set alignment
        plt.setp(
            ax.get_xticklabels(),
            #  rotation=45,
            ha="right",
            rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text_color = "w" if matrix[i, j] > np.max(matrix) / 2 else "k"
                text = ax.text(j,
                               i,
                               matrix[i][j],
                               ha="center",
                               va="center",
                               color=text_color)

        # Set title and labels
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        if save:
            plt.savefig(save)
        if show:
            plt.show()


class ClassificationAnalysis(ConfusionAnalyzer):
    """Analyze the results of inference with classification"""

    # def __init__(self,
    #              *args,
    #              estimators=['peak', 'mean', 'mode'],
    #              loglevel='INFO'):
    #     """Instantiate the class."""
    #     super().__init__(*args, estimators=estimators, loglevel=loglevel)
    #     self.class_var = None
    #     self.reactor_map = {}

    def reactor_map_from_mixing(self, var_names=None):
        """Determine the reactor map from the mixing ratio variables."""
        if not var_names:
            alpha_variables = []
            for i, it in self.idata.items():
                alpha_variables.extend(
                    [v for v in it['posterior'].data_vars if 'alpha' in v])
            alpha_variables = sorted(set(alpha_variables))
        else:
            alpha_variables = sorted(set(var_names))
        # To-Do: find neater way to access the reactor ID
        self.reactor_map = {v[-1]: i for i, v in enumerate(alpha_variables)}
        return self.reactor_map

    def get_class_posteriors(self, varname=None):
        """Access only the posteriors from each class"""
        if not varname:
            varname = self.class_var
        self._calc_class_posteriors(varname=varname)
        self.get_class_results()
        for n, it in self.class_results.items():
            if it == 'inconclusive':
                self.idata.pop(n)
                continue
            pos = self.idata[n].posterior
            self.logger.debug(f'Data vars {n}: {list(pos.data_vars)}')
            drop = [v for v in pos.data_vars if it not in v]
            drop += [f'alpha{it}']
            self.logger.debug(f'Drop variables {n}: {drop}')
            self.idata[n].posterior = pos.drop_vars(drop)
        return {n: it.posterior for n, it in self.idata.items()}

    def load_truth(self, filepath):
        """Load the truth dataframe from the file"""
        self.truth = pd.read_csv(filepath, index_col=0)
        return self.truth

    def calc_class_metrics(self,
                           truthfile,
                           estimators=['mean', 'mode', 'peak'],
                           varname=None):
        """Calculate metrics for the posteriors of the classification result"""
        self.get_class_posteriors(varname=varname)
        truth = self.load_truth(truthfile)
        self.analyses = {}
        for i, d in self.idata.items():
            id_ = '_'.join(list(self.parse_name(i).values()))
            tr = truth.loc[id_]
            tr.rename({n: f'{n}{self.class_results[i]}'
                       for n in tr.index},
                      inplace=True)
            tr = xr.Dataset(tr.to_dict())
            self.logger.debug(f'Truth ids: {list(tr)}')
            self.logger.debug(f'Idata ids: {list(d["posterior"].data_vars)}')
            self.analyses[i] = metrics.MetricSet(d, tr, estimators)

    def calculate_distances(self, **kwargs):
        """Calculate distances between estimators and true parameters."""
        self.logger.info(
            f'Calculating distances for {len(self.analyses)} items.')
        dist = []
        for n, it in self.analyses.items():
            self.logger.info(f'ID: {n}...')
            d = it.calculate_distances(**kwargs)
            # To-Do: find a less hard-coded way to rename the data variables
            d = d.rename_vars({n: n[:-1] for n in d.data_vars})
            self.logger.debug(f'Distance variables: {list(d.data_vars)}')
            dist.append(d)
            self.logger.info(f'Done.')
        ds = xr.concat(dist, dim='ID')
        return ds.assign_coords(ID=list(self.analyses))
