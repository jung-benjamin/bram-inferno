#! /usr/bin/env python3
"""Plot the posterior distribution of the inference parameters."""

import argparse
import logging
import re
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

from newbie.estimators import EstimatorFactory

PARAM_REGEX = re.compile('([a-zA-Z])([a-z]+)(\d|[A-Z]?)')


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    infile = 'Json file with the inference data.'
    parser.add_argument('infile', help=infile, type=Path)
    truth_file = 'File with true parameter values of synthetic evidence.'
    parser.add_argument('-t', '--truth-file', help=truth_file)
    truth_key = 'Key for indexing the truth file.'
    parser.add_argument('-k', '--truth-key', help=truth_key)
    figsize = 'Figure size for the plot.'
    parser.add_argument('-f',
                        '--figsize',
                        help=figsize,
                        type=float,
                        nargs=2,
                        default=[8, 6])
    loglevel = 'Set logging level.'
    parser.add_argument('-l', '--log-level', help=loglevel, default='INFO')
    estimators = 'Select estimators to plot.'
    parser.add_argument('-e', '--estimators', help=estimators, nargs='*')
    save = 'Filepath for saving the plot.'
    parser.add_argument('-s', '--save', help=save, type=Path)
    return parser.parse_args()


def config_logging(args):
    """Configure logging module."""
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))


def config_logger(args):
    """Configure logger of this script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, args.log_level.upper()))
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, args.log_level.upper()))
    logger.addHandler(sh)


def logger():
    """Return the logger of this script."""
    return logging.getLogger(__name__)


def get_param_batch_id(data_var):
    """Extract the batch id of an inference variable."""
    c = PARAM_REGEX.fullmatch(data_var)
    if c:
        # param = ''.join([c.group(1), c.group(2)])
        batch_id = c.group(3)
        return batch_id


def group_batch_parameters(idata):
    """Group inference variables by their batch id."""
    batch_params = {}
    skipped_params = []
    for n in list(idata.posterior.data_vars):
        batch_id = get_param_batch_id(n)
        if batch_id:
            try:
                batch_params[batch_id].extend([n])
            except KeyError:
                batch_params[batch_id] = [n]
        else:
            skipped_params.append(n)
    return {n: sorted(it) for n, it in batch_params.items()}, skipped_params


def plot_posteriors(idata, **plot_kw):
    """Plot the posterior distributions of an inference run"""
    idata = az.from_json(args.infile)
    batch_params, others = group_batch_parameters(idata)
    logger().debug(f'Batch parameters: {batch_params}')
    logger().debug(f'Skipped parameters: {others}')
    nrows = len(batch_params)
    ncols = max([len(it) for n, it in batch_params.items()])
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, **plot_kw)
    ax_dict = {}
    for i, (n, it) in enumerate(batch_params.items()):
        az.plot_posterior(idata,
                          var_names=it,
                          ax=axes[i, :],
                          point_estimate=None,
                          hdi_prob='hide')
        for i, a in zip(it, axes[i, :]):
            ax_dict[i] = a
    return fig, ax_dict


def plot_truth(args, ax_dict):
    """Add vertical lines of true parameter values to the plots"""
    truth = pd.read_csv(args.truth_file, index_col=0)
    if args.truth_key:
        tr = truth.loc[args.truth_key]
    else:
        try:
            truth_key = '_'.join(args.infile.stem.split('_')[2:])
            tr = truth.loc[truth_key]
        except KeyError:
            msg = 'Truth key could not be inferred from the file name.'
            raise KeyError(msg)
    for n, ax in ax_dict.items():
        ax.axvline(tr[n], ls='dashed')
    return ax_dict


def plot_estimator(estimator, ax_dict, color, label):
    """Add vertical lines of posterior estimators to the plots."""
    for n, ax in ax_dict.items():
        e = estimator[n]
        ax.axvline(e, ls='dotted', color=color, label=label)


def prettyplot_inferencedata(args):
    """Create a plot of a single inference data file."""
    idata = az.from_json(args.infile)
    fig, axes = plot_posteriors(idata,
                                figsize=args.figsize,
                                constrained_layout=True)
    if args.truth_file:
        plot_truth(args, axes)
    if args.estimators:
        colors = plt.cm.tab10
        for i, e in enumerate(args.estimators):
            esti = EstimatorFactory.create_estimator(e, idata)
            esti_data = esti.calculate_estimator()
            plot_estimator(esti_data, axes, colors(i), label=e)
    legend = list(axes.values())[0].get_legend_handles_labels()
    # fig.legend(*legend, loc='outside upper center')
    # fig.subplots_adjust(top=0.95)
    if args.save:
        plt.savefig(args.save)
    plt.show()


if __name__ == '__main__':
    args = argparser()
    config_logger(args)
    prettyplot_inferencedata(args)
