#! /usr/bin/env python3
"""Plot the posterior distribution of the inference parameters."""

import argparse
import logging
import re
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

from newbie.plot_utils import PosteriorPlot

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


def get_truth(args):
    """Load the synthetic truth from a csv file."""
    if args.truth_file:
        truth = pd.read_csv(args.truth_file, index_col=0)
        if args.truth_key:
            return truth.loc[args.truth_key]
        else:
            try:
                truth_key = '_'.join(args.infile.stem.split('_')[2:])
                return truth.loc[truth_key]
            except KeyError:
                msg = 'Truth key could not be inferred from the file name.'
                raise KeyError(msg)


def prettyplot_inferencedata(args):
    """Create a plot of a single inference data file."""
    idata = az.from_json(args.infile)
    PosteriorPlot.config_logger(loglevel=args.log_level)
    plotter = PosteriorPlot(inference_data=idata)
    plotter.plot(
        truth=get_truth(args),
        estimators=args.estimators,
        save=args.save,
        figsize=args.figsize,
        constrained_layout=True,
    )


if __name__ == '__main__':
    args = argparser()
    config_logger(args)
    prettyplot_inferencedata(args)
