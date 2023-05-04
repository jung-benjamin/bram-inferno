#! /usr/bin/env python3
"""A command line tool to evaluate Bayesian inference results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from newbie.analysis import PosteriorAnalysis
from newbie.plot_utils import DistanceHistogram, DistanceScatter


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    inference_files = 'Json files conatining inference data.'
    parser.add_argument('idata', help=inference_files, type=Path, nargs='+')
    truth_file = 'Csv file containing true parameter values.'
    parser.add_argument('-t', '--truth', help=truth_file, type=Path)
    ids = 'List of strings that identify individual inference results.'
    parser.add_argument('-i', '--ids', help=ids, nargs='*')
    save = 'Path to save the results to.'
    parser.add_argument('-s', '--save', type=Path, help=save)
    print_ = 'Print output to console.'
    parser.add_argument('-p', '--print', help=print_, action='store_true')
    loglevel = 'Set the logging level.'
    parser.add_argument('-l', '--log-level', help=loglevel, default='INFO')
    abs_distance = 'Calculate absolute values of distances.'
    parser.add_argument('--abs-distance',
                        help=abs_distance,
                        action='store_true')
    scatterplot = 'Create a scatterplot of the metrics.'
    parser.add_argument('--scatterplot', help=scatterplot, action='store_true')
    histplot = 'Create a histogram plot of the metrics.'
    parser.add_argument('--histplot', help=histplot, action='store_true')

    # subparser = argparse.ArgumentParser()
    # sub = subparser.add_subparsers(title='Analysis options')
    # histogram = 'Create a histogram plot of the metrics.'
    # hist_sub = sub.add_parser('histogram', help=histogram, parents=[parser])
    # hist_sub.set_defaults(func=plot_histogram)

    return parser.parse_args()


def create_analyzer(args):
    """Create instance of PosteriorAnalysis."""
    return PosteriorAnalysis.from_filepaths(args.idata,
                                            args.truth,
                                            args.ids,
                                            loglevel=args.log_level)


def plot_histogram(distances, args):
    """Create a histogram plot of the distances."""
    dh = DistanceHistogram(distances)
    fig, axes = dh.plot_distances()
    if args.save:
        fp = f'{args.save.stem}_hist.png'
        plt.savefig(fp)
    plt.show()


def plot_scatter(distances, args):
    ds = DistanceScatter(distances)
    fig, axes = ds.plot_distances()
    if args.save:
        fp = f'{args.save.stem}_scatter.png'
        plt.savefig(fp)
    plt.show()


def evaluate_posteriors():
    """Evaluate the results of Bayesian inference posteriors."""
    args = argparser()
    pa = create_analyzer(args)
    distances = pa.calculate_distances(absolute=args.abs_distance)
    if args.print:
        print(distances)
    if args.save:
        with args.save.with_suffix('.json').open('w') as f:
            json.dump(distances.to_dict(), f, indent=True)
    if args.histplot:
        plot_histogram(distances, args)
    if args.scatterplot:
        plot_scatter(distances, args)


if __name__ == '__main__':
    evaluate_posteriors()