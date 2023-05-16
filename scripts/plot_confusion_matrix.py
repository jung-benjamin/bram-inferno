#! /usr/bin/env python3
"""Calculate and plot confusion matrix for reactor discrimination."""

import argparse
import json
from pathlib import Path

from newbie.analysis import ConfusionAnalyzer


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    idata = 'Json files with inference data.'
    parser.add_argument('idata', help=idata, nargs='+', type=Path)
    loglevel = 'Set the logging level.'
    parser.add_argument('-l', '--loglevel', help=loglevel, default='INFO')
    show = 'Show the confusion matrix plot.'
    parser.add_argument('--show', help=show, action='store_true')
    parser.add_argument('-o', '--output', type=Path)
    print_ = 'Print the confusion matrix.'
    parser.add_argument('--print', help=print_, action='store_true')
    reactor_map = 'Json file with a mapping of labels to integers.'
    parser.add_argument('--reactor-map', help=reactor_map, type=Path)
    class_var = 'Variable name of the classification parameter.'
    parser.add_argument('--class-var', help=class_var, default='cat')
    return parser.parse_args()


def plot_confusion():
    args = argparser()
    ConfusionAnalyzer.config_logger(loglevel=args.loglevel)
    ca = ConfusionAnalyzer.from_json(args.idata, class_var=args.class_var)
    if args.reactor_map:
        with args.reactor_map.open('r') as f:
            ca.reactor_map = json.load(f)
    ca.get_class_results()
    ca.calc_confusion_matrix()
    if args.print:
        print(ca.confusion_matrix)
    ca.plot_confusion_matrix(args.show, args.output)


if __name__ == '__main__':
    plot_confusion()