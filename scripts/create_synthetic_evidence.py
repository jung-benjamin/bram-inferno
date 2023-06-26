#! /usr/bin/env python3
"""Create a set of synthetic data as as test points."""

import argparse
import operator as op
import re
from pathlib import Path

import numpy as np
import pandas as pd

OP = {
    '=': op.eq,
    '<': op.lt,
    '>': op.gt,
    '=>': op.ge,
    '>=': op.ge,
    '<=': op.le,
    '=<': op.le,
    '==': op.eq
}

RNG = np.random.default_rng(seed=12345)


def get_condition(s):
    cond_regex = re.compile(r'([a-zA-Z]+)([=<>]+)([0-9\.]+)')
    cond = cond_regex.fullmatch(s)
    return {
        'param': cond.group(1),
        'op': OP[cond.group(2)],
        'val': float(cond.group(3))
    }


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    param_files = 'Csv files specifying parameter values.'
    parser.add_argument('--param-files',
                        help=param_files,
                        type=Path,
                        nargs='+')
    data_files = 'Csv files specifying corresponding isotopic data.'
    parser.add_argument('--data-files', help=data_files, type=Path, nargs='+')
    group_names = 'Names for identifying each dataset.'
    parser.add_argument('--group-names', help=group_names, nargs='+')
    skip_values = 'Number of values in datasets to be skipped.'
    parser.add_argument('--skip-values',
                        help=skip_values,
                        type=int,
                        default=200)
    param_select = 'Conditions for restricting the parameter space.'
    parser.add_argument('--param-select', help=param_select, nargs='*')
    num_choices = 'Number of samples to select per data set.'
    parser.add_argument('--num-choices',
                        help=num_choices,
                        type=int,
                        default=200)
    save = 'Name for saving the synthetic evidence data set.'
    parser.add_argument('-s', '--save', help=save)
    return parser.parse_args()


def rename_sample(idx, group_name):
    """Rename sample id with its group name."""
    num = idx.split('_')[-1]
    return f'{group_name}_{num}'


def reindex_dict(index, group_name):
    """Create dict for renaming sample ids."""
    return {i: rename_sample(i, group_name) for i in index}


def load_data_groups(args):
    """Load grouped data from the csv files."""
    x_data, y_data = [], []
    for x, y, g in zip(args.param_files, args.data_files, args.group_names):
        x_df = pd.read_csv(x)
        y_df = pd.read_csv(y, index_col=0)
        y_df.rename(reindex_dict(y_df.columns, g), axis=1, inplace=True)
        n_x = x_df.shape[0]
        n_y = y_df.shape[1]
        if not n_x == n_y:
            max_idx = min(n_x, n_y)
            x_df = x_df.iloc[:max_idx]
            y_df = y_df.iloc[:, :max_idx]
        else:
            pass
        x_df.index = y_df.columns
        x_data.append(x_df.iloc[args.skip_values:])
        y_data.append(y_df.iloc[:, args.skip_values:])
    x_data = pd.concat(x_data, keys=args.group_names, axis=0)
    y_data = pd.concat(y_data, keys=args.group_names, axis=1).dropna(axis=0)
    return x_data, y_data


if __name__ == '__main__':
    args = argparser()
    x_data, y_data = load_data_groups(args)
    if args.param_select:
        conditions = [get_condition(s) for s in args.param_select]
        compared = [
            c['op'](x_data.loc[:, c['param']], c['val']) for c in conditions
        ]
        select = compared.pop(0)
        for s in compared:
            select *= s
        x_data = x_data[select]
        y_data = y_data.T[select].T

    subsets = []
    for g, group in x_data.groupby(level=0):
        choices = RNG.choice(group.index, size=args.num_choices, replace=False)
        subsets.append(group.loc[choices])
    x_chosen = pd.concat(subsets)
    y_chosen = y_data.loc[:, x_chosen.index]
    if args.save:
        x_chosen.droplevel(0,
                           axis=0).to_csv(f'synthetic_truth_{args.save}.csv')
        y_chosen.droplevel(
            0, axis=1).to_csv(f'synthetic_evidence_{args.save}.csv')
