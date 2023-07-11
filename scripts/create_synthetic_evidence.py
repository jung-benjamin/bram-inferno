#! /usr/bin/env python3
"""Create a set of synthetic data as as test points.

Takes associated sets of reactor parameter values and
isotopic concentration data and creates files of
synthetic evidence.

The range of acceptable parameter values can be restricted,
although currently the restriction cannot differentiate
between different data sets (e.g. from different reactors)
that may require separate consideration.
"""

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
    sub = parser.add_subparsers(required=True)

    single = sub.add_parser('single')
    single.set_defaults(func=single_batch_evidence)
    mixture = sub.add_parser('mixture')
    mixture.set_defaults(func=mixture_evidence)

    n_components = 'Number of batches in the mixture'
    mixture.add_argument('--n-components',
                         help=n_components,
                         type=int,
                         default=2)
    mix_with_self = 'Set flag to mix evidence from the group.'
    mixture.add_argument('--mix-with-self',
                         help=mix_with_self,
                         action='store_true')
    mixing_bounds = 'Sample mixing ratios uniformly between these bounds.'
    mixture.add_argument('--mixing-bounds',
                         help=mixing_bounds,
                         type=float,
                         nargs=2,
                         default=(0.3, 0.7))

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


def gen_batch_ids(num_batches):
    """Create an iterator of upper case letters."""
    return map(chr, range(65, 65 + num_batches))


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


def restrict_parameter_ranges(x_data, y_data, args):
    """Remove data points outside user-specified bounds."""
    conditions = [get_condition(s) for s in args.param_select]
    compared = [
        c['op'](x_data.loc[:, c['param']], c['val']) for c in conditions
    ]
    select = compared.pop(0)
    for s in compared:
        select *= s
    x_data = x_data[select]
    y_data = y_data.T[select].T
    return x_data, y_data


def choose_single_batch_evidence(x_data, y_data, args):
    """Select evidence values by random sampling."""
    subsets = []
    for g, group in x_data.groupby(level=0):
        choices = RNG.choice(group.index, size=args.num_choices, replace=False)
        subsets.append(group.loc[choices])
    x_chosen = pd.concat(subsets)
    y_chosen = y_data.loc[:, x_chosen.index]
    return x_chosen, y_chosen


def single_batch_evidence(args):
    """Create synthetic evidence of single batches of waste."""
    x_data, y_data = load_data_groups(args)
    if args.param_select:
        x_data, y_data = restrict_parameter_ranges(x_data, y_data, args)

    x_chosen, y_chosen = choose_single_batch_evidence(x_data, y_data, args)
    if args.save:
        x_chosen.droplevel(0,
                           axis=0).to_csv(f'synthetic_truth_{args.save}.csv')
        y_chosen.droplevel(
            0, axis=1).to_csv(f'synthetic_evidence_{args.save}.csv')


def combine_batches(args):
    """Randomly combine batch names to form mixtures
    
    Randomly combines batches to determine the components
    of the synthetic mixtures. Depending on the arguments,
    the same batch group can be selected more than once or
    not.
    """
    if (args.n_components > len(args.group_names)) and (args.mix_with_self
                                                        == False):
        msg = (f'Cannot select {args.n_components} ' +
               f'from {len(args.group_names)} if `--mix-with-self` is False.')
        raise Exception(msg)
    else:
        combos = [
            RNG.choice(args.group_names,
                       size=args.n_components,
                       replace=args.mix_with_self)
            for i in range(args.num_choices)
        ]
        return np.array(combos)


def add_id_to_params(params, id, sep='_'):
    """Return a dictionary mapping parameters with ids."""
    return {p: f'{p}{sep}{id}' for p in params}


def choose_mixture_evidenc(x_data, y_data, args):
    """Add evidence values to create mixtures."""
    combo_arr = combine_batches(args)
    labels = list(gen_batch_ids(args.n_components))
    sub_y = []
    sub_x = []
    mix_ids = []
    for combo in combo_arr:
        select_ids = [
            RNG.choice(x_data.loc[pd.IndexSlice[i, :], :].index, size=1)
            for i in combo
        ]
        alphas = RNG.uniform(*args.mixing_bounds, size=len(combo))
        # Normalize and round the mixing ratios
        alphas = (alphas / alphas.sum()).round(2)
        new_id = '-'.join(
            [f'{i[0][1]}_{a}' for i, a in zip(select_ids, alphas)])
        fraction_y = [
            alphas[i] * y_data.loc[:, select_ids[i]] for i in range(len(combo))
        ]
        mixed_y = pd.concat(fraction_y, axis=1).sum(axis=1)
        mixed_y.name = new_id
        mix_ids.append(new_id)
        sub_y.append(mixed_y)

        alpha_df = pd.DataFrame(alphas[None, :],
                                columns=[f'alpha_{l}' for l in labels
                                         ]).set_index([[new_id]])
        mixed_x = pd.concat([
            x_data.loc[i, :].rename(add_id_to_params(x_data.columns, l),
                                    axis=1).reset_index(drop=True)
            for i, l in zip(select_ids, labels)
        ],
                            axis=1).set_index([[new_id]])
        sub_x.append(pd.concat([mixed_x, alpha_df], axis=1))
    x_chosen = pd.concat(sub_x)
    y_chosen = pd.concat(sub_y, axis=1)
    return x_chosen, y_chosen


def mixture_evidence(args):
    """Create synthetic evidence of mixtures of batches of waste."""
    x_data, y_data = load_data_groups(args)
    if args.param_select:
        x_data, y_data = restrict_parameter_ranges(x_data, y_data, args)
    x_chosen, y_chosen = choose_mixture_evidenc(x_data, y_data, args)
    if args.save:
        x_chosen.to_csv(f'synthetic_truth_{args.save}.csv')
        y_chosen.to_csv(f'synthetic_evidence_{args.save}.csv')


if __name__ == '__main__':
    args = argparser()
    args.func(args)
