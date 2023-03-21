from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from ml_utilities.output_loader.result_loader import SweepResult


def _convert_to_series(df: pd.DataFrame, val_col: str, ind_col: str) -> pd.Series:
    df = df.sort_values(ind_col)
    return df[[val_col, ind_col]].set_index(ind_col).squeeze()


def _get_samples_y_col_series(df: pd.DataFrame, label: str = '', y_col_name: str = 'Accuracy-top-1-epoch-10'):
    ss_dict = {}
    for fraction in df['data.sample_selector.kwargs.fraction'].unique():
        ss = _convert_to_series(df[df['data.sample_selector.kwargs.fraction'] == fraction], y_col_name,
                                'data.sample_selector.kwargs.restrict_n_samples')
        ss_dict[f'{label}-frac{fraction}'] = ss
    return ss_dict


def make_pruning_figure(
    pruning_results: Dict[str, pd.Series],
    title: str = '',
    y_label: str = 'Accuracy-top-1',
    save_format: str = '',
    save_dir: str = '.',
    x_scale: str = 'linear',
    y_lim: Tuple[float, float] = (),  #(-0.05, 1),
    figsize=(1.5 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54)):
    #* Plotting: create figure
    f, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
    f.suptitle(title)

    for name, series in pruning_results.items():
        ax.plot(series.index, series.values, label=name)

    ax.legend()
    ax.grid(alpha=.3)
    ax.set_xlabel('number of samples')
    ax.set_ylabel(y_label)
    ax.set_xscale(x_scale)
    if y_lim:
        ax.set_ylim(y_lim)

    if save_format:
        assert title
        f.savefig(f'{str(save_dir)}/{title}.{save_format}', dpi=300, bbox_inches='tight')
    return f


def plot_pruning_results(random_pruning_results: SweepResult,
                         pd_pruning_results: SweepResult,
                         ft_epoch: int,
                         ft_col_sel: str,
                         title: str = '',
                         save_format: str = '',
                         save_dir: str = '.') -> plt.Figure:
    ft_row_sel = ('epoch', ft_epoch)
    col_name = f'{ft_col_sel}-{ft_row_sel[0]}-{ft_row_sel[1]}'
    rand_prune_df = random_pruning_results.get_summary(log_source='val', col_sel=ft_col_sel, row_sel=ft_row_sel)
    rand_prune_ss = _convert_to_series(rand_prune_df, col_name, 'data.sample_selector.kwargs.fraction')

    pd_prune_df = pd_pruning_results.get_summary(log_source='val', col_sel=ft_col_sel, row_sel=ft_row_sel)
    # pd_prune_keep_highest_ss = convert_to_series(pd_prune_df, col_name, 'data.sample_selector.kwargs.fraction')
    pd_prune_khighest_ss = _convert_to_series(
        pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == True], col_name,
        'data.sample_selector.kwargs.fraction')
    pd_prune_klowest_ss = _convert_to_series(
        pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == False], col_name,
        'data.sample_selector.kwargs.fraction')

    # TODO support different sample sizes

    pruning_results = {
        'rand_prune': rand_prune_ss,
        'pd_prune_khighest': pd_prune_khighest_ss,
        'pd_prune_klowest': pd_prune_klowest_ss
    }

    return make_pruning_figure(pruning_results, title=title, save_format=save_format, save_dir=save_dir)


def plot_pruning_results_fixed_samples(random_pruning_results: SweepResult,
                                       pd_pruning_results: SweepResult,
                                       ft_epoch: int,
                                       ft_col_sel: str,
                                       title: str = '',
                                       save_format: str = '',
                                       save_dir: str = '.',
                                       fraction: float = None, 
                                       keep_highest: bool = None) -> plt.Figure:
    ft_row_sel = ('epoch', ft_epoch)
    col_name = f'{ft_col_sel}-{ft_row_sel[0]}-{ft_row_sel[1]}'
    pruning_results = {}
    if random_pruning_results is not None:
        rand_prune_df = random_pruning_results.get_summary(log_source='val', col_sel=ft_col_sel, row_sel=ft_row_sel)
        if fraction is not None:
            rand_prune_df = rand_prune_df[rand_prune_df['data.sample_selector.kwargs.fraction'] == fraction]
        ret = _get_samples_y_col_series(rand_prune_df, label='rand_prune', y_col_name=col_name)
        pruning_results.update(ret)

    if pd_pruning_results is not None:
        pd_prune_df = pd_pruning_results.get_summary(log_source='val', col_sel=ft_col_sel, row_sel=ft_row_sel)
        if fraction is not None:
            pd_prune_df = pd_prune_df[pd_prune_df['data.sample_selector.kwargs.fraction'] == fraction]
        if keep_highest is not None:
            if keep_highest:
                ret = _get_samples_y_col_series(pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == True], label='pd_prune_khighest', y_col_name=col_name)
                pruning_results.update(ret)
            else:
                ret = _get_samples_y_col_series(pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == False], label='pd_prune_klowest', y_col_name=col_name)
                pruning_results.update(ret)
        else:
            ret = _get_samples_y_col_series(pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == True], label='pd_prune_khighest', y_col_name=col_name)
            pruning_results.update(ret)
            ret = _get_samples_y_col_series(pd_prune_df[pd_prune_df['data.sample_selector.kwargs.keep_highest'] == False], label='pd_prune_klowest', y_col_name=col_name)
            pruning_results.update(ret)

    return make_pruning_figure(pruning_results, title=title, save_format=save_format, save_dir=save_dir)