from typing import List, Tuple
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ml_utilities.plot import get_color_gradient


def plot_barriers(
    instability_df: pd.DataFrame,
    title: str = '',
    save_format: str = '',
    save_dir: str = './',
    select_interpolate_at_idxes: int = -1,
    y_label: str = 'classification error',
    color_gradient_between: Tuple[str, str] = ('#3e1c70', '#feae7c'),
    figsize=(2 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54)
) -> Figure:
    # make a plot of barriers over init_model_idx_k (alpha vs. error/accuracy)
    # create an axes for every dataset
    assert len(
        list(instability_df.index.names)
    ) == 4, f'Instability df must have 4 index levels, but has {len(list(instability_df.index.names))}: {list(instability_df.index.names)}'

    # instability_df contains columns: init_model_idx_k, job, seeds, model_idxes
    plot_df = instability_df.droplevel(level='job', axis=0)

    # get all datasets
    datasets = list(instability_df.columns.get_level_values('datasets').unique())

    # create figure
    f, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)
    f.suptitle(title)
    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    handles_per_dataset = {}
    for ax, dataset in zip(axes, datasets):
        handles_per_dataset[dataset] = []
        if color_gradient_between:
            ax.set_prop_cycle(
                color=get_color_gradient(*color_gradient_between, len(plot_df.groupby(level='init_model_idx_k'))))
        for init_idx_k, init_idx_k_df in plot_df.groupby(level='init_model_idx_k'):
            seed_df = init_idx_k_df.droplevel(0)
            mean_df = seed_df.groupby(by='model_idxes').mean()
            std_df = seed_df.groupby(by='model_idxes').std()  # TODO plot also stddevs
            # TODO allow for different model_idx selection
            interpolation_series = mean_df.iloc[select_interpolate_at_idxes, :]
            interp_sc = interpolation_series[dataset]['interpolation_scores']
            x_vals = interp_sc.index.values
            y_vals = interp_sc.values
            line, = ax.plot(x_vals, y_vals, label=f'rewind_idx_k={init_idx_k}')
            handles_per_dataset[dataset].append(line)
            ax.set_title(dataset)
            ax.grid(alpha=.3)
            ax.set_xlabel('interpolation factors')

    axes[0].set_ylabel(y_label)
    plt.figlegend(frameon=False, loc='lower left', handles=handles_per_dataset[datasets[0]], bbox_to_anchor=(0.9, 0.1))

    if save_format:
        assert title
        f.savefig(f'{str(save_dir)}{title}.{save_format}', dpi=300, bbox_inches='tight')
    return f


def plot_instability(instability_df: pd.DataFrame,
                     title: str = '',
                     select_interpolate_at_idxes: int = -1,
                     save_format: str = '',
                     save_dir: str = './',
                     x_scale: str = 'symlog',
                     y_lim: Tuple[float, float] = (-0.05, 1),
                     figsize=(1.5 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54)):
    assert len(
        list(instability_df.index.names)
    ) == 4, f'Instability df must have 4 index levels, but has {len(list(instability_df.index.names))}: {list(instability_df.index.names)}'

    #* Data handling
    # instability_df contains columns: init_model_idx_k, job, seeds, model_idxes
    # select only instability values
    df = instability_df.droplevel(level='job', axis=0)
    plot_idf = df.xs('instability', axis=1, level='score').droplevel(level='alpha', axis=1)

    # get all datasets
    datasets = list(instability_df.columns.get_level_values('datasets').unique())

    # create mean and std df
    mean_df = plot_idf.groupby(level=['init_model_idx_k', 'model_idxes']).mean()
    std_df = plot_idf.groupby(level=['init_model_idx_k', 'model_idxes']).std()  # TODO plot this too

    # select the row at 'select_interpolate_at_idxes' of level 'init_model_idx_k'
    selected_interpolate_at_model_idxes = {}
    for i, k in mean_df.groupby(level='init_model_idx_k'):
        selected_interpolate_at_model_idxes[i] = k.iloc[select_interpolate_at_idxes]

    plot_mean_df = pd.DataFrame(selected_interpolate_at_model_idxes).transpose().rename_axis('init_model_idx_k', axis=0)

    #* Plotting: create figure
    f, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
    f.suptitle(title)

    handles_per_dataset = {}
    for dataset in datasets:
        x_vals = plot_mean_df[dataset].index.values
        y_vals = plot_mean_df[dataset].values
        line, = ax.plot(x_vals, y_vals, label=dataset, marker='o', markersize=3)
        handles_per_dataset[dataset] = line

    ax.legend()
    ax.grid(alpha=.3)
    ax.set_xlabel('rewind index k')
    ax.set_ylabel('instability')
    ax.set_xscale(x_scale)
    ax.set_ylim(y_lim)

    if save_format:
        assert title
        f.savefig(f'{str(save_dir)}{title}.{save_format}', dpi=300, bbox_inches='tight')
    return f



def plot_distances(distances_df: pd.DataFrame,
                   title: str = '',
                   select_interpolate_at_idxes: int = -1,
                   save_format: str = '',
                   save_dir: str = './',
                   x_scale: str = 'symlog',
                   figsize=(1.5 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)):
    assert len(
        list(distances_df.index.names)
    ) == 4, f'Instability df must have 4 index levels, but has {len(list(distances_df.index.names))}: {list(distances_df.index.names)}'

    #* Data handling
    # instability_df contains columns: init_model_idx_k, job, seeds, model_idxes
    # select only instability values
    df = distances_df.droplevel(level='job', axis=0)

    # create mean and std df
    mean_df = df.groupby(level=['init_model_idx_k', 'model_idxes']).mean()
    std_df = df.groupby(level=['init_model_idx_k', 'model_idxes']).std()  # TODO plot this too

    # select the row at 'select_interpolate_at_idxes' of level 'init_model_idx_k'
    selected_interpolate_at_model_idxes = {}
    for i, k in mean_df.groupby(level='init_model_idx_k'):
        selected_interpolate_at_model_idxes[i] = k.iloc[select_interpolate_at_idxes]

    plot_mean_df = pd.DataFrame(selected_interpolate_at_model_idxes).transpose().rename_axis('init_model_idx_k', axis=0)

    # get all distances
    distances = list(distances_df.columns.get_level_values('distances').unique())

    #* Plotting: create figure
    f, axes = plt.subplots(len(distances), 1, figsize=figsize, sharex=True)
    f.suptitle(title)
    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    handles_per_dataset = {}
    for ax, distance in zip(axes, distances):
        handles_per_dataset[distance] = []
        x_vals = plot_mean_df[distance].index.values
        y_vals = plot_mean_df[distance].values
        line, = ax.plot(x_vals, y_vals, label=distance, marker='o', markersize=3)
        handles_per_dataset[distance].append(line)

        ax.legend()
        ax.grid(alpha=.3)
        ax.set_ylabel(distance)
        ax.set_xscale(x_scale)

    axes[-1].set_xlabel('rewind index k')

    if save_format:
        assert title
        f.savefig(f'{str(save_dir)}{title}.{save_format}', dpi=300, bbox_inches='tight')
    return f
