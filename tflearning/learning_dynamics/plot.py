from typing import List, Tuple
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd


def plot_covariance_statistics(
    cov_stats_df: pd.DataFrame,
    title: str = '',
    statistic_names: List[str] = ['max_eigval', 'inverse_condition_number', 'erank'],
    figsize=(3 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54)
) -> Figure:
    # create figure
    f, axes = plt.subplots(1, len(statistic_names), figsize=figsize)
    f.suptitle(title)
    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    plot_df = cov_stats_df.swaplevel(0,1, axis=1).sort_index(axis=1, level=0)

    datasets = list(plot_df.columns.get_level_values('dataset').unique())


    # plot on axes for every statistic
    for ax, statistic_name in zip(axes, statistic_names):
        # get all datasets
        for bs_nb, df in plot_df.groupby(level=[0,1]):
            df = df.loc[bs_nb][statistic_name]
            bs, nb = bs_nb
            for dataset in datasets:
                x_vals = df[dataset].index.values
                y_vals = df[dataset].values
                ax.plot(x_vals, y_vals, label=f'{dataset}--bs={bs},nb={nb}')

        ax.set_title(statistic_name)
        ax.grid(alpha=.3)
        ax.set_xlabel('checkpoint_idx')

    # TODO from here add figlegend

    return f