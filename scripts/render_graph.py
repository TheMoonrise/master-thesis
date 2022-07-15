"""
Draws graphs from given data.
"""
import os
import json
import argparse

from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

import numpy as np
import matplotlib.pyplot as plt


def source_data(args, property=None):
    """
    Reads all .json files from the source directory.
    If a property is given this property is taken from dicts.
    If no property is given the json must be a list.
    :param args: The command line arguments.
    :param property: The property to collect the data for.
    :return: The data dictionary.
    """
    data = {}
    ignore = (args.ignore or '').split(',')

    # iterate all files in the given directory
    for file in os.listdir(args.source):
        if not file.endswith('.json'): continue
        if file.replace('.json', '') in ignore: continue
        name = file.replace('.json', '').replace('_', ' ').upper()

        with open(os.path.join(args.source, file)) as stream:
            source = json.loads(stream.read())

        if property and property in source:
            data[name] = source[property]

        if not property and type(source) == list:
            data[name] = source

    data = {k: v[args.trim:] for k, v in data.items()}

    if args.length:
        data = {k: v[:args.length] for k, v in data.items()}

    return data


def color(key: str):
    """
    Provides the color for a given key.
    :param key: The key.
    :return: The color as hex string.
    """
    if 'discrete' in key.lower(): return '#64813a'
    if 'meta' in key.lower(): return '#24a19c'
    if 'raa' in key.lower(): return '#d96098'
    if 'ras' in key.lower(): return '#d99061'
    if 'rnb' in key.lower(): return '#315288'

    if 'rng' in key.lower(): return '#5c3388'
    if 'bah' in key.lower(): return '#c4573e'

    return '#000000'


def style_legend(legend):
    """
    Updates the style of a legend.
    :param legend: The legend to style.
    """
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_boxstyle('Round', pad=0.3, rounding_size=1)
    legend.get_frame().set_linewidth(2)


def smooth_data(data, window=2000, smooth=2, scale=.3):
    """
    Smooths the given data by using exponential moving average.
    :param data: The data array to smooth.
    :param window: The window size to compute the error over.
    :param smooth: The factor used in ema computation.
    :param scale: The multiplier by which error is scaled.
    :return: The smoothed data array
    :return: The error margin for the window from std.
    """

    out, err = [], []
    ema, sum1, sum2 = 0, 0, 0

    for i in range(len(data)):
        sum1, sum2 = sum1 + data[i], sum2 + data[i] ** 2
        if i >= window: sum1, sum2 = sum1 - data[i - window], sum2 - data[i - window] ** 2

        k = smooth / (i + 100)
        ema = data[i] * k + ema * (1 - k)

        out.append(ema)

        cnt = min(window, i + 1)
        var = sum2 / cnt - sum1 / cnt ** 2
        err.append(np.sqrt(max(var, 1e-9)) * (cnt / window) * scale)

    return np.array(out), np.array(err)


def smooth_line(data, x, multiplier=10, k=3):
    """
    Smooths out the given curve.
    :param data: The data points for the original curve.
    :param x: The x coordinates of the points.
    :param multiplier: The multiplier to increase the point count by.
    :param k: The interpolation method.
    :return: The new x coordinates.
    :return: The smoothed data points.
    """
    x_smooth = np.linspace(np.min(x), np.max(x), len(x) * multiplier)
    interps = make_interp_spline(x, data, k)
    return x_smooth, interps(x_smooth)


def plot_mean_error(args, data, figure, axes: plt.Axes, line='-', plot_err=True, plot_legend=True):
    """
    Plots a mean error graph.
    :param args: The command line arguments.
    :param data: The data to plot the graph for.
    :param figure: The figure to plot onto.
    :param axes: The axes to use for the plot.
    :param line: The line rendering style.
    :param plot_err: Whether to plot the error 
    :param plot_legend: Whether to plot a legend.
    """
    for k, v in data.items():
        smooth, err = smooth_data(v)
        x = np.arange(len(smooth))

        label = k if plot_legend else None
        axes.plot(x, smooth, color=color(k), label=label, ls=line)

        if not plot_err: continue
        axes.fill_between(x, smooth - err, smooth + err, alpha=0.2, color=color(k))

    axes.set_ylabel(args.ylabel or 'Economic Returns (Log Returns)')
    axes.set_xlabel('Training Iterations')
    axes.spines['bottom'].set_visible(True)
    axes.grid(color='#a79f9a', axis='y')
    style_legend(axes.legend())


def plot_violin(args, data, figure, axes: plt.Axes):
    """
    Plots a boxplot with violin
    :param args: The command line arguments.
    :param data: The data to plot the graph for.
    :param figure: The figure to plot onto.
    :param axes: The axes to use for the plot.
    """
    keys = list(data.keys())
    values = list(data.values())

    props = dict(marker='D', markeredgecolor='black', markerfacecolor='#faeee7')

    comp_vio = axes.violinplot(values, showextrema=False, widths=0.75)
    comp_box = axes.boxplot(values, widths=0.4, showmeans=True, showfliers=False, meanprops=props)

    for i, x in enumerate(comp_vio['bodies']):
        x.set_color(color(keys[i]))

    for x in comp_box['medians']:
        x.set_color('k')

    for x in comp_box['boxes'] + comp_box['medians'] + comp_box['whiskers'] + comp_box['caps']:
        x.set_linewidth(2)

    axes.set_xticks(np.arange(1, len(keys) + 1))
    axes.set_xticklabels(labels=keys)
    axes.set_ylabel(args.ylabel or 'Economic Returns (Log Returns)')
    axes.grid(color='#a79f9a', axis='y')
    axes.tick_params(axis='x', which='both', bottom=False, top=False)


def plot_decisions(args, data, figure, axes: plt.Axes, episode_id):
    """
    Plots a graph showing trading decisions of the agent.
    :param args: The command line arguments.
    :param data: The data to plot the graph for.
    :param figure: The figure to plot onto.
    :param axes: The axes to use for the plot.
    """

    # pick the episode from any of the provided runs
    # all runs must have the same seed for the plot to make sense
    episode = np.array(data[list(data.keys())[0]][episode_id])
    x = np.arange(episode.shape[0])

    # close to close value every 7 indices starting at 19
    # c2c = episode[:, 19:: 7]
    c2c = episode[:, 1:-1]

    # c2c only hods change in value? sum over episode
    for i in range(1, c2c.shape[0]): c2c[i] += c2c[i - 1]

    # plot the coin price changes
    for i in range(c2c.shape[1]):
        xs, ys = smooth_line(c2c[:, i], x, 4)
        axes.plot(xs, ys, color='#a79f9a', lw=1)

    # plot purchasing behavior for all included runs
    for k, episodes in data.items():
        episode = np.array(episodes[episode_id])

        # current allocation is one hot encoded at index 2 to 13
        # asset = np.argmax(episode[:, 2: 13], axis=1)
        asset = episode[:, 0]

        # pick the values from the current asset
        values = np.zeros_like(asset, dtype=np.float64)
        for i in range(len(asset)): values[i] = c2c[i, int(asset[i])]

        tracker = (asset[0], 0)

        # plot a line for holding and dots for brief purchases
        for i in range(1, len(asset)):
            if asset[i] == tracker[0] and i < len(asset) - 1: continue

            if i - tracker[1] == 1:
                axes.scatter(x[i - 1], values[i - 1], s=7, color=color(k), alpha=1, zorder=4)

            else:
                xs, ys = x[tracker[1]:i], values[tracker[1]:i]
                if i - tracker[1] > 3: xs, ys = smooth_line(values[tracker[1]:i], x[tracker[1]:i], 4)
                axes.plot(xs, ys, color=color(k), zorder=3)

            tracker = (asset[i], i)

    axes.set_ylabel('Price Change (ETH)')
    axes.set_xlabel('Step')
    axes.spines['bottom'].set_visible(True)
    axes.grid(color='#a79f9a', axis='y')

    # define a legend with custom lines
    lines = [Line2D([0], [0], color='#a79f9a', lw=4)] + [Line2D([0], [0], color=color(k), lw=4) for k in data.keys()]
    labels = ['Coins'] + [k for k in data.keys()]
    style_legend(axes.legend(lines, labels))


def setup_figure(args):
    fig, ax = plt.subplots(figsize=(10, 6))
    if args.log: ax.set_yscale('log')

    # ax.set_title(args.title or 'Mock Graph', fontsize='16')

    for v in ax.spines.values(): v.set_visible(False)  # remove the bounding box
    for v in ax.spines.values(): v.set_color('#a79f9a')

    ax.tick_params(axis='both', color='#a79f9a', labelcolor='#484442')
    # ax.xaxis.label.set_color('#a79f9a')
    return fig, ax


def save_file(args, counter=-1):
    if args.out:
        path = args.out
        if counter >= 0: path += '_' + str(counter)
        plt.savefig(path, bbox_inches='tight')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('type', type=str)
    parse.add_argument('source', type=str)
    parse.add_argument('--property', type=str)
    parse.add_argument('--overlay', type=str)
    parse.add_argument('--length', type=int)
    parse.add_argument('--title', type=str)
    parse.add_argument('--ylabel', type=str)
    parse.add_argument('--out', type=str)
    parse.add_argument('--episode', type=str)
    parse.add_argument('--ignore', type=str)
    parse.add_argument('--log', action='store_true')
    parse.add_argument('--trim', type=int, default=0)

    args = parse.parse_args()

    # read the data from the source folder
    data = source_data(args, args.property)

    # prepare the figure and axes
    # install the font on the system
    # go to users/.matplotlib and delete font cache
    plt.rcParams['font.family'] = 'Noto Sans Math'
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2

    fig, ax = setup_figure(args)

    if args.type == 'mean-error':
        print('Plotting mean error')
        plot_mean_error(args, data, fig, ax)
        if (args.overlay): plot_mean_error(args, source_data(args, args.overlay), fig, ax, '--', True, False)
        save_file(args)
        plt.show()

    if args.type == 'mean':
        print('Plotting mean')
        plot_mean_error(args, data, fig, ax, plot_err=False)
        if (args.overlay): plot_mean_error(args, source_data(args, args.overlay), fig, ax, '--', False, False)
        save_file(args)
        plt.show()

    if args.type == 'violin':
        print('Plotting violin plot')
        plot_violin(args, data, fig, ax)
        save_file(args)
        plt.show()

    if args.type == 'decisions':

        if '-' not in args.episode:
            print('Plotting decisions, episode', args.episode)
            plot_decisions(args, data, fig, ax, int(args.episode))
            save_file(args)
            plt.show()

        else:
            fromto = [int(x) for x in args.episode.split('-')]
            for e in range(fromto[0], fromto[1]):
                print('Plotting decisions, episode', e)
                plt.close()
                fig, ax = setup_figure(args)
                plot_decisions(args, data, fig, ax, e)
                save_file(args, e)
