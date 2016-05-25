from __future__ import print_function
import numpy as np
import os
import argparse
import traceback
import datetime
from astropy.io import fits
from astropy.time import Time
from astropy.time import TimeDelta

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


def dead_times(_path_pipe, _path_output, _date):
    """ RMJC's dead-time calculator

    :param _path_pipe:
    :param _path_output:
    :param _date:
    :return:
    """

    start_times = []
    end_times = []

    for pot in ('high_flux', 'faint', 'zero_flux'):
        path_pot = os.path.join(_path_pipe, _date, pot)
        if os.path.isdir(path_pot):
            source_data = ([d for d in os.listdir(path_pot) if os.path.isdir(os.path.join(path_pot, d))])

            for source in source_data:
                if 'pointing' not in source:
                    try:
                        hdulist = fits.open(os.path.join(path_pot, source, '100p.fits'))
                        header_start = datetime.datetime.strptime(hdulist[0].header['UTSHUT'], '%Y%m%d_%H%M%S.%f')
                        header_end = datetime.datetime.strptime(hdulist[0].header['END_TIME'], '%Y%m%d_%H%M%S.%f')
                    except KeyError:
                        print('Header Error: {:s}'.format(os.path.join(path_pot, source)))
                        continue
                    except IOError:
                        print('Open Error: {:s}'.format(os.path.join(path_pot, source)))
                        continue
                    # succeed? append then:
                    start_times.append(header_start)
                    end_times.append(header_end)

    data_starts = Time(map(str, start_times), format='iso', scale='utc')
    sorted_starts = Time.sort(data_starts)
    data_ends = Time(map(str, end_times), format='iso', scale='utc')
    sorted_ends = Time.sort(data_ends)

    diffs = np.zeros(len(sorted_ends)-1)
    for i in range(len(sorted_ends)-1):
        diffs[i] = TimeDelta(sorted_starts[i+1].jd - sorted_ends[i].jd, format='jd').sec

    fig_names = []
    try:
        fig = plt.figure('1h', figsize=(7, 3.5))
        ax = fig.add_subplot(111)
        ax.hist((diffs[diffs < 3600e7])/60, 10, facecolor=plt.cm.Oranges(0.7))  # "#ff3300"
        ax.set_yscale('log')
        ax.set_xlabel('Minutes between the end of one exposure and the start of the next')  # , fontsize=18)
        ax.set_ylabel('Number of Instances')  # , fontsize=18)
        ax.set_title('Dead Time Distribution (<1hr)', fontsize=20)
        plt.tight_layout()
        f_name = 'deadtime_1hr.{:s}.png'.format(_date)
        fig.savefig(os.path.join(_path_output, f_name), dpi=200)
        fig_names.append(f_name)
    except Exception as err:
        print(err)

    try:
        fig = plt.figure('5m', figsize=(7, 3.5))
        ax = fig.add_subplot(111)
        ax.hist((diffs[diffs < 60*5e7]), 10, facecolor=plt.cm.Oranges(0.7))  # "#ff3300"
        ax.set_yscale('log')
        ax.set_xlabel('Seconds between the end of one exposure and the start of the next')  # , fontsize=18)
        ax.set_ylabel('Number of Instances')  # , fontsize=18)
        ax.set_title('Dead Time Distribution (<5min)', fontsize=20)
        plt.tight_layout()
        f_name = 'deadtime_5min.{:s}.png'.format(_date)
        fig.savefig(os.path.join(_path_output, 'deadtime_5min.{:s}.png'.format(_date)), dpi=200)
        fig_names.append(f_name)
    except Exception as err:
        print(err)

    try:
        fig = plt.figure('1min', figsize=(7, 3.5))
        ax = fig.add_subplot(111)
        ax.hist((diffs[diffs < 60e7]), 10, facecolor=plt.cm.Oranges(0.7))  # "#ff3300"
        ax.set_yscale('log')
        ax.set_xlabel('Seconds between the end of one exposure and the start of the next')  # , fontsize=18)
        ax.set_ylabel('Number of Instances')  # , fontsize=18)
        ax.set_title('Dead Time Distribution (<1min)', fontsize=20)
        plt.tight_layout()
        f_name = 'deadtime_1min.{:s}.png'.format(_date)
        fig.savefig(os.path.join(_path_output, 'deadtime_1min.{:s}.png'.format(_date)), dpi=200)
        fig_names.append(f_name)
    except Exception as err:
        print(err)

    # return diffs for future reference
    return diffs, fig_names


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Becky\'s dead time calculator')

    parser.add_argument('path_pipe', metavar='path_pipe',
                        action='store', help='path to pipelined data.', type=str)
    parser.add_argument('path_output', metavar='path_output',
                        action='store', help='output path.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)

    args = parser.parse_args()

    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')
    date_str = datetime.datetime.strftime(date, '%Y%m%d')

    # create output folder if it does not exist
    path_output_date = os.path.join(args.path_output, date_str)
    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)
    if not os.path.exists(path_output_date):
        os.mkdir(path_output_date)

    try:
        dead_times(_path_pipe=args.path_pipe, _path_output=path_output_date, _date=date_str)
    except Exception as e:
        traceback.print_exc()
        print(e)
        print('Failed to compute dead times for {:s}'.format(date_str))
