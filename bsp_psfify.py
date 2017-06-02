from __future__ import print_function

import ConfigParser
import argparse

import datetime
import inspect
import re

import numpy as np
from astropy.io import fits
import os


def get_xy_from_frames_txt(_path):
    """
        Get median centroid position for a lucky-pipelined image
    :param _path:
    :return:
    """
    with open(os.path.join(_path, 'frames.txt'), 'r') as _f:
        f_lines = _f.readlines()
    xy = np.array([map(float, l.split()[3:5]) for l in f_lines if l[0] != '#'])

    return np.median(xy[:, 0]), np.median(xy[:, 1])


def export_fits(path, _data, _header=None):
    """
        Save fits file overwriting if exists
    :param path:
    :param _data:
    :param _header:
    :return:
    """
    if _header is not None:
        hdu = fits.PrimaryHDU(_data, header=_header)
    else:
        hdu = fits.PrimaryHDU(_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(path, overwrite=True)


if __name__ == '__main__':
    ''' Bright star pipeline PSFify observation for AIDA deconvolution software '''

    parser = argparse.ArgumentParser(description='stack multiple fits images')

    parser.add_argument('obs', metavar='obs', action='store',
                        help='obs name', type=str)
    parser.add_argument('date', metavar='date', action='store',
                        help='obs date', type=str)
    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose?')
    # parser.add_argument('-p', '--interactive_plot', action='store_true',
    #                     help='plot interactively?')
    parser.add_argument('--cx0', metavar='cx0', action='store', dest='cx0',
                        help='x lock position [pix]', type=int)
    parser.add_argument('--cy0', metavar='cy0', action='store', dest='cy0',
                        help='y lock position [pix]', type=int)
    parser.add_argument('--win', metavar='win', action='store', dest='win',
                        help='window size [pix]', type=int, default=100)
    args = parser.parse_args()

    # script absolute location
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    # parse config file
    config = ConfigParser.RawConfigParser()
    if args.config_file[0] not in ('/', '.'):
        config.read(os.path.join(abs_path, args.config_file))
    else:
        config.read(os.path.join(args.config_file))

    # path to Nick's pipeline:
    path_lucky_pipeline = config.get('Path', 'path_pipe')

    # set date and observation
    date = args.date

    path_obs_list = [[os.path.join(path_lucky_pipeline, date, tag, args.obs), tag] for
                     tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                     os.path.exists(os.path.join(path_lucky_pipeline, date, tag, args.obs))]

    if len(path_obs_list) != 1:
        raise Exception('failed to find pipelined observation')
    else:
        path_obs = path_obs_list[0][0]

    # print(path_obs)

    win = args.win

    y, x = get_xy_from_frames_txt(path_obs)

    print(y, x)
