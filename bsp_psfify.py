from __future__ import print_function

import ConfigParser
import argparse

import datetime
import inspect
import re
import vip

import numpy as np
from astropy.io import fits
import os

from scipy.stats import sigmaclip, gaussian_kde


def padd_random(vector, pad_width, iaxis, kwargs):
    # print(kwargs)
    # if 'median' in kwargs:
    #     median = kwargs['median']
    # else:
    #     median = 0
    # if 'std' in kwargs:
    #     std = kwargs['std']
    # else:
    #     std = 20
    # vector[:pad_width[0]] = (np.random.rand(pad_width[0]) - 0.5 + median) * std
    # vector[-pad_width[1]:] = (np.random.rand(pad_width[1]) - 0.5 + median) * std

    _kde = kwargs['kde']

    vector[:pad_width[0]] = _kde.resample(vector[:pad_width[0]].shape)
    vector[-pad_width[1]:] = _kde.resample(vector[-pad_width[1]:].shape)

    return vector


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

    if (not args.cx0) and (not args.cy0):
        cx0, cy0 = map(int, get_xy_from_frames_txt(path_obs))
        cx0 *= 2
        cy0 *= 2
    else:
        # note that these are oversampled!
        cy0, cx0 = args.cy0, args.cx0

    print(cy0, cx0, win)

    # open fits:
    with fits.open(os.path.join(path_obs, '100p.fits')) as _hdulist:
        frame = _hdulist[0].data
    image_size = frame.shape
    # print(image_size)

    _trimmed_frame = np.array([frame[cy0 - win: cy0 + win, cx0 - win: cx0 + win]])
    mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta = \
        (vip.var.fit_2dgaussian(_trimmed_frame[0], crop=True, cropsize=40,
                                debug=False, full_output=True))
    _fwhm = np.mean([fwhm_y, fwhm_x])
    if _fwhm < 2:
        _fwhm = 2.0
        # print('Too small, changing to ', _fwhm)
    _fwhm = int(_fwhm)

    # Center the filtered frame
    centered_cube, shy, shx = \
        (vip.calib.cube_recenter_gauss2d_fit(array=_trimmed_frame, xy=(win, win), fwhm=_fwhm,
                                             subi_size=6, nproc=1, full_output=True))

    centered_frame = centered_cube[0]

    # padd with noise to full frame size if necessary:
    padd_num = (image_size[0] - win * 2) / 2
    # padd_num = (512 - _win * 2) / 2
    # print(image_size[0], _win, padd_num)
    # centered_frame = np.lib.pad(centered_frame, [(padd_num, padd_num), (padd_num, padd_num)],
    #                             'constant', constant_values=(-1e-3, -1e-3))
    # print(np.median(centered_frame))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # flattened = np.ndarray.flatten(centered_frame)
    flattened = np.ndarray.flatten(frame)

    # sigma clip the upper end of the distribution to get rid of the source:
    temp = sigmaclip(flattened, 1000.0, 5.)
    # temp = sigmaclip(temp[0], 1000.0, 7.)
    # temp = sigmaclip(temp[0], 1000.0, 7.)
    flattened = temp[0]  # return arr is 1st element

    # estimate the noise distribution from the flattened data
    bandwidth = 0.01
    kde = gaussian_kde(flattened, bw_method=bandwidth / flattened.std(ddof=1))
    # ax.hist(flattened, bins=100, normed=True)
    # ax.plot(np.arange(np.min(flattened), np.max(flattened)),
    #         kde(np.arange(np.min(flattened), np.max(flattened))))
    # plt.show()

    # and padd the cropped frame with noise drawing values from the estimated distribution:
    centered_frame = np.lib.pad(centered_frame, [(padd_num + 1, padd_num),
                                                 (padd_num + 1, padd_num)],
                                padd_random, kde=kde)

    export_fits(os.path.join(path_obs, 'psf.fits'), centered_frame)
