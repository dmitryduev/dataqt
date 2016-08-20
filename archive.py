"""
    Data Archiving for Robo-AO

    Generates stuff to be displayed on the archive
    Updates the database

    Dmitry Duev (Caltech) 2016
"""
from __future__ import print_function

import ConfigParser
import argparse
import inspect
import traceback
import os
import logging
import datetime
import pytz
import time
from astropy.io import fits
from pymongo import MongoClient
import sys
import re
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import ast
from scipy.optimize import fmin
from astropy.modeling import models, fitting
import sewpy

from skimage import exposure, img_as_float
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import matplotlib.pyplot as plt
import seaborn as sns

from beckys import pca

# import numba
from huey import RedisHuey
huey = RedisHuey('roboao.archive', result_store=True)

# set up plotting
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


# Scale bars
class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        loc:
            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4,
            'right'        : 5,
            'center left'  : 6,
            'center right' : 7,
            'lower center' : 8,
            'upper center' : 9,
            'center'       : 10
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, 0, fc='none', color='white', lw=3))

        self.txt_label = TextArea(label, dict(color='white', size='x-large', weight='normal'),
                                  minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon)


@huey.task()
# @numba.jit
def job_pca(_config, _date, _out_path, _x=None, _y=None, _drizzled=True):
    try:
        pass
    #     # TODO: first run the Strehl calculator on the 100p
    #     trimmed_frame = (trim_frame(_path=_config['path_pipe'], _fits_name='100p.fits',
    #                                 _win=_config['win'], _method='sextractor',
    #                                 _x=_x, _y=_y, _drizzled=_drizzled))
    #
    #     # Check of observation passes quality check:
    #
    #     try:
    #         cy1, cx1 = np.unravel_index(trimmed_frame.argmax(), trimmed_frame.shape)
    #         core, halo = bad_obs_check(trimmed_frame[cy1 - 30:cy1 + 30 + 1, cx1 - 30:cx1 + 30 + 1],
    #                                    ps=plate_scale)
    #         # f_handle = file('/Data2/becky/compile_data/core_and_halo.txt', 'a')
    #         # np.savetxt(f_handle, np.array(['\n'+path_sou, core, halo]), newline=" ",fmt="%s")
    #         # f_handle.close()
    #     except:
    #         core = 0.14
    #         halo = 1.0
    #
    #     # run PCA
    #     if core > 0.14 and halo < 1.0:
    #         # run on lucky-pipelined image
    #         output = pca(_trimmed_frame=trimmed_frame, _win=_config['pca']['win'], _sou_name=sou_name,
    #                      _sou_dir=sou_dir, _out_path=_out_path,
    #                      _library=psf_reference_library,
    #                      _library_names_short=psf_reference_library_short_names,
    #                      _fwhm=fwhm, _plsc=plsc, _sigma=sigma, _nrefs=nrefs, _klip=klip)
    #     else:
    #         # run on faint-pipelined image
    #         pass
    except Exception as _e:
        print(_e)
        return False

    return True


@huey.task()
def job_strehl(_path_in, _fits_name, _obs, _path_out, _plate_scale, _Strehl_factor):

    # do the work
    try:
        img, x, y = trim_frame(_path_in, _fits_name=_fits_name,
                               _win=100, _method='sextractor',
                               _x=None, _y=None, _drizzled=True)
        core, halo = bad_obs_check(img, ps=_plate_scale)

        boxsize = int(round(3. / _plate_scale))
        SR, FWHM, box = Strehl_calculator(img, _Strehl_factor[0], _plate_scale, boxsize)

    except Exception as _e:
        print(_obs, _e)
        # traceback.print_exc()
        # x, y = 0, 0
        # core, halo = 0, 999
        # SR, FWHM = 0, 0
        return False

    if core >= 0.14 and halo <= 1.0:
        flag = 'OK'
    else:
        flag = 'BAD?'

    # print(core, halo, SR*100, FWHM)

    # dump results to disk
    if not os.path.exists(os.path.join(_path_out, _obs)):
        os.mkdir(os.path.join(_path_out, _obs))

    # save box around selected object:
    hdu = fits.PrimaryHDU(box)
    hdu.writeto(os.path.join(_path_out, _obs, '{:s}_box.fits'.format(_obs)), clobber=True)

    # save the Strehl data to txt-file:
    with open(os.path.join(_path_out, _obs, '{:s}_strehl.txt'.format(_obs)), 'w') as _f:
        _f.write('# lock_x[px] lock_y[px] core["] halo["] SR[%] FWHM["] flag\n')
        output_entry = '{:d} {:d} {:.5f} {:.5f} {:.5f} {:.5f} {:s}\n'.\
            format(x, y, core, halo, SR * 100, FWHM, flag)
        _f.write(output_entry)

    return True


@huey.task()
# @numba.jit
def job_bogus(_obs):
    tic = time.time()
    a = 0
    for i in range(100):
        for j in range(100):
            for k in range(500):
                a += 3**2
    print('It took {:.2f} s to finish the job on {:s}'.format(time.time() - tic, _obs))

    return True


def utc_now():
    return datetime.datetime.now(pytz.utc)


def naptime(nap_time_start, nap_time_stop):
    """
        Return time to sleep in seconds for the archiving engine
        before waking up to rerun itself.
         In the daytime, it's 1 hour
         In the nap time, it's nap_time_start_utc - utc_now()
    :return:
    """
    now_local = datetime.datetime.now()
    # TODO: finish!


def load_fits(fin):
    with fits.open(fin) as _f:
        scidata = _f[0].data
    return scidata


def load_strehl(fin):
    with open(fin) as _f:
        f_lines = _f.readlines()
    _tmp = f_lines[1].split()
    _x = int(_tmp[0])
    _y = int(_tmp[1])
    _core = float(_tmp[2])
    _halo = float(_tmp[3])
    _SR = float(_tmp[4])
    _FWHM = float(_tmp[5])
    _flag = _tmp[6]

    return _x, _y, _core, _halo, _SR, _FWHM, _flag


def scale_image(image, correction='local'):
    """

    :param image:
    :param correction: 'local', 'log', or 'global'
    :return:
    """
    # scale image for beautification:
    scidata = deepcopy(image)
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata / norm * 65535)

    # add more contrast to the image:
    if correction == 'log':
        return exposure.adjust_log(img_as_float(scidata/norm) + 1, 1)
    elif correction == 'global':
        p_1, p_2 = np.percentile(scidata, (5, 100))
        return exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
    elif correction == 'local':
        # perform local histogram equalization instead:
        return exposure.equalize_adapthist(scidata, clip_limit=0.03)
    else:
        raise Exception('Contrast correction option not recognized')


def generate_pca_images(_out_path, _sou_dir, _preview_img, _cc,
                        _fow_x=36, _pix_x=1024, _drizzle=True):
    """
            Generate preview images for the pca pipeline

        :param _out_path:
        :param _sou_dir:
        :param _preview_img:
        :param _cc:
        :param _fow_x: full FoW in arcseconds in the x direction
        :param _pix_x: original full frame size in pixels
        :param _drizzle: drizzle on or off?

        :return:
        """
    try:
        ''' plot psf-subtracted image '''
        plt.close('all')
        fig = plt.figure(_sou_dir)
        fig.set_size_inches(3, 3, forward=False)
        # ax = fig.add_subplot(111)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(_preview_img, cmap='gray', origin='lower', interpolation='nearest')
        # add scale bar:
        # draw a horizontal bar with length of 0.1*x_size
        # (ax.transData) with a label underneath.
        bar_len = _preview_img.shape[0] * 0.1
        bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / 2) if _drizzle \
            else '{:.1f}'.format(bar_len * _fow_x / _pix_x)
        asb = AnchoredSizeBar(ax.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax.add_artist(asb)

        # save figure
        fig.savefig(os.path.join(_out_path, _sou_dir + '_pca.png'), dpi=300)

        ''' plot the contrast curve '''
        plt.close('all')
        fig = plt.figure('Contrast curve for {:s}'.format(_sou_dir), figsize=(8, 3.5), dpi=200)
        ax = fig.add_subplot(111)
        ax.set_title(_sou_dir)  # , fontsize=14)
        ax.plot(_cc[:, 0], -2.5 * np.log10(_cc[:, 1]), 'k-', linewidth=2.5)
        ax.set_xlim([0.2, 1.45])
        ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
        ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
        ax.set_ylim([0, 8])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.grid(linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(_out_path, _sou_dir + '_contrast_curve.png'), dpi=200)
    except Exception as _e:
        print(_e)
        return False

    return True


def get_xy_from_frames_txt(_path):
    """
        Get median centroid position
    :param _path:
    :return:
    """
    with open(os.path.join(_path, 'frames.txt'), 'r') as _f:
        f_lines = _f.readlines()
    xy = np.array([map(float, l.split()[3:5]) for l in f_lines if l[0] != '#'])

    return np.median(xy[:, 0]), np.median(xy[:, 1])


# detect observatiosn which are bad because of being too faint
# 1. make a radial profile, snip out the central 3 pixels
#    (removing the ones which are affected by photon noise)
# 2. measure the width of the remaining flux
# 3. check for too small a width (=> no flux) or too large a width (=> crappy performance)
def gaussian(p, x):
    return p[0] + p[1] * (np.exp(-x * x / (2.0 * p[2] * p[2])))


def moffat(p, x):
    base = 0.0
    scale = p[1]
    fwhm = p[2]
    beta = p[3]

    if np.power(2.0, (1.0 / beta)) > 1.0:
        alpha = fwhm / (2.0 * np.sqrt(np.power(2.0, (1.0 / beta)) - 1.0))
        return base + scale * np.power(1.0 + ((x / alpha) ** 2), -beta)
    else:
        return 1.0


def residuals(p, x, y):
    res = 0.0
    for a, b in zip(x, y):
        res += np.fabs(b - moffat(p, a))

    return res


def bad_obs_check(p, return_halo=True, ps=0.0175797):
    pix_rad = []
    pix_vals = []
    core_pix_rad = []
    core_pix_vals = []

    # Icy, Icx = numpy.unravel_index(p.argmax(), p.shape)

    for x in range(p.shape[1] / 2 - 20, p.shape[1] / 2 + 20 + 1):
        for y in range(p.shape[0] / 2 - 20, p.shape[0] / 2 + 20 + 1):
            r = np.sqrt((x - p.shape[1] / 2) ** 2 + (y - p.shape[0] / 2) ** 2)
            if r > 3:  # remove core
                pix_rad.append(r)
                pix_vals.append(p[y][x])
            else:
                core_pix_rad.append(r)
                core_pix_vals.append(p[y][x])

    try:
        if return_halo:
            p0 = [0.0, np.max(pix_vals), 20.0, 2.0]
            p = fmin(residuals, p0, args=(pix_rad, pix_vals), maxiter=1000000, maxfun=1000000,
                     ftol=1e-3, xtol=1e-3, disp=False)

        p0 = [0.0, np.max(core_pix_vals), 5.0, 2.0]
        core_p = fmin(residuals, p0, args=(core_pix_rad, core_pix_vals), maxiter=1000000, maxfun=1000000,
                      ftol=1e-3, xtol=1e-3, disp=False)

        # Palomar PS = 0.021, KP PS = 0.0175797
        _core = core_p[2] * ps
        if return_halo:
            _halo = p[2] * ps
            return _core, _halo
        else:
            return _core

    except OverflowError:
        _core = 0
        _halo = 999

        if return_halo:
            return _core, _halo
        else:
            return _core


def log_gauss_score(_x, _mu=1.27, _sigma=0.17):
    """
        _x: pixel for pixel in [1,2048] - source FWHM.
            has a max of 1 around 35 pix, drops fast to the left, drops slower to the right
    """
    return np.exp(-(np.log(np.log(_x)) - _mu)**2 / (2*_sigma**2))  # / 2


def gauss_score(_r, _mu=0, _sigma=512):
    """
        _r - distance from centre to source in pix
    """
    return np.exp(-(_r - _mu)**2 / (2*_sigma**2))  # / 2


def rho(x, y, x_0=1024, y_0=1024):
    return np.sqrt((x-x_0)**2 + (y-y_0)**2)


def trim_frame(_path, _fits_name, _win=100, _method='sextractor', _x=None, _y=None, _drizzled=True):
    """

    :param _path: path
    :param _fits_name: fits-file name
    :param _win: window width
    :param _method: from 'frames.txt', using 'sextractor', a simple 'max', or 'manual'
    :param _x: source x position -- if known in advance
    :param _y: source y position -- if known in advance
    :param _drizzled: was it drizzled?

    :return: cropped image
    """
    scidata = fits.open(os.path.join(_path, _fits_name))[0].data

    if _method == 'sextractor':
        # extract sources
        sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                                "A_IMAGE", "B_IMAGE", "FWHM_IMAGE", "FLAGS"],
                                config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
                                sexpath="sex")

        out = sew(os.path.join(_path, _fits_name))
        # sort by FWHM
        out['table'].sort('FWHM_IMAGE')
        # descending order
        out['table'].reverse()

        # print(out['table'])  # This is an astropy table.

        # get first 10 and score them:
        scores = []
        # maximum width of a fix Gaussian. Real sources usually have larger 'errors'
        gauss_error_max = [np.max([sou['A_IMAGE'] for sou in out['table'][0:10]]),
                           np.max([sou['B_IMAGE'] for sou in out['table'][0:10]])]
        for sou in out['table'][0:10]:
            if sou['FWHM_IMAGE'] > 1:
                score = (log_gauss_score(sou['FWHM_IMAGE']) +
                         gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE'])) +
                         np.mean([sou['A_IMAGE'] / gauss_error_max[0],
                                  sou['B_IMAGE'] / gauss_error_max[1]])) / 3.0
            else:
                score = 0  # it could so happen that reported FWHM is 0
            scores.append(score)

        # print('scores: ', scores)

        N_sou = len(out['table'])
        # do not crop large planets and crowded fields
        if N_sou != 0 and N_sou < 30:
            # sou_xy = [out['table']['X_IMAGE'][0], out['table']['Y_IMAGE'][0]]
            best_score = np.argmax(scores) if len(scores) > 0 else 0
            # sou_size = np.max((int(out['table']['FWHM_IMAGE'][best_score] * 3), 90))
            # print(out['table']['XPEAK_IMAGE'][best_score], out['table']['YPEAK_IMAGE'][best_score])
            # print(get_xy_from_frames_txt(_path))
            x = out['table']['YPEAK_IMAGE'][best_score]
            y = out['table']['XPEAK_IMAGE'][best_score]
            scidata_cropped = scidata[x - _win: x + _win + 1,
                                      y - _win: y + _win + 1]
        else:
            # use a simple max instead:
            x, y = np.unravel_index(scidata.argmax(), scidata.shape)
            scidata_cropped = scidata[x - _win: x + _win + 1,
                                      y - _win: y + _win + 1]
    elif _method == 'max':
        x, y = np.unravel_index(scidata.argmax(), scidata.shape)
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    elif _method == 'frames.txt':
        y, x = get_xy_from_frames_txt(_path)
        if _drizzled:
            x *= 2.0
            y *= 2.0
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    elif _method == 'manual' and _x is not None and _y is not None:
        x, y = _x, _y
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    else:
        raise Exception('unrecognized trimming method.')

    return scidata_cropped, x, y


def makebox(array, halfwidth, peak1, peak2):
    boxside1a = peak1 - halfwidth
    boxside1b = peak1 + halfwidth
    boxside2a = peak2 - halfwidth
    boxside2b = peak2 + halfwidth

    box = array[boxside1a:boxside1b, boxside2a:boxside2b]
    box_fraction = np.sum(box) / np.sum(array)
    # print('box has: {:.2f}% of light'.format(box_fraction * 100))

    return box, box_fraction


def Strehl_calculator(image_data, _Strehl_factor, _plate_scale, _boxsize):

    """ Calculates the Strehl ratio of an image
    Inputs:
        - image_data: image data
        - Strehl_factor: from model PSF
        - boxsize: from model PSF
        - plate_scale: plate scale of telescope in arcseconds/pixel
    Output:
        Strehl ratio (as a decimal)

        """

    ##################################################
    # normalize real image PSF by the flux in some radius
    ##################################################

    ##################################################
    #  Choose radius with 95-99% light in model PSF ##
    ##################################################

    # find peak image flux to center box around
    peak_ind = np.where(image_data == np.max(image_data))
    peak1, peak2 = peak_ind[0][0], peak_ind[1][0]
    # print("max intensity =", np.max(image_data), "located at:", peak1, ",", peak2, "pixels")

    # find array within desired radius
    box_roboao, box_roboao_fraction = makebox(image_data, round(_boxsize / 2.), peak1, peak2)
    # print("size of box", np.shape(box_roboao))

    # sum the fluxes within the desired radius
    total_box_flux = np.sum(box_roboao)
    # print("total flux in box", total_box_flux)

    # normalize real image PSF by the flux in some radius:
    image_norm = image_data / total_box_flux

    ########################
    # divide normalized peak image flux by strehl factor
    ########################

    image_norm_peak = np.max(image_norm)
    # print("normalized peak", image_norm_peak)

    #####################################################
    # ############# CALCULATE STREHL RATIO ##############
    #####################################################
    Strehl_ratio = image_norm_peak / _Strehl_factor
    # print('\n----------------------------------')
    # print("Strehl ratio", Strehl_ratio * 100, '%')
    # print("----------------------------------")

    y, x = np.mgrid[:len(box_roboao), :len(box_roboao)]
    max_inds = np.where(box_roboao == np.max(box_roboao))

    g_init = models.Gaussian2D(amplitude=1., x_mean=max_inds[1][0], y_mean=max_inds[0][0],
                               x_stddev=1., y_stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y, box_roboao)

    sig_x = g.x_stddev[0]
    sig_y = g.y_stddev[0]

    FWHM = 2.3548 * np.mean([sig_x, sig_y])
    fwhm_arcsec = FWHM * _plate_scale

    # print('image FWHM: {:.5f}\"\n'.format(fwhm_arcsec))

    return Strehl_ratio, fwhm_arcsec, box_roboao


def empty_db_record():
    time_now_utc = utc_now()
    return {
            '_id': None,
            'date_added': time_now_utc,
            'name': None,
            'alternative_names': [],
            'science_program': {
                'program_id': None,
                'program_PI': None,
                'distributed': None
            },
            'date_utc': None,
            'telescope': None,
            'camera': None,
            'filter': None,
            'exposure': None,
            'magnitude': None,
            'coordinates': {
                'epoch': None,
                'radec': None,
                'radec_str': None,
                'azel': None
            },
            'pipelined': {
                'automated': {
                    'status': {
                        'done': False,
                        'preview': False
                    },
                    'location': [],
                    'classified_as': None,
                    'fits_header': {},
                    'strehl': {
                        'status': {
                            'done': False,
                            'retries': 0
                        },
                        'lock_position': None,
                        'ratio_percent': None,
                        'core_arcsec': None,
                        'halo_arcsec': None,
                        'fwhm_arcsec': None,
                        'flag': None,
                        'last_modified': time_now_utc
                    },
                    'pca': {
                        'status': {
                            'done': False,
                            'preview': False,
                            'retries': 0
                        },
                        'location': [],
                        'lock_position': None,
                        'contrast_curve': None,
                        'last_modified': time_now_utc
                    },
                    'last_modified': time_now_utc
                },
                'faint': {
                    'status': {
                        'done': False,
                        'preview': False,
                        'retries': 0
                    },
                    'location': [],
                    'strehl': {
                        'status': {
                            'done': False,
                            'retries': 0
                        },
                        'lock_position': None,
                        'ratio_percent': None,
                        'core_arcsec': None,
                        'halo_arcsec': None,
                        'fwhm_arcsec': None,
                        'flag': None,
                        'last_modified': time_now_utc
                    },
                    'pca': {
                        'status': {
                            'done': False,
                            'preview': False,
                            'retries': 0
                        },
                        'location': [],
                        'lock_position': None,
                        'contrast_curve': None,
                        'last_modified': time_now_utc
                    },
                    'last_modified': time_now_utc
                },

            },

            'seeing': {
                'median': None,
                'mean': None,
                'last_modified': time_now_utc
            },
            'bzip2': {
                'location': [],
                'last_modified': time_now_utc
            },
            'raw_data': {
                'location': [],
                'data': [],
                'last_modified': time_now_utc
            },
            'comment': None
        }


def set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG, _mode='w'):
    """ Set up logging

    :param _path:
    :param _name:
    :param _level: DEBUG, INFO, etc.
    :param _mode: overwrite log-file or append: w or a
    :return: logger instance
    """

    if not os.path.exists(_path):
        os.makedirs(_path)
    utc_now = datetime.datetime.utcnow()

    # http://www.blog.pythonlibrary.org/2012/08/02/python-101-an-intro-to-logging/
    _logger = logging.getLogger(_name)

    _logger.setLevel(_level)
    # create the logging file handler
    fh = logging.FileHandler(os.path.join(_path,
                                          '{:s}.{:s}.log'.format(_name, utc_now.strftime('%Y%m%d'))),
                             mode=_mode)
    logging.Formatter.converter = time.gmtime

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    _logger.addHandler(fh)

    return _logger


def get_config(_config_file='config.ini'):
    """ Get config data

    :return:
    """
    config = ConfigParser.RawConfigParser()

    if _config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(abs_path, _config_file)):
            config.read(os.path.join(abs_path, _config_file))
            if len(config.read(os.path.join(abs_path, _config_file))) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')
    else:
        if os.path.isfile(_config_file):
            config.read(_config_file)
            if len(config.read(_config_file)) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')

    _config = dict()
    # planetary program number (do no crop planetary images!)
    _config['program_num_planets'] = int(config.get('Programs', 'planets'))
    # path to raw data:
    _config['path_raw'] = config.get('Path', 'path_raw')
    # path to lucky-pipeline data:
    _config['path_pipe'] = config.get('Path', 'path_pipe')
    # path to pca-pipeline data:
    _config['path_pca'] = config.get('Path', 'path_pca')
    # path to Strehl data:
    _config['path_strehl'] = config.get('Path', 'path_strehl')
    # path to seeing plots:
    _config['path_seeing'] = config.get('Path', 'path_seeing')
    # website data dwelling place:
    _config['path_to_website_data'] = config.get('Path', 'path_to_website_data')

    # path to model PSFs:
    _config['path_model_psf'] = config.get('Path', 'path_model_psf')

    # telescope data (voor, o.a., Strehl computation)
    _tmp = ast.literal_eval(config.get('Strehl', 'Strehl_factor'))
    _config['telescope_data'] = dict()
    for telescope in 'Palomar', 'KittPeak':
        _config['telescope_data'][telescope] = ast.literal_eval(config.get('Strehl', telescope))
        _config['telescope_data'][telescope]['Strehl_factor'] = _tmp[telescope]

    # pca pipeline
    _config['pca'] = dict()
    # path to PSF library:
    path_psf_reference_library = config.get('Path', 'path_psf_reference_library')
    path_psf_reference_library_short_names = config.get('Path', 'path_psf_reference_library_short_names')
    _config['pca']['psf_reference_library'] = fits.open(path_psf_reference_library)[0].data
    _config['pca']['psf_reference_library_short_names'] = np.genfromtxt(path_psf_reference_library_short_names,
                                                                 dtype='|S')

    _config['pca']['win'] = int(config.get('PCA', 'win'))
    _config['pca']['plate_scale'] = float(config.get('PCA', 'plate_scale'))
    _config['pca']['sigma'] = float(config.get('PCA', 'sigma'))
    _config['pca']['nrefs'] = float(config.get('PCA', 'nrefs'))
    _config['pca']['klip'] = float(config.get('PCA', 'klip'))

    _config['pca']['planets_prog_num'] = int(config.get('Programs', 'planets'))

    # database access:
    _config['mongo_host'] = config.get('Database', 'host')
    _config['mongo_port'] = int(config.get('Database', 'port'))
    _config['mongo_db'] = config.get('Database', 'db')
    _config['mongo_collection_obs'] = config.get('Database', 'collection_obs')
    _config['mongo_collection_pwd'] = config.get('Database', 'collection_pwd')
    _config['mongo_user'] = config.get('Database', 'user')
    _config['mongo_pwd'] = config.get('Database', 'pwd')

    # server ip addresses
    _config['analysis_machine_external_host'] = config.get('Server', 'analysis_machine_external_host')
    _config['analysis_machine_external_port'] = config.get('Server', 'analysis_machine_external_port')

    # consider data from:
    _config['archiving_start_date'] = datetime.datetime.strptime(
                            config.get('Auxiliary', 'archiving_start_date'), '%Y/%m/%d')
    # how many times to try to rerun pipelines:
    _config['max_pipelining_retries'] = config.get('Auxiliary', 'max_pipelining_retries')

    return _config


def connect_to_db(_config, _logger=None):
    """ Connect to the mongodb database

    :return:
    """
    try:
        client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'])
        _db = client[_config['mongo_db']]
        if _logger is not None:
            _logger.debug('Successfully connected to the Robo-AO database at {:s}:{:d}'.
                          format(_config['mongo_host'], _config['mongo_port']))
    except Exception as _e:
        _db = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to connect to the Robo-AO database at {:s}:{:d}'.
                          format(_config['mongo_host'], _config['mongo_port']))
    try:
        _db.authenticate(_config['mongo_user'], _config['mongo_pwd'])
        if _logger is not None:
            _logger.debug('Successfully authenticated with the Robo-AO database at {:s}:{:d}'.
                          format(_config['mongo_host'], _config['mongo_port']))
    except Exception as _e:
        _db = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Authentication failed for the Robo-AO database at {:s}:{:d}'.
                          format(_config['mongo_host'], _config['mongo_port']))
    try:
        _coll = _db[_config['mongo_collection_obs']]
        # cursor = coll.find()
        # for doc in cursor:
        #     print(doc)
        if _logger is not None:
            _logger.debug('Using collection {:s} with obs data in the database'.
                          format(_config['mongo_collection_obs']))
    except Exception as _e:
        _coll = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with obs data in the database'.
                          format(_config['mongo_collection_obs']))
    try:
        _coll_usr = _db[_config['mongo_collection_pwd']]
        # cursor = coll.find()
        # for doc in cursor:
        #     print(doc)
        if _logger is not None:
            _logger.debug('Using collection {:s} with user access credentials in the database'.
                          format(_config['mongo_collection_pwd']))
    except Exception as _e:
        _coll_usr = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with user access credentials in the database'.
                          format(_config['mongo_collection_pwd']))
    try:
        # build dictionary program num -> pi name
        cursor = _coll_usr.find()
        _program_pi = {}
        for doc in cursor:
            # handle admin separately
            if doc['_id'] == 'admin':
                continue
            _progs = doc['programs']
            for v in _progs:
                _program_pi[str(v)] = doc['_id'].encode('ascii', 'ignore')
                # print(program_pi)
    except Exception as _e:
        _program_pi = None
        if _logger is not None:
            _logger.error(_e)

    return _db, _coll, _program_pi


def get_fits_header(fits_file):
    """
        Get fits-file header
    :param fits_file:
    :return:
    """
    # read fits:
    with fits.open(os.path.join(fits_file)) as hdulist:
        # header:
        header = OrderedDict()
        for _entry in hdulist[0].header.cards:
            header[_entry[0]] = _entry[1:]

    return header


def check_pipe_automated(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been automatically pipelined
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    if not _select['pipelined']['automated']['status']['done']:
        # check if actually processed
        path_obs_list = [[os.path.join(_config['path_pipe'], _date, tag, _obs), tag] for
                         tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                         os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
        # yes?
        if len(path_obs_list) == 1:
            # this also considers the pathological case when an obs ended up in several classes
            path_obs = path_obs_list[0][0]
            tag = path_obs_list[0][1]

            # then update database entry and do Strehl and PCA
            try:
                # check folder modified date:
                time_tag = datetime.datetime.utcfromtimestamp(
                    os.stat(path_obs).st_mtime)

                fits100p = os.path.join(path_obs, '100p.fits')
                header = get_fits_header(fits100p) if tag != 'failed' else {}

                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.automated.status.done': True,
                            'pipelined.automated.classified_as': tag,
                            'pipelined.automated.last_modified': time_tag,
                            'pipelined.automated.fits_header': header
                        },
                        '$push': {
                            'pipelined.automated.location': ['{:s}:{:s}'.format(
                                                _config['analysis_machine_external_host'],
                                                _config['analysis_machine_external_port']),
                                                _config['path_pipe']],
                        }
                    }
                )
                _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))
                # TODO: make preview images
                # TODO: calculate Strehl
                check_strehl(_config, _logger, _coll, _select, _date, _obs, _pipe='automated')
                # TODO: run PCA
            except Exception as _e:
                print(_e)
                _logger.error(_e)
                return False
        elif len(path_obs_list) == 0:
            _logger.debug('{:s} not yet processed'.format(_obs))
        elif len(path_obs_list) > 1:
            _logger.debug('{:s} ended up in several lucky classes, check on it.'.format(_obs))

    # marked as done? check Strehl and PCA + check modified time tag for updates
    else:
        # check if actually processed
        path_obs_list = [[os.path.join(_config['path_pipe'], _date, tag, _obs), tag] for
                         tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                         os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
        # yes?
        if len(path_obs_list) == 1:
            # this also considers the pathological case when an obs ended up in several classes
            path_obs = path_obs_list[0][0]
            tag = path_obs_list[0][1]
            try:
                # check folder modified date:
                time_tag = datetime.datetime.utcfromtimestamp(
                    os.stat(path_obs).st_mtime)
                # changed? update database entry + make sure to rerun Strehl and PCA
                if _select['pipelined']['automated']['last_modified'] != time_tag:
                    fits100p = os.path.join(path_obs, '100p.fits')
                    header = get_fits_header(fits100p) if tag != 'failed' else {}

                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.automated.classified_as': tag,
                                'pipelined.automated.last_modified': time_tag,
                                'pipelined.automated.fits_header': header,
                                'pipelined.automated.status.preview': False,
                                'pipelined.automated.strehl.status.done': False,
                                'pipelined.automated.pca.status.done': False
                            }
                        }
                    )
                    _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))
                # check the following in any case:
                # TODO: remake preview images
                # TODO: recalculate Strehl
                check_strehl(_config, _logger, _coll, _select, _date, _obs, _pipe='automated')
                # TODO: rerun PCA
            except Exception as _e:
                print(_e)
                _logger.error(_e)
                return False
        elif len(path_obs_list) == 0:
            _logger.debug(
                '{:s} not yet processed (at least I could not find it), marking undone'.format(_obs))
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.automated.status.done': False
                    }
                }
            )
        elif len(path_obs_list) > 1:
            _logger.debug('{:s} ended up in several lucky classes, check on it.'.format(_obs))
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.automated.status.done': False
                    }
                }
            )

    return True


def check_pipe_faint(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been processed
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    # TODO
    if not _select['pipelined']['faint']['status']['done'] and \
            _select['pipelined']['faint']['status']['retries'] < _config['max_pipelining_retries']:
        return False

    return True


def check_strehl(_config, _logger, _coll, _select, _date, _obs, _pipe='automated'):
    """
        Check if Strehl has been calculated, and calculate if necessary
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name
    :param _pipe: which pipelined data to use? 'automated' or 'faint'?

    :return:
    """

    # if 'done' is changed to False externally, the if clause is triggered,
    # which in turn triggers a job placement into the queue to recalculate Strehl
    # when invoked for the next time, the else clause will trigger,
    # resulting in an update of the database entry (since the last_modified value
    # will be different from the new folder modification date)

    if not _select['pipelined'][_pipe]['strehl']['status']['done'] and \
                   _select['pipelined'][_pipe]['strehl']['status']['retries'] < \
                   _config['max_pipelining_retries']:
        if _pipe == 'automated':
            # check if actually processed
            path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                             tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                             os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
            # yes?
            if len(path_obs_list) == 1:
                # this also considers the pathological case when an obs ended up in several classes
                path_obs = path_obs_list[0]

                path_out = os.path.join(_config['path_strehl'], _pipe, _date)
                if not os.path.exists(_config['path_strehl']):
                    os.mkdir(_config['path_strehl'])
                if not os.path.exists(os.path.join(_config['path_strehl'], _pipe)):
                    os.mkdir(os.path.join(_config['path_strehl'], _pipe))
                if not os.path.exists(path_out):
                    os.mkdir(path_out)

                try:
                    # set stuff up:
                    telescope = 'KittPeak' if datetime.datetime.strptime(_date, '%Y%m%d') > \
                                              datetime.datetime(2015, 9, 1) else 'Palomar'
                    Strehl_factor = _config['telescope_data'][telescope]['Strehl_factor'][_select['filter']]

                    # lucky images are drizzled, use scale_red therefore
                    plate_scale = _config['telescope_data'][telescope]['scale_red']

                    # put a job into the queue
                    job_strehl(_path_in=path_obs, _fits_name='100p.fits',
                               _obs=_obs, _path_out=path_out,
                               _plate_scale=plate_scale, _Strehl_factor=Strehl_factor)
                    _logger.info('put a Strehl job into the queue for {:s}'.format(_obs))

                except Exception as _e:
                    traceback.print_exc()
                    _logger.error(_e)
                    return False

    # under path_strehl, there are folders for different pipelines,
    # then come dates, then simply obs names
    path_strehl = os.path.join(_config['path_strehl'], _pipe, _date, _obs)

    # path exists? (has been created by job_strehl)
    if os.path.exists(path_strehl):
        try:
            # check folder modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_strehl).st_mtime)
            # changed? reload data from disk + update database entry
            if _select['pipelined']['automated']['last_modified'] != time_tag:
                # reload data from disk
                f_strehl = os.path.join(path_strehl, '{:s}_strehl.txt'.format(_obs))
                x, y, core, halo, SR, FWHM, flag = load_strehl(f_strehl)
                # print(x, y, core, halo, SR, FWHM, flag)
                # update database entry
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.{:s}.strehl.status.done'.format(_pipe): True,
                            'pipelined.{:s}.strehl.lock_position'.format(_pipe): [x, y],
                            'pipelined.{:s}.strehl.ratio_percent'.format(_pipe): SR,
                            'pipelined.{:s}.strehl.core_arcsec'.format(_pipe): core,
                            'pipelined.{:s}.strehl.halo_arcsec'.format(_pipe): halo,
                            'pipelined.{:s}.strehl.fwhm_arcsec'.format(_pipe): FWHM,
                            'pipelined.{:s}.strehl.flag'.format(_pipe): flag,
                            'pipelined.{:s}.strehl.last_modified'.format(_pipe): time_tag
                        },
                        '$inc': {
                            'pipelined.{:s}.strehl.status.retries'.format(_pipe): 1
                        }
                    }
                )
                _logger.info('Updated strehl entry for {:s}'.format(_obs))
        except Exception as _e:
            traceback.print_exc()
            _logger.error(_e)
            return False

    return True


def check_pipe_pca(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been processed
        - when run for the first time, will put a task into the queue and retries++
        - when run for the second time, will generate preview images and mark as done
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs full name

    :return:
    """
    try:
        # 'done' flag = False and <3 retries?
        if not _select['pipelined']['pca']['status']['done'] and \
                        _select['pipelined']['pca']['status']['retries'] < \
                        _config['max_pipelining_retries']:
            # check if processed
            # name in accordance with the automated pipeline output,
            # but check the faint pipeline output folder too
            for tag in ('high_flux', 'faint'):
                path_obs = os.path.join(_config['path_pca'], _date, tag, _obs)
                # already processed?
                if os.path.exists(path_obs):
                    # check folder modified date:
                    time_tag = datetime.datetime.utcfromtimestamp(
                        os.stat(path_obs).st_mtime)
                    # load contrast curve
                    f_cc = [f for f in os.listdir(path_obs) if '_contrast_curve.txt' in f][0]
                    cc = np.loadtxt(f_cc)

                    # previews generated?
                    if not _select['pipelined']['pca']['status']['review']:
                        # load first image frame from a fits file
                        f_fits = [f for f in os.listdir(path_obs) if '.fits' in f][0]
                        _fits = load_fits(f_fits)
                        # scale with local contrast optimization for preview:
                        preview_img = scale_image(_fits, correction='local')

                        _status = generate_pca_images(_out_path=_config['path_pca'],
                                                      _sou_dir=_obs, _preview_img=preview_img,
                                                      _cc=cc, _fow_x=36, _pix_x=1024, _drizzle=True)

                        if _status:
                            _coll.update_one(
                                {'_id': _obs},
                                {
                                    '$set': {
                                        'pipelined.pca.status.preview': True
                                    }
                                }
                            )
                            _logger.debug('Updated pca pipeline entry [status.review] for {:s}'.format(_obs))
                        else:
                            _logger.debug('Failed to generate pca pipeline preview images for {:s}'.format(_obs))
                    # update database entry
                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.pca.status.done': True,
                                'pipelined.pca.contrast_curve': cc.tolist(),
                                'pipelined.pca.last_modified': time_tag
                            },
                            '$push': {
                                'pipelined.automated.location': ['{:s}:{:s}'.format(
                                    _config['analysis_machine_external_host'],
                                    _config['analysis_machine_external_port']),
                                    _config['path_pca']]
                            }
                        }
                    )
                    _logger.debug('Updated pca pipeline entry for {:s}'.format(_obs))
                # not processed yet? put a job into the queue then:
                else:
                    # this will produce a fits file with the psf-subtracted image
                    # and a text file with the contrast curve
                    # TODO:
                    job_pca(_obs)
                    _logger.debug('put a pca job into the queue for {:s}'.format(_obs))
                    # increment number of tries
                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$inc': {
                                'pipelined.pca.status.retries': 1
                            }
                        }
                    )
                    _logger.debug('Updated pca pipeline entry [status.retries] for {:s}'.format(_obs))

    except Exception as _e:
        print(_e)
        logger.error(_e)
        return False

    return True


if __name__ == '__main__':
    """
        - create argument parser, parse command line arguments
        - set up logging
        - load config
    """

    ''' Create command line argument parser '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data archive for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)

    args = parser.parse_args()

    ''' set up logging '''
    logger = set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG)

    # if you start me up... if you start me up I'll never stop (hopefully not)
    logger.info('Started daily archiving job.')

    try:
        ''' script absolute location '''
        abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

        ''' load config data '''
        try:
            config = get_config(_config_file=args.config_file)
            logger.debug('Successfully read in the config file {:s}'.format(args.config_file))
        except Exception as e:
            logger.error(e)
            logger.error('Failed to read in the config file {:s}'.format(args.config_file))
            sys.exit()

        ''' Connect to the mongodb database '''
        try:
            db, coll, program_pi = connect_to_db(_config=config, _logger=logger)
            if None in (db, coll, program_pi):
                raise Exception('Failed to connect to the database')
        except Exception as e:
            logger.error(e)
            sys.exit()

        '''
         ###############################
         CHECK IF DATABASE IS UP TO DATE
         ###############################
        '''

        ''' check all raw data starting from config['archiving_start_date'] '''
        # get all dates with some raw data
        dates = [p for p in os.listdir(config['path_raw'])
                 if os.path.isdir(os.path.join(config['path_raw'], p))
                 and datetime.datetime.strptime(p, '%Y%m%d') >= config['archiving_start_date']]
        print(dates)
        # for each date get all unique obs names (used as _id 's in the db)
        for date in dates:
            date_files = os.listdir(os.path.join(config['path_raw'], date))
            # check the endings (\Z) and skip _N.fits.bz2:
            # must start with program number (e.g. 24_ or 24.1_)
            pattern_start = r'\d+.?\d??_'
            # must be a bzipped fits file
            pattern_end = r'.[0-9]{6}.fits.bz2\Z'
            pattern_fits = r'.fits.bz2\Z'
            # skip calibration files and pointings
            date_obs = [re.split(pattern_fits, s)[0] for s in date_files
                        if re.search(pattern_end, s) is not None and
                        re.match(pattern_start, s) is not None and
                        re.match('pointing_', s) is None and
                        re.match('bias_', s) is None and
                        re.match('dark_', s) is None and
                        re.match('flat_', s) is None and
                        re.match('seeing_', s) is None]
            print(date_obs)
            # TODO: handle seeing files separately [lower priority]
            date_seeing = [re.split(pattern_end, s)[0] for s in date_files
                           if re.search(pattern_end, s) is not None and
                           re.match('seeing_', s) is not None]
            print(date_seeing)
            # for each source name see if there's an entry in the database
            for obs in date_obs:
                print('processing {:s}'.format(obs))
                logger.debug('processing {:s}'.format(obs))
                # parse name:
                tmp = obs.split('_')
                # print(tmp)
                # program num
                prog_num = str(tmp[0])
                # who's pi?
                if prog_num in program_pi.keys():
                    prog_pi = program_pi[prog_num]
                else:
                    # play safe if pi's unknown:
                    prog_pi = 'admin'
                # stack name together if necessary (if contains underscores):
                sou_name = '_'.join(tmp[1:-5])
                # code of the filter used:
                filt = tmp[-4:-3][0]
                # date and time of obs:
                date_utc = datetime.datetime.strptime(tmp[-2] + tmp[-1], '%Y%m%d%H%M%S.%f')
                # camera:
                camera = tmp[-5:-4][0]
                # marker:
                marker = tmp[-3:-2][0]

                # look up entry in the database:
                select = coll.find_one({'_id': obs})
                # if entry not in database, create empty one and populate it
                if select is None:
                    print('{:s} not in database, adding...'.format(obs))
                    logger.info('{:s} not in database, adding'.format(obs))
                    entry = empty_db_record()
                    # populate:
                    entry['_id'] = obs
                    entry['name'] = sou_name
                    entry['science_program']['program_id'] = prog_num
                    entry['science_program']['program_PI'] = prog_pi
                    entry['date_utc'] = date_utc
                    if date_utc > datetime.datetime(2015, 10, 1):
                        entry['telescope'] = 'KPNO_2.1m'
                    else:
                        entry['telescope'] = 'Palomar_P60'
                    entry['camera'] = camera
                    entry['filter'] = filt  # also get this from FITS header

                    # find raw fits files:
                    raws = [s for s in date_files if re.match(obs, s) is not None]
                    entry['raw_data']['location'].append(['{:s}:{:s}'.format(
                                                    config['analysis_machine_external_host'],
                                                    config['analysis_machine_external_port']),
                                                    config['path_raw']])
                    entry['raw_data']['data'] = raws
                    entry['raw_data']['last_modified'] = datetime.datetime.now(pytz.utc)

                    # insert into database
                    result = coll.insert_one(entry)

                # entry found in database, check if pipelined, update entry if necessary
                else:
                    print('{:s} in database, checking...'.format(obs))
                    ''' check lucky-pipelined data '''
                    status_ok = check_pipe_automated(_config=config, _logger=logger, _coll=coll,
                                                     _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for lucky pipeline: {:s}'.format(obs))

                    # ''' check Strehl data '''
                    # status_ok = check_strehl(_config=config, _logger=logger, _coll=coll,
                    #                          _select=select, _date=date, _obs=obs)
                    # if not status_ok:
                    #     logger.error('Checking failed for Strehl pipeline: {:s}'.format(obs))

                    # TODO: if it is not a planetary, observation, do the following:
                    if True is False:
                        ''' check faint-pipelined data '''
                        # TODO: if core and halo tell you it's faint, run faint pipeline:
                        status_ok = check_pipe_faint(_config=config, _logger=logger, _coll=coll,
                                                     _select=select, _date=date, _obs=obs)
                        if not status_ok:
                            logger.error('Checking failed for faint pipeline: {:s}'.format(obs))

                        ''' check PCA-pipelined data '''
                        # TODO: also depending on Strehl data, run PCA pipeline either on
                        # TODO: the lucky or the faint image
                        # TODO: if a faint image is not ready (yet), will skip and do it next time
                        status_ok = check_pipe_pca(_config=config, _logger=logger, _coll=coll,
                                                   _select=select, _date=date, _obs=obs)
                        if not status_ok:
                            logger.error('Checking failed for PCA pipeline: {:s}'.format(obs))

                    # TODO: if it is a planetary, run the planetary pipeline

                    ''' check seeing data '''
                    # TODO: [lower priority]
                    # for each date check if lists of processed and raw seeing files match
                    # rerun seeing.py for each date if necessary
                    # update last_modified if necessary

                # TODO: mark distributed when all pipelines done or n_retries>3,
                # TODO: compress everything with bzip2, store and transfer over to Caltech

            # TODO: query database for all contrast curves and Strehls [+seeing - lower priority]
            # TODO: make joint plots to display on the website

    except Exception as e:
        print(e)
        logger.error(e)
        logger.error('Unknown error.')
    finally:
        logger.info('Finished daily archiving job.')
