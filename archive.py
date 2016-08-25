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
import vip
import photutils
from scipy import stats
import operator

from skimage import exposure, img_as_float
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import matplotlib.pyplot as plt
import seaborn as sns

# import numba
from huey import RedisHuey
from redis.exceptions import ConnectionError
huey = RedisHuey(name='roboao.archive', host='127.0.0.1', port='6379', result_store=True)

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
def job_pca(_config, _path_in, _fits_name, _obs, _path_out,
            _plate_scale, _method='sextractor', _x=None, _y=None, _drizzled=True):
    try:
        with fits.open(_config['pca']['path_psf_reference_library']) as _lib:
            _library = _lib[0].data
        _library_names_short = np.genfromtxt(_config['pca']['path_psf_reference_library_short_names'],
                                             dtype='|S')

        _win = _config['pca']['win']
        _sigma = _config['pca']['sigma']
        _nrefs = _config['pca']['nrefs']
        _klip = _config['pca']['klip']

        _trimmed_frame, x_lock, y_lock = trim_frame(_path_in, _fits_name=_fits_name,
                                                    _win=_win, _method=_method,
                                                    _x=_x, _y=_x, _drizzled=_drizzled)
        # print(x_lock, y_lock)

        # Filter the trimmed frame with IUWT filter, 2 coeffs
        filtered_frame = (vip.var.cube_filter_iuwt(
            np.reshape(_trimmed_frame, (1, np.shape(_trimmed_frame)[0],
                                        np.shape(_trimmed_frame)[1])),
            coeff=5, rel_coeff=2))

        mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta = \
            (vip.var.fit_2dgaussian(filtered_frame[0], crop=True, cropsize=50,
                                    debug=False, full_output=True))
        _fwhm = np.mean([fwhm_y, fwhm_x])

        # Print the resolution element size
        # print('Using resolution element size = ', _fwhm)
        if _fwhm < 2:
            _fwhm = 2.0
            # print('Too small, changing to ', _fwhm)

        # Center the filtered frame
        centered_cube, shy, shx = \
            (vip.calib.cube_recenter_gauss2d_fit(array=filtered_frame, pos_y=_win,
                                                 pos_x=_win, fwhm=_fwhm,
                                                 subi_size=6, nproc=1, full_output=True))

        centered_frame = centered_cube[0]
        if shy > 5 or shx > 5:
            raise TypeError('Centering failed: pixel shifts too big')

        # Do aperture photometry on the central star
        center_aperture = photutils.CircularAperture(
            (int(len(centered_frame) / 2), int(len(centered_frame) / 2)), _fwhm / 2.0)
        center_flux = photutils.aperture_photometry(centered_frame, center_aperture)['aperture_sum'][0]

        # Make PSF template for calculating PCA throughput
        psf_template = (
            centered_frame[len(centered_frame) / 2 - 3 * _fwhm:len(centered_frame) / 2 + 3 * _fwhm,
            len(centered_frame) / 2 - 3 * _fwhm:len(centered_frame) / 2 + 3 * _fwhm])

        # Choose reference frames via cross correlation
        library_notmystar = _library[~np.in1d(_library_names_short, _obs)]
        cross_corr = np.zeros(len(library_notmystar))
        flattened_frame = np.ndarray.flatten(centered_frame)

        for c in range(len(library_notmystar)):
            cross_corr[c] = stats.pearsonr(flattened_frame,
                                           np.ndarray.flatten(library_notmystar[c, :, :]))[0]

        cross_corr_sorted, index_sorted = (np.array(zip(*sorted(zip(cross_corr, np.arange(len(cross_corr))),
                                                                key=operator.itemgetter(0), reverse=True))))
        index_sorted = np.int_(index_sorted)
        library = library_notmystar[index_sorted[0:_nrefs], :, :]
        # print('Library correlations = ', cross_corr_sorted[0:_nrefs])

        # Do PCA
        reshaped_frame = np.reshape(centered_frame, (1, centered_frame.shape[0], centered_frame.shape[1]))
        pca_frame = vip.pca.pca(reshaped_frame, np.zeros(1), library, ncomp=_klip)

        pca_file_name = os.path.join(_path_out, _obs + '_pca.fits')
        # print(pca_file_name)

        # dump results to disk
        if not (os.path.exists(_path_out)):
            os.makedirs(_path_out)

        # save fits after PCA
        hdu = fits.PrimaryHDU(pca_frame)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(pca_file_name, clobber=True)

        # Make contrast curve
        [con, cont, sep] = (vip.phot.contrcurve.contrast_curve(cube=reshaped_frame, angle_list=np.zeros(1),
                                                               psf_template=psf_template,
                                                               cube_ref=library, fwhm=_fwhm,
                                                               pxscale=_plate_scale,
                                                               starphot=center_flux, sigma=_sigma,
                                                               ncomp=_klip, algo='pca-rdi-fullfr',
                                                               debug=False,
                                                               plot=False, nbranch=3, scaling=None,
                                                               mask_center_px=_fwhm, fc_rad_sep=6))

        # save txt for nightly median calc/plot
        with open(os.path.join(_path_out, _obs + '_contrast_curve.txt'), 'w') as f:
            f.write('# lock position: {:d} {:d}\n'.format(x_lock, y_lock))
            for _s, dm in zip(sep, -2.5 * np.log10(cont)):
                f.write('{:.3f} {:.3f}\n'.format(_s, dm))

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        return False

    return True


@huey.task()
def job_strehl(_path_in, _fits_name, _obs, _path_out, _plate_scale, _Strehl_factor,
               _method='pipeline_settings.txt', _win=100, _x=None, _y=None, _drizzled=True):
    """
        The task that calculates Strehl ratio

    :param _path_in:
    :param _fits_name:
    :param _obs:
    :param _path_out:
    :param _plate_scale:
    :param _Strehl_factor:
    :param _method: which method to use with trim_frame(...) to get frame lock position
                    by default, it's imported from lucky pipeline's pipeline_settings.txt
                    don't trust it or it's not a lucky-processed image?
                    then use 'sextractor'. (see trim_fram(**kvargs) for details and other options)
    :return:
    """

    # do the work
    try:
        img, x, y = trim_frame(_path_in, _fits_name=_fits_name,
                               _win=_win, _method=_method,
                               _x=_x, _y=_x, _drizzled=_drizzled)
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
    if not(os.path.exists(_path_out)):
        os.makedirs(_path_out)

    # save box around selected object:
    hdu = fits.PrimaryHDU(box)
    hdu.writeto(os.path.join(_path_out, '{:s}_box.fits'.format(_obs)), clobber=True)

    # save the Strehl data to txt-file:
    with open(os.path.join(_path_out, '{:s}_strehl.txt'.format(_obs)), 'w') as _f:
        _f.write('# lock_x[px] lock_y[px] core["] halo["] SR[%] FWHM["] flag\n')
        output_entry = '{:d} {:d} {:.5f} {:.5f} {:.5f} {:.5f} {:s}\n'.\
            format(x, y, core, halo, SR * 100, FWHM, flag)
        _f.write(output_entry)

    return True


@huey.task()
# @numba.jit
def job_bogus(_obs):
    """
        I've been using this to test the redis queue + jit compilation
    :param _obs:
    :return:
    """
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


def mkdirs(_path):
    """
        mimic os.makedirs() why? why not?..
    :param _path:
    :return:
    """
    p, stack = _path, []
    while not os.path.exists(p):
        _tmp = os.path.split(p)
        p = _tmp[0]
        stack.insert(0, _tmp[1])
    for _dir in stack:
        p = os.path.join(p, _dir)
        os.mkdir(p)


def load_fits(fin):
    with fits.open(fin) as _f:
        scidata = _f[0].data
    return scidata


def load_strehl(fin):
    """
        Load Strehl data from a text-file
        For format, see any of the *_strehl.txt files
    :param fin:
    :return:
    """
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


def load_cc(fin):
    """
            Load contrast curve data from a text-file
            Format: first line: lock position (x, y) on the original frame
                    next lines: separation["] contrast[mag]
        :param fin:
        :return:
        """
    with open(fin) as _f:
        f_lines = _f.readlines()

    _tmp = f_lines[0].split()
    x, y = int(_tmp[-2]), int(_tmp[-1])

    cc = []
    for l in f_lines[1:]:
        cc.append(map(float, l.split()))

    return x, y, cc


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


def generate_pipe_preview(_path_out, _obs, preview_img, preview_img_cropped,
                          SR=None, _fow_x=36, _pix_x=1024, _drizzled=True,
                          _x=None, _y=None, objects=None):
    """
    :param _path_out:
    :param preview_img:
    :param preview_img_cropped:
    :param SR:
    :param _x: cropped image will be centered around these _x
    :param _y: and _y + a box will be drawn on full image around this position
    :param objects: np.array([[x_0,y_0], ..., [x_N,y_N]])
    :return:
    """
    try:
        ''' full image '''
        plt.close('all')
        fig = plt.figure()
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        # plot detected objects:
        if objects is not None:
            ax.plot(objects[:, 0]-1, objects[:, 1]-1, 'o',
                    markeredgewidth=1, markerfacecolor='None', markeredgecolor=plt.cm.Oranges(0.8))
        # ax.imshow(preview_img, cmap='gray', origin='lower', interpolation='nearest')
        ax.imshow(preview_img, cmap='gray', origin='lower', interpolation='nearest')
        # plot a box around the cropped object
        if _x is not None and _y is not None:
            _h = int(preview_img_cropped.shape[0])
            _w = int(preview_img_cropped.shape[1])
            ax.add_patch(Rectangle((_y-_w/2, _x-_h/2), _w, _h,
                                   fill=False, edgecolor='#f3f3f3', linestyle='dotted'))
        # ax.imshow(preview_img, cmap='gist_heat', origin='lower', interpolation='nearest')
        # plt.axis('off')
        plt.grid('off')

        # save full figure
        fname_full = '{:s}_full.png'.format(_obs)
        if not (os.path.exists(_path_out)):
            os.makedirs(_path_out)
        plt.savefig(os.path.join(_path_out, fname_full), dpi=300)

        ''' cropped image: '''
        # save cropped image
        plt.close('all')
        fig = plt.figure()
        fig.set_size_inches(3, 3, forward=False)
        # ax = fig.add_subplot(111)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(preview_img_cropped, cmap='gray', origin='lower', interpolation='nearest')
        # add scale bar:
        # draw a horizontal bar with length of 0.1*x_size
        # (ax.transData) with a label underneath.
        bar_len = preview_img_cropped.shape[0] * 0.1
        mltplr = 2 if _drizzled else 1
        bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / mltplr)
        asb = AnchoredSizeBar(ax.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax.add_artist(asb)
        # add Strehl ratio
        if SR is not None:
            asb2 = AnchoredSizeBar(ax.transData,
                                   0,
                                   'Strehl: {:.2f}%'.format(float(SR)),
                                   loc=2, pad=0.3, borderpad=0.4, sep=5, frameon=False)
            ax.add_artist(asb2)
            # asb3 = AnchoredSizeBar(ax.transData,
            #                        0,
            #                        'SR: {:.2f}%'.format(float(SR)),
            #                        loc=3, pad=0.3, borderpad=0.5, sep=10, frameon=False)
            # ax.add_artist(asb3)

        # save cropped figure
        fname_cropped = '{:s}_cropped.png'.format(_obs)
        if not (os.path.exists(_path_out)):
            os.makedirs(_path_out)
        fig.savefig(os.path.join(_path_out, fname_cropped), dpi=300)

    except Exception as _e:
        traceback.print_exc()
        print(_e)
        return False

    return True


def generate_pca_images(_path_out, _obs, _preview_img, _cc,
                        _fow_x=36, _pix_x=1024, _drizzled=True):
    """
            Generate preview images for the pca pipeline

        :param _out_path:
        :param _obs:
        :param _preview_img:
        :param _cc: contrast curve
        :param _fow_x: full FoW in arcseconds in the x direction
        :param _pix_x: original (raw) full frame size in pixels
        :param _drizzled: drizzle on or off?

        :return:
        """
    try:
        ''' plot psf-subtracted image '''
        plt.close('all')
        fig = plt.figure(_obs)
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
        # account for possible drizzling
        mltplr = 2 if _drizzled else 1
        bar_len_str = '{:.1f}'.format(bar_len * _fow_x / _pix_x / mltplr)
        asb = AnchoredSizeBar(ax.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax.add_artist(asb)

        # save figure
        fig.savefig(os.path.join(_path_out, _obs + '_pca.png'), dpi=300)

        ''' plot the contrast curve '''
        # convert cc to numpy array if necessary:

        if not isinstance(_cc, np.ndarray):
            _cc = np.array(_cc)

        plt.close('all')
        fig = plt.figure('Contrast curve for {:s}'.format(_obs), figsize=(8, 3.5), dpi=200)
        ax = fig.add_subplot(111)
        ax.set_title(_obs)  # , fontsize=14)
        # ax.plot(_cc[:, 0], -2.5 * np.log10(_cc[:, 1]), 'k-', linewidth=2.5)
        # _cc[:, 1] is already in mag:
        ax.plot(_cc[:, 0], _cc[:, 1], 'k-', linewidth=2.5)
        ax.set_xlim([0.2, 1.45])
        ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
        ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
        ax.set_ylim([0, 8])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.grid(linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(_path_out, _obs + '_contrast_curve.png'), dpi=200)
    except Exception as _e:
        print(_e)
        traceback.print_exc()
        return False

    return True


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


def get_xy_from_pipeline_settings_txt(_path, _first=True):
    """
        Get centroid position for a lucky-pipelined image
    :param _path:
    :param _first: output the x,y from the first run? if False, output from the last
    :return:
    """
    with open(os.path.join(_path, 'pipeline_settings.txt'), 'r') as _f:
        f_lines = _f.readlines()

    for l in f_lines:
        _tmp = re.search(r'\d\s+\d\s+\((\d+),(\d+)\),\((\d+),(\d+)\)\n', l)
        if _tmp is not None:
            _x = (int(_tmp.group(1)) + int(_tmp.group(3)))/2
            _y = (int(_tmp.group(2)) + int(_tmp.group(4)))/2
            if _first:
                break

    return _x, _y


# detect observations, which are bad because of being too faint
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
        Crop image around a star, which is detected by one of the _methods
        (e.g. SExtracted and rated)

    :param _path: path
    :param _fits_name: fits-file name
    :param _win: window width
    :param _method: from 'frames.txt' (if this is the output of the standard lucky pipeline),
                    from 'pipeline_settings.txt', using 'sextractor', a simple 'max', or 'manual'
    :param _x: source x position -- if known in advance
    :param _y: source y position -- if known in advance
    :param _drizzled: was it drizzled?

    :return: image, cropped around a lock position and the lock position itself
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
        # maximum error of a Gaussian fit. Real sources usually have larger 'errors'
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
            # window size not set? set it automatically based on source fwhm
            if _win is None:
                sou_size = np.max((int(out['table']['FWHM_IMAGE'][best_score] * 3), 100))
                _win = sou_size
            # print(out['table']['XPEAK_IMAGE'][best_score], out['table']['YPEAK_IMAGE'][best_score])
            # print(get_xy_from_frames_txt(_path))
            x = out['table']['YPEAK_IMAGE'][best_score]
            y = out['table']['XPEAK_IMAGE'][best_score]
            scidata_cropped = scidata[x - _win: x + _win + 1,
                                      y - _win: y + _win + 1]
        else:
            if _win is None:
                _win = 100
            # use a simple max instead:
            x, y = np.unravel_index(scidata.argmax(), scidata.shape)
            scidata_cropped = scidata[x - _win: x + _win + 1,
                                      y - _win: y + _win + 1]
    elif _method == 'max':
        if _win is None:
            _win = 100
        x, y = np.unravel_index(scidata.argmax(), scidata.shape)
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    elif _method == 'frames.txt':
        if _win is None:
            _win = 100
        y, x = get_xy_from_frames_txt(_path)
        if _drizzled:
            x *= 2.0
            y *= 2.0
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    elif _method == 'pipeline_settings.txt':
        if _win is None:
            _win = 100
        y, x = get_xy_from_pipeline_settings_txt(_path)
        if _drizzled:
            x *= 2.0
            y *= 2.0
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    elif _method == 'manual' and _x is not None and _y is not None:
        if _win is None:
            _win = 100
        x, y = _x, _y
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]
    else:
        raise Exception('unrecognized trimming method.')

    return scidata_cropped, int(x), int(y)


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
    """
        A dummy database record
    :return:
    """
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
                        'done': False
                    },
                    'preview': {
                        'done': False,
                        'retries': 0,
                        'last_modified': time_now_utc
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
                            'retries': 0
                        },
                        'preview': {
                            'done': False,
                            'retries': 0,
                            'last_modified': time_now_utc
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
                        'retries': 0
                    },
                    'preview': {
                        'done': False,
                        'retries': 0,
                        'last_modified': time_now_utc
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
                            'retries': 0
                        },
                        'preview': {
                            'done': False,
                            'retries': 0,
                            'last_modified': time_now_utc
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
    # path to data archive:
    _config['path_archive'] = config.get('Path', 'path_archive')

    # telescope data (voor, o.a., Strehl computation)
    _tmp = ast.literal_eval(config.get('Strehl', 'Strehl_factor'))
    _config['telescope_data'] = dict()
    for telescope in 'Palomar', 'KittPeak':
        _config['telescope_data'][telescope] = ast.literal_eval(config.get('Strehl', telescope))
        _config['telescope_data'][telescope]['Strehl_factor'] = _tmp[telescope]

    # pca pipeline
    _config['pca'] = dict()
    # path to PSF library:
    _config['pca']['path_psf_reference_library'] = config.get('Path', 'path_psf_reference_library')
    _config['pca']['path_psf_reference_library_short_names'] = \
        config.get('Path', 'path_psf_reference_library_short_names')
    # have to do it inside job_pca - otherwise will have to send hundreds of Mb
    # back and forth between redis queue and task consumer. luckily, it's pretty fast to do
    # with fits.open(path_psf_reference_library) as _lib:
    #     _config['pca']['psf_reference_library'] = _lib[0].data
    # _config['pca']['psf_reference_library_short_names'] = np.genfromtxt(path_psf_reference_library_short_names,
    #                                                              dtype='|S')

    _config['pca']['win'] = int(config.get('PCA', 'win'))
    _config['pca']['sigma'] = float(config.get('PCA', 'sigma'))
    _config['pca']['nrefs'] = float(config.get('PCA', 'nrefs'))
    _config['pca']['klip'] = float(config.get('PCA', 'klip'))

    _config['planets_prog_num'] = str(config.get('Programs', 'planets'))

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
        Check if observation has been automatically lucky-pipelined
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name

    :return:
    """
    try:
        # check if actually processed
        path_obs_list = [[os.path.join(_config['path_pipe'], _date, tag, _obs), tag] for
                         tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                         os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
        # yes?
        if len(path_obs_list) == 1:
            # this also considers the pathological case when an obs ended up in several classes
            path_obs = path_obs_list[0][0]
            # lucky pipeline classified it as:
            tag = path_obs_list[0][1]

            # check folder modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_obs).st_mtime)

            # make sure db reflects reality: not yet in db or had been modified
            if not _select['pipelined']['automated']['status']['done'] or \
                            _select['pipelined']['automated']['last_modified'] != time_tag:

                # get fits header:
                fits100p = os.path.join(path_obs, '100p.fits')
                header = get_fits_header(fits100p) if tag != 'failed' else {}

                # update db entry. reset status flags to (re)run Strehl/PCA/preview
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.automated.status.done': True,
                            'pipelined.automated.classified_as': tag,
                            'pipelined.automated.last_modified': time_tag,
                            'pipelined.automated.fits_header': header,
                            'pipelined.automated.location': ['{:s}:{:s}'.format(
                                _config['analysis_machine_external_host'],
                                _config['analysis_machine_external_port']),
                                _config['path_pipe']],
                            'pipelined.automated.preview.done': False,
                            'pipelined.automated.strehl.status.done': False,
                            'pipelined.automated.pca.status.done': False
                        }
                    }
                )
                _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))

                # reload entry from db:
                _select = _coll.find_one({'_id': _obs})

            # check on Strehl:
            check_strehl(_config, _logger, _coll, _select, _date, _obs, _pipe='automated')
            # check on PCA
            # Strehl done and ok? then proceed:
            # FIXME:  core and halo:
            if _select['pipelined']['automated']['strehl']['status']['done'] and \
                            _select['pipelined']['automated']['strehl']['core_arcsec'] > 0.14e-5 and \
                            _select['pipelined']['automated']['strehl']['halo_arcsec'] < 1.0e5:
                check_pca(_config, _logger, _coll, _select, _date, _obs, _pipe='automated')
            # make preview images
            check_preview(_config, _logger, _coll, _select, _date, _obs, _pipe='automated')
            # once(/if) Strehl is ready, it'll rerun preview generation to SR on the image

        # not processed?
        elif len(path_obs_list) == 0:
            _logger.debug(
                '{:s} not yet processed (at least I could not find it), marking undone'.format(_obs))
            # make sure it's not marked 'done'
            if _select['pipelined']['automated']['status']['done']:
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
            # make sure it's not marked 'done'
            if _select['pipelined']['automated']['status']['done']:
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.automated.status.done': False
                        }
                    }
                )

    except Exception as _e:
        print(_e)
        _logger.error(_e)
        return False

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
    try:
        # following structure.md:
        path_strehl = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'strehl')

        # path exists? (if yes - it must have been created by job_strehl)
        if os.path.exists(path_strehl):
            # check folder modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_strehl).st_mtime)
            # new/changed? (re)load data from disk + update database entry + (re)make preview
            if _select['pipelined'][_pipe]['strehl']['last_modified'] != time_tag:
                # load data from disk
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
                        }
                    }
                )
                # reload entry from db:
                _select = _coll.find_one({'_id': _obs})
                _logger.info('Updated strehl entry for {:s}'.format(_obs))
                # remake preview images
                # check_preview(_config, _logger, _coll, _select, _date, _obs, _pipe=_pipe)
                # _logger.info('Remade preview images for {:s}'.format(_obs))

        # path does not exist? make sure it's not marked 'done'
        elif _select['pipelined'][_pipe]['strehl']['status']['done']:
            # update database entry if incorrectly marked 'done'
            # (could not find the respective directory)
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.{:s}.strehl.status.done'.format(_pipe): False,
                        'pipelined.{:s}.strehl.lock_position'.format(_pipe): None,
                        'pipelined.{:s}.strehl.ratio_percent'.format(_pipe): None,
                        'pipelined.{:s}.strehl.core_arcsec'.format(_pipe): None,
                        'pipelined.{:s}.strehl.halo_arcsec'.format(_pipe): None,
                        'pipelined.{:s}.strehl.fwhm_arcsec'.format(_pipe): None,
                        'pipelined.{:s}.strehl.flag'.format(_pipe): None,
                        'pipelined.{:s}.strehl.last_modified'.format(_pipe): utc_now()
                    }
                }
            )
            # reload entry from db:
            _select = _coll.find_one({'_id': _obs})
            _logger.info('Corrected strehl entry for {:s}'.format(_obs))
            # a job will be placed into the queue at the next invocation of check_strehl

        # if 'done' is changed to False (ex- or internally), the if clause is triggered,
        # which in turn triggers a job placement into the queue to recalculate Strehl.
        # when invoked for the next time, the previous portion of the code will make sure
        # to update the database entry (since the last_modified value
        # will be different from the new folder modification date)

        if not _select['pipelined'][_pipe]['strehl']['status']['done'] and \
                       _select['pipelined'][_pipe]['strehl']['status']['retries'] < \
                       _config['max_pipelining_retries']:
            if _pipe == 'automated':
                # check if actually processed through pipeline
                path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                                 tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                                 os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
                _fits_name = '100p.fits'
                _drizzled = True
            elif _pipe == 'faint':
                raise NotImplementedError
                # path_obs_list = []
            else:
                raise NotImplementedError
                # path_obs_list = []

            # pipelined?
            if len(path_obs_list) == 1:
                # this also considers the pathological case when an obs ended up in several classes
                path_obs = path_obs_list[0]

                # this follows the definition from structure.md
                path_out = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'strehl')

                # set stuff up:
                telescope = 'KittPeak' if datetime.datetime.strptime(_date, '%Y%m%d') > \
                                          datetime.datetime(2015, 9, 1) else 'Palomar'
                Strehl_factor = _config['telescope_data'][telescope]['Strehl_factor'][_select['filter']]

                # lucky images are drizzled, use scale_red therefore
                plate_scale = _config['telescope_data'][telescope]['scale_red']

                # put a job into the queue
                job_strehl(_path_in=path_obs, _fits_name=_fits_name,
                           _obs=_obs, _path_out=path_out,
                           _plate_scale=plate_scale, _Strehl_factor=Strehl_factor,
                           _method='pipeline_settings.txt',
                           _drizzled=_drizzled)
                # update database entry
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.{:s}.strehl.last_modified'.format(_pipe): utc_now()
                        },
                        '$inc': {
                            'pipelined.{:s}.strehl.status.retries'.format(_pipe): 1
                        }
                    }
                )
                _logger.info('put a Strehl job into the queue for {:s}'.format(_obs))

    except Exception as _e:
        traceback.print_exc()
        _logger.error(_e)
        return False

    return True


def check_pca(_config, _logger, _coll, _select, _date, _obs, _pipe='automated'):
    """

    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name
    :param _pipe: which pipelined data to use? 'automated' or 'faint'?

    :return:
    """
    try:
        # following structure.md:
        path_pca = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'pca')

        # path exists? (if yes - it must have been created by job_strehl)
        if os.path.exists(path_pca):
            # check folder modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_pca).st_mtime)
            # new/changed? (re)load data from disk + update database entry + (re)make preview
            if _select['pipelined'][_pipe]['pca']['last_modified'] != time_tag:
                # load data from disk
                # load contrast curve
                f_cc = [f for f in os.listdir(path_pca) if '_contrast_curve.txt' in f][0]
                x, y, cc = load_cc(os.path.join(path_pca, f_cc))
                # update database entry
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.{:s}.pca.status.done'.format(_pipe): True,
                            'pipelined.{:s}.pca.lock_position'.format(_pipe): [x, y],
                            'pipelined.{:s}.pca.contrast_curve'.format(_pipe): cc,
                            'pipelined.{:s}.pca.location'.format(_pipe): ['{:s}:{:s}'.format(
                                _config['analysis_machine_external_host'],
                                _config['analysis_machine_external_port']),
                                _config['path_archive']],
                            'pipelined.{:s}.pca.preview.done'.format(_pipe): False,
                            'pipelined.{:s}.pca.preview.last_modified'.format(_pipe): utc_now(),
                            'pipelined.{:s}.pca.last_modified'.format(_pipe): time_tag
                        }
                    }
                )
                # reload entry from db:
                _select = _coll.find_one({'_id': _obs})
                _logger.info('Updated pca entry for {:s}'.format(_obs))
            # (re)make preview images
            check_pca_preview(_config, _logger, _coll, _select, _date, _obs, _pipe=_pipe)

        # path does not exist? make sure it's not marked 'done'
        elif _select['pipelined'][_pipe]['pca']['status']['done']:
            # update database entry if incorrectly marked 'done'
            # (could not find the respective directory)
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.{:s}.pca.status.done'.format(_pipe): False,
                        'pipelined.{:s}.pca.lock_position'.format(_pipe): None,
                        'pipelined.{:s}.pca.contrast_curve'.format(_pipe): None,
                        'pipelined.{:s}.pca.location'.format(_pipe): None,
                        'pipelined.{:s}.pca.preview.done'.format(_pipe): False,
                        'pipelined.{:s}.pca.preview.last_modified'.format(_pipe): utc_now(),
                        'pipelined.{:s}.pca.last_modified'.format(_pipe): utc_now()
                    }
                }
            )
            # reload entry from db:
            _select = _coll.find_one({'_id': _obs})
            _logger.info('Corrected PCA entry for {:s}'.format(_obs))
            # a job will be placed into the queue at the next invocation of check_strehl

        # if 'done' is changed to False (ex- or internally), the if clause is triggered,
        # which in turn triggers a job placement into the queue to rerun PCA.
        # when invoked for the next time, the previous portion of the code will make sure
        # to update the database entry (since the last_modified value
        # will be different from the new folder modification date)

        if not _select['pipelined'][_pipe]['pca']['status']['done'] and \
                       _select['pipelined'][_pipe]['pca']['status']['retries'] < \
                       _config['max_pipelining_retries']:
            if _pipe == 'automated':
                # check if actually processed through pipeline
                path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                                 tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                                 os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
                _fits_name = '100p.fits'
                _drizzled = True
            elif _pipe == 'faint':
                raise NotImplementedError
                # path_obs_list = []
            else:
                raise NotImplementedError
                # path_obs_list = []

            # pipelined?
            if len(path_obs_list) == 1:
                # this also considers the pathological case when an obs ended up in several classes
                path_obs = path_obs_list[0]

                # this follows the definition from structure.md
                path_out = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'pca')

                # set stuff up:
                telescope = 'KittPeak' if datetime.datetime.strptime(_date, '%Y%m%d') > \
                                          datetime.datetime(2015, 9, 1) else 'Palomar'

                # lucky images are drizzled, use scale_red therefore
                if _pipe == 'automated':
                    plate_scale = _config['telescope_data'][telescope]['scale_red']
                elif _pipe == 'faint':
                    plate_scale = _config['telescope_data'][telescope]['scale']
                else:
                    raise NotImplementedError

                # put a job into the queue
                job_pca(_config=_config, _path_in=path_obs, _fits_name=_fits_name, _obs=_obs,
                        _path_out=path_out, _plate_scale=plate_scale,
                        _method='sextractor', _x=None, _y=None, _drizzled=_drizzled)
                # update database entry
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.{:s}.pca.last_modified'.format(_pipe): utc_now()
                        },
                        '$inc': {
                            'pipelined.{:s}.pca.status.retries'.format(_pipe): 1
                        }
                    }
                )
                _logger.info('put a PCA job into the queue for {:s}'.format(_obs))

    except Exception as _e:
        traceback.print_exc()
        _logger.error(_e)
        return False

    return True


def check_preview(_config, _logger, _coll, _select, _date, _obs, _pipe='automated'):
    """
        Make pipeline preview images
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name
    :param _pipe: which pipelined data to use? 'automated' or 'faint'?

    :return:
    """

    try:
        # preview_done = False, done = True, tried not too many times
        # OR preview_done = True, done = True, but a new Strehl is available, tried not too many times
        if (not _select['pipelined'][_pipe]['preview']['done'] and
                _select['pipelined'][_pipe]['status']['done'] and
                _select['pipelined'][_pipe]['preview']['retries']
                    < _config['max_pipelining_retries'])\
                or (_select['pipelined'][_pipe]['preview']['done'] and
                    _select['pipelined'][_pipe]['preview']['done'] and
                    _select['pipelined'][_pipe]['strehl']['last_modified'] >
                    _select['pipelined'][_pipe]['preview']['last_modified'] and
                    _select['pipelined'][_pipe]['preview']['retries']
                            < _config['max_pipelining_retries']):
            if _pipe == 'automated':
                # check if actually processed through pipeline
                path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                                 tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
                                 os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
            elif _pipe == 'faint':
                raise NotImplemented()
                # path_obs_list = []
            else:
                raise NotImplemented()
                # path_obs_list = []

            # processed?
            if len(path_obs_list) == 1:
                # this also considers the pathological case when an obs ended up in several classes
                path_obs = path_obs_list[0]

                # what's in a fits name?
                if _pipe == 'automated':
                    f_fits = os.path.join(path_obs, '100p.fits')
                elif _pipe == 'faint':
                    raise NotImplemented()
                else:
                    raise NotImplemented()

                # this follows the definition from structure.md
                path_out = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'preview')

                try:
                    # load first image frame from the fits file
                    preview_img = load_fits(f_fits)
                    # scale with local contrast optimization for preview:
                    if _select['science_program']['program_id'] != _config['planets_prog_num']:
                        preview_img = scale_image(preview_img, correction='local')
                        # cropped image [_win=None to try to detect]
                        preview_img_cropped, _x, _y = trim_frame(_path=path_obs,
                                                                 _fits_name=os.path.split(f_fits)[1],
                                                                 _win=None, _method='sextractor',
                                                                 _x=None, _y=None, _drizzled=True)
                    else:
                        # don't crop planets
                        preview_img_cropped = preview_img
                        _x, _y = None, None
                    # Strehl ratio (if available, otherwise will be None)
                    SR = _select['pipelined'][_pipe]['strehl']['ratio_percent']

                    _pix_x = int(re.search(r'(:)(\d+)',
                                   _select['pipelined'][_pipe]['fits_header']['DETSIZE'][0]).group(2))

                    _status = generate_pipe_preview(path_out, _obs, preview_img, preview_img_cropped,
                                                    SR, _fow_x=36, _pix_x=_pix_x, _drizzled=True,
                                                    _x=_x, _y=_y)

                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.{:s}.preview.done'.format(_pipe): _status,
                                'pipelined.{:s}.preview.last_modified'.format(_pipe): utc_now()
                            },
                            '$inc': {
                                'pipelined.{:s}.preview.retries'.format(_pipe): 1
                            }
                        }
                    )
                    if _status:
                        _logger.info('Generated lucky pipeline entry [preview] for {:s}'.format(_obs))
                    else:
                        _logger.error('Failed to generate lucky pipeline preview images for {:s}'.format(_obs))

                except Exception as _e:
                    traceback.print_exc()
                    _logger.error(_e)
                    return False

        # following structure.md:
        path_preview = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'preview')

        # path does not exist? make sure it's not marked 'done'
        if not os.path.exists(path_preview) and \
                _select['pipelined'][_pipe]['preview']['done']:
            # update database entry if incorrectly marked 'done'
            # (could not find the respective directory)
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.{:s}.preview.done'.format(_pipe): False,
                        'pipelined.{:s}.preview.last_modified'.format(_pipe): utc_now()
                    }
                }
            )
            _logger.info('Corrected {:s} pipeline preview entry for {:s}'.format(_pipe, _obs))

    except Exception as _e:
        traceback.print_exc()
        _logger.error(_e)
        return False

    return True


def check_pca_preview(_config, _logger, _coll, _select, _date, _obs, _pipe='automated'):
    """
        Make PCA pipeline preview images
    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name
    :param _pipe: which pipelined data to use? 'automated' or 'faint'?

    :return:
    """

    try:
        # preview_done = False, done = True, tried not too many times
        if not _select['pipelined'][_pipe]['pca']['preview']['done'] and \
                _select['pipelined'][_pipe]['pca']['status']['done'] and \
                _select['pipelined'][_pipe]['pca']['preview']['retries'] \
                        < _config['max_pipelining_retries']:
            # following structure.md:
            path_pca = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'pca')

            # processed?
            if os.path.exists(path_pca):

                # what's in a fits name?
                f_fits = os.path.join(path_pca, '{:s}_pca.fits'.format(obs))

                # load first image frame from the fits file
                preview_img = load_fits(f_fits)
                # scale with local contrast optimization for preview:
                preview_img = scale_image(preview_img, correction='local')

                # contrast_curve:
                _cc = _select['pipelined'][_pipe]['pca']['contrast_curve']

                _pix_x = int(re.search(r'(:)(\d+)',
                                   _select['pipelined'][_pipe]['fits_header']['DETSIZE'][0]).group(2))

                _status = generate_pca_images(_path_out=path_pca,
                                              _obs=_obs, _preview_img=preview_img,
                                              _cc=_cc, _fow_x=36, _pix_x=_pix_x, _drizzled=True)

                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.{:s}.pca.preview.done'.format(_pipe): _status,
                            'pipelined.{:s}.pca.preview.last_modified'.format(_pipe): utc_now()
                        },
                        '$inc': {
                            'pipelined.{:s}.pca.preview.retries'.format(_pipe): 1
                        }
                    }
                )
                if _status:
                    _logger.info('Generated PCA pipeline entry [preview] for {:s}'.format(_obs))
                else:
                    _logger.error('Failed to generate PCA pipeline preview images for {:s}'.format(_obs))

        # following structure.md:
        path_pca = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'pca')

        # path does not exist? make sure it's not marked 'done'
        if not os.path.exists(path_pca) and \
                _select['pipelined'][_pipe]['pca']['preview']['done']:
            # update database entry if incorrectly marked 'done'
            # (could not find the respective directory)
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'pipelined.{:s}.pca.preview.done'.format(_pipe): False,
                        'pipelined.{:s}.pca.preview.last_modified'.format(_pipe): utc_now()
                    }
                }
            )
            _logger.info('Corrected {:s} PCA pipeline preview entry for {:s}'.format(_pipe, _obs))

    except Exception as _e:
        traceback.print_exc()
        _logger.error(_e)
        return False

    return True


def parse_obs_name(_obs, _program_pi):
    """
        Parse Robo-AO observation name
    :param _obs:
    :param _program_pi: dict program_num -> PI
    :return:
    """
    # parse name:
    _tmp = _obs.split('_')
    # program num. it will be a string in the future
    _prog_num = str(_tmp[0])
    # who's pi?
    if _prog_num in _program_pi.keys():
        _prog_pi = _program_pi[_prog_num]
    else:
        # play safe if pi's unknown:
        _prog_pi = 'admin'
    # stack name together if necessary (if contains underscores):
    _sou_name = '_'.join(_tmp[1:-5])
    # code of the filter used:
    _filt = _tmp[-4:-3][0]
    # date and time of obs:
    _date_utc = datetime.datetime.strptime(_tmp[-2] + _tmp[-1], '%Y%m%d%H%M%S.%f')
    # camera:
    _camera = _tmp[-5:-4][0]
    # marker:
    _marker = _tmp[-3:-2][0]

    return _prog_num, _prog_pi, _sou_name, _filt, _date_utc, _camera, _marker


def init_db_entry(_obs, _sou_name, _prog_num, _prog_pi, _date_utc, _camera, _filt, _date_files):
    """
        Initialize a database entry
    :param _obs:
    :param _sou_name:
    :param _prog_num:
    :param _prog_pi:
    :param _date_utc:
    :param _camera:
    :param _filt:
    :param _date_files:
    :return:
    """

    _entry = empty_db_record()
    # populate:
    _entry['_id'] = _obs
    _entry['name'] = _sou_name
    _entry['science_program']['program_id'] = _prog_num
    _entry['science_program']['program_PI'] = _prog_pi
    _entry['date_utc'] = _date_utc
    if _date_utc > datetime.datetime(2015, 10, 1):
        _entry['telescope'] = 'KPNO_2.1m'
    else:
        _entry['telescope'] = 'Palomar_P60'
    _entry['camera'] = _camera
    _entry['filter'] = _filt  # also get this from FITS header

    # find raw fits files:
    _raws = [_s for _s in _date_files if re.match(_obs, _s) is not None]
    _entry['raw_data']['location'].append(['{:s}:{:s}'.format(
        config['analysis_machine_external_host'],
        config['analysis_machine_external_port']),
        config['path_raw']])
    _entry['raw_data']['data'] = _raws
    _entry['raw_data']['last_modified'] = datetime.datetime.now(pytz.utc)

    return _entry


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

        ''' check connection to redis server that processes pipeline tasks '''
        try:
            pubsub = huey.storage.listener()
            logger.debug('Successfully connected to the redis server at 127.0.0.1:6379')
        except ConnectionError as e:
            logger.error(e)
            logger.error('Redis server not responding')
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
            # print(date_obs)
            # TODO: handle seeing files separately [lower priority]
            date_seeing = [re.split(pattern_end, s)[0] for s in date_files
                           if re.search(pattern_end, s) is not None and
                           re.match('seeing_', s) is not None]
            # print(date_seeing)
            # for each source name see if there's an entry in the database
            for obs in date_obs:
                print('processing {:s}'.format(obs))
                logger.debug('processing {:s}'.format(obs))

                # parse observation name
                prog_num, prog_pi, sou_name, \
                    filt, date_utc, camera, marker = parse_obs_name(obs, program_pi)

                # look up entry in the database:
                select = coll.find_one({'_id': obs})
                # if entry not in database, create empty one and populate it
                if select is None:
                    print('{:s} not in database, adding...'.format(obs))
                    logger.info('{:s} not in database, adding'.format(obs))

                    # initialize db entry and populate it
                    entry = init_db_entry(_obs=obs, _sou_name=sou_name,
                                          _prog_num=prog_num, _prog_pi=prog_pi,
                                          _date_utc=date_utc, _camera=camera,
                                          _filt=filt, _date_files=date_files)
                    # insert it into database
                    result = coll.insert_one(entry)
                    # and select it:
                    select = coll.find_one({'_id': obs})

                # entry found in database, check if pipelined, update entry if necessary
                else:
                    print('{:s} in database, checking...'.format(obs))

                # proceed immediately
                ''' check lucky-pipelined data '''
                # Strehl and PCA are checked from within check_pipe_automated
                status_ok = check_pipe_automated(_config=config, _logger=logger, _coll=coll,
                                                 _select=select, _date=date, _obs=obs)
                if not status_ok:
                    logger.error('Checking failed for lucky pipeline: {:s}'.format(obs))

                ''' check faint-pipelined data '''
                if True is False:
                    # TODO: if core and halo tell you it's faint, run faint pipeline:
                    status_ok = check_pipe_faint(_config=config, _logger=logger, _coll=coll,
                                                 _select=select, _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for faint pipeline: {:s}'.format(obs))

                # TODO: if it is a planetary observation, run the planetary pipeline
                ''' check planetary-pipelined data '''

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
