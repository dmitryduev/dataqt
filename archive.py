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
import glob
from distutils.dir_util import copy_tree
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
import pyprind
import subprocess
from scipy.stats import sigmaclip
from scipy.ndimage import gaussian_filter
import image_registration
import functools
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from skimage import exposure, img_as_float
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit
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


def memoize(f):
    """ Minimalistic memoization decorator.
    http://code.activestate.com/recipes/577219-minimalistic-memoization/ """

    cache = {}

    @functools.wraps(f)
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf


class Star(object):
    """ Define a star by its coordinates and modelled FWHM
        Given the coordinates of a star within a 2D array, fit a model to the star and determine its
        Full Width at Half Maximum (FWHM).The star will be modelled using astropy.modelling. Currently
        accepted models are: 'Gaussian2D', 'Moffat2D'
    """

    _GAUSSIAN2D = 'Gaussian2D'
    _MOFFAT2D = 'Moffat2D'
    # _MODELS = set([_GAUSSIAN2D, _MOFFAT2D])

    def __init__(self, x0, y0, data, model_type=_GAUSSIAN2D, box=100, fow_x=36, out_path='./'):
        """ Instantiation method for the class Star.
        The 2D array in which the star is located (data), together with the pixel coordinates (x0,y0) must be
        passed to the instantiation method. .
        """
        self.x = x0
        self.y = y0
        self._box = box
        # field of view in x in arcsec:
        self._fow_x = fow_x
        self._pix_x = data.shape[0]
        self._XGrid, self._YGrid = self._grid_around_star(x0, y0, data)
        self.data = data[self._XGrid, self._YGrid]
        self.model_type = model_type
        self.out_path = out_path

    def model(self):
        """ Fit a model to the star. """
        return self._fit_model()

    @property
    def model_psf(self):
        """ Return a modelled PSF for the given model  """
        return self.model()(self._XGrid, self._YGrid)

    @property
    def fwhm(self):
        """ Extract the FWHM from the model of the star.
            The FWHM needs to be calculated for each model. For the Moffat, the FWHM is a function of the gamma and
            alpha parameters (in other words, the scaling factor and the exponent of the expression), while for a
            Gaussian FWHM = 2.3548 * sigma. Unfortunately, our case is a 2D Gaussian, so a compromise between the
            two sigmas (sigma_x, sigma_y) must be reached. We will use the average of the two.
        """
        model_dict = dict(zip(self.model().param_names, self.model().parameters))
        if self.model_type == self._MOFFAT2D:
            gamma, alpha = [model_dict[ii] for ii in ("gamma_0", "alpha_0")]
            FWHM = 2. * gamma * np.sqrt(2 ** (1/alpha) -1)
            FWHM_x, FWHM_y = None, None
        elif self.model_type == self._GAUSSIAN2D:
            sigma_x, sigma_y = [model_dict[ii] for ii in ("x_stddev_0", "y_stddev_0")]
            FWHM = 2.3548 * np.mean([sigma_x, sigma_y])
            FWHM_x, FWHM_y = 2.3548 * sigma_x, 2.3548 * sigma_y
        return FWHM, FWHM_x, FWHM_y

    @memoize
    def _fit_model(self):
        fit_p = fitting.LevMarLSQFitter()
        model = self._initialize_model()
        _p = fit_p(model, self._XGrid, self._YGrid, self.data)
        return _p

    def _initialize_model(self):
        """ Initialize a model with first guesses for the parameters.
        The user can select between several astropy models, e.g., 'Gaussian2D', 'Moffat2D'. We will use the data to get
        the first estimates of the parameters of each model. Finally, a Constant2D model is added to account for the
        background or sky level around the star.
        """
        max_value = self.data.max()

        if self.model_type == self._GAUSSIAN2D:
            model = models.Gaussian2D(x_mean=self.x, y_mean=self.y, x_stddev=1, y_stddev=1)
            model.amplitude = max_value

            # Establish reasonable bounds for the fitted parameters
            model.x_stddev.bounds = (0, self._box/4)
            model.y_stddev.bounds = (0, self._box/4)
            model.x_mean.bounds = (self.x - 5, self.x + 5)
            model.y_mean.bounds = (self.y - 5, self.y + 5)

        elif self.model_type == self._MOFFAT2D:
            model = models.Moffat2D()
            model.x_0 = self.x
            model.y_0 = self.y
            model.gamma = 2
            model.alpha = 2
            model.amplitude = max_value

            #  Establish reasonable bounds for the fitted parameters
            model.alpha.bounds = (1,6)
            model.gamma.bounds = (0, self._box/4)
            model.x_0.bounds = (self.x - 5, self.x + 5)
            model.y_0.bounds = (self.y - 5, self.y + 5)

        model += models.Const2D(self.fit_sky())
        model.amplitude_1.fixed = True
        return model

    def fit_sky(self):
        """ Fit the sky using a Ring2D model in which all parameters but the amplitude are fixed.
        """
        min_value = self.data.min()
        ring_model = models.Ring2D(min_value, self.x, self.y, self._box * 0.4, width=self._box * 0.4)
        ring_model.r_in.fixed = True
        ring_model.width.fixed = True
        ring_model.x_0.fixed = True
        ring_model.y_0.fixed = True
        fit_p = fitting.LevMarLSQFitter()
        return fit_p(ring_model, self._XGrid, self._YGrid, self.data).amplitude

    def _grid_around_star(self, x0, y0, data):
        """ Build a grid of side 'box' centered in coordinates (x0,y0). """
        lenx, leny = data.shape
        xmin, xmax = max(x0-self._box/2, 0), min(x0+self._box/2+1, lenx-1)
        ymin, ymax = max(y0-self._box/2, 0), min(y0+self._box/2+1, leny-1)
        return np.mgrid[int(xmin):int(xmax), int(ymin):int(ymax)]

    def plot_resulting_model(self, frame_name):
        """ Make a plot showing data, model and residuals. """
        data = self.data
        model = self.model()(self._XGrid, self._YGrid)
        _residuals = data - model

        bar_len = data.shape[0] * 0.1
        bar_len_str = '{:.1f}'.format(bar_len * self._fow_x / self._pix_x)

        fig = plt.figure(figsize=(9, 3))
        # data
        ax1 = fig.add_subplot(1, 3, 1)
        # print(sns.diverging_palette(10, 220, sep=80, n=7))
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.magma)
        # ax1.title('Data', fontsize=14)
        ax1.grid('off')
        ax1.set_axis_off()

        asb = AnchoredSizeBar(ax1.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax1.add_artist(asb)

        # model
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.imshow(model, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.magma)
        # RdBu_r, magma, inferno, viridis
        ax2.set_axis_off()
        # ax2.title('Model', fontsize=14)
        ax2.grid('off')

        asb = AnchoredSizeBar(ax2.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax2.add_artist(asb)

        # residuals
        ax3 = fig.add_subplot(1, 3, 3, sharey=ax1)
        ax3.imshow(_residuals, origin='lower', interpolation='nearest', cmap=plt.cm.magma)
        # ax3.title('Residuals', fontsize=14)
        ax3.grid('off')
        ax3.set_axis_off()

        asb = AnchoredSizeBar(ax3.transData,
                              bar_len,
                              bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                              loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        ax3.add_artist(asb)

        # plt.tight_layout()

        # dancing with a tambourine to remove the white spaces on the plot:
        fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1, left=0)
        plt.margins(0, 0)
        from matplotlib.ticker import NullLocator
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        fig.savefig(os.path.join(self.out_path, '{:s}.png'.format(frame_name)), dpi=200)
        # plt.show()


def process_seeing(_path_in, _seeing_frame, _path_calib, _path_out,
                   _frame_size_x_arcsec=36, _fit_model='Gaussian2D', _box_size=100):
    # parse observation name
    _, _, _, _filt, _date_utc, _, _ = parse_obs_name('9999_' + _seeing_frame, {})
    _mode = get_mode(os.path.join(_path_in, '{:s}.fits'.format(_seeing_frame)))

    # load darks and flats
    dark, flat = load_darks_and_flats(_path_calib, _mode, _filt)
    if dark is None or flat is None:
        raise Exception('Could not open darks and flats')

    with fits.open(os.path.join(_path_in, '{:s}.fits'.format(_seeing_frame))) as _hdulist:
        # get image size (this would be (1024, 1024) for the Andor camera)
        image_size = _hdulist[0].shape
        # number of frames in the data cube:
        nf = len(_hdulist)
        # Stack to seeing-limited image
        summed_seeing_limited_frame = np.zeros((image_size[0], image_size[1]), dtype=np.float)
        for ii, _ in enumerate(_hdulist):
            # im_tmp = np.array(_hdulist[ii].data, dtype=np.float)
            # im_tmp = calibrate_frame(im_tmp, dark, flat, _iter=2)
            # im_tmp = gaussian_filter(im_tmp, sigma=5)
            # summed_seeing_limited_frame += im_tmp
            summed_seeing_limited_frame += _hdulist[ii].data

        #
        summed_seeing_limited_frame = calibrate_frame(summed_seeing_limited_frame / nf, dark, flat, _iter=2)
        summed_seeing_limited_frame = gaussian_filter(summed_seeing_limited_frame, sigma=5)  # 5, 10

    # dump fits for sextraction:
    _fits_stacked = '{:s}.summed.fits'.format(_seeing_frame)
    try:
        export_fits(os.path.join(_path_in, _fits_stacked), summed_seeing_limited_frame)

        _, x, y = trim_frame(_path_in, _fits_name=_fits_stacked,
                             _win=_box_size, _method='sextractor', _x=None, _y=None, _drizzled=False)
        print('centroid position: ', x, y)

        # remove fits:
        os.remove(os.path.join(_path_in, _fits_stacked))

        centroid = Star(x, y, summed_seeing_limited_frame, model_type=_fit_model, box=_box_size,
                        fow_x=_frame_size_x_arcsec, out_path=os.path.join(_path_out, 'seeing'))
        seeing, seeing_x, seeing_y = centroid.fwhm
        print('Estimated seeing = {:.3f} pixels'.format(seeing))
        print('Estimated seeing = {:.3f}\"'.format(seeing * _frame_size_x_arcsec / image_size[0]))
        # plot image, model, and residuals:
        centroid.plot_resulting_model(frame_name=_seeing_frame)

        return _date_utc, seeing * _frame_size_x_arcsec / image_size[0], seeing, _filt, \
                seeing_x * _frame_size_x_arcsec / image_size[0], seeing_y * _frame_size_x_arcsec / image_size[0]

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        if os.path.exists(os.path.join(_path_in, _fits_stacked)):
            # remove fits:
            os.remove(os.path.join(_path_in, _fits_stacked))
        return _date_utc, None, None, _filt, None, None


def inqueue(job_type, *_args):
    """
        Check if a job has already been enqueued

    :param job_type: string corresponding to the name of function decorated with @huey.task()
    :param args: list of strings. all must be in the huey-redis task string representation
                 do print(pending) to see what I mean by that
    :return:
    """
    pending = huey.get_storage().enqueued_items()
    # print(huey.pending())

    if len(pending) > 0:
        isin = len([task for task in pending if (job_type in task) and all([_s in task for _s in _args])]) >= 1
    else:
        isin = False
    return isin


@huey.task()
# @numba.jit
def job_faint_pipeline(_config, _raws_zipped, _date, _obs, _path_out):
    """
        The task that runs the faint pipeline
    :param _config:
    :param _raws_zipped:
    :param _date:
    :param _obs:
    :param _path_out:
    :return:
    """
    try:
        # path to store unzipped raw files
        _path_tmp = _config['path_tmp']
        # path to raw files:
        _path_in = os.path.join(_config['path_raw'], _date)
        # path to lucky-pipelined data:
        _path_lucky = os.path.join(_config['path_pipe'], _date)
        # path to calibration data produced by lucky pipeline:
        _path_calib = os.path.join(_config['path_pipe'], _date, 'calib')

        # zipped raw files:
        # raws_zipped = sorted([_f for _f in os.listdir(_path_in) if _obs in _f])[0:]
        # print(raws_zipped)

        # unbzip source file(s):
        lbunzip2(_path_in=_path_in, _files=_raws_zipped, _path_out=_path_tmp, _keep=True)

        # unzipped file names:
        raws = [os.path.splitext(_f)[0] for _f in _raws_zipped]
        # print('\n', raws)

        tag = [tag for tag in ('high_flux', 'faint', 'zero_flux', 'failed') if
               os.path.exists(os.path.join(_path_lucky, tag, _obs))][0]

        # get lock position and (square) window size
        if tag in ('high_flux', 'faint'):
            x_lock, y_lock = \
                get_xy_from_pipeline_settings_txt(os.path.join(_path_lucky, tag, _obs))

            win = int(np.min([_config['faint']['win'], x_lock, y_lock]))
            # use highest-Strehl frame to align individual frames to:
            pivot = get_best_pipe_frame(os.path.join(_path_lucky, tag, _obs))
        else:
            # zero flux or failed? try the whole (square) image (or the largest square subset of it)
            with fits.open(os.path.join(_path_tmp, raws[0])) as tmp_fits:
                # print(hdulist[0].header)
                tmp_header = tmp_fits[0].header
                x_lock = tmp_header.get('NAXIS1') // 2
                y_lock = tmp_header.get('NAXIS2') // 2
            # window must be square:
            win = int(np.min([x_lock, y_lock]))
            # use seeing-limited image to align individual frames to:
            pivot = (-1, -1)

        # print('Initial lock position: ', x_lock, y_lock)

        # parse observation name
        _, _, _, _filt, _, _, _ = parse_obs_name(_obs, {})

        _mode = get_mode(os.path.join(_path_tmp, raws[0]))

        reduce_faint_object_noram(_path_in=_path_tmp, _files=raws,
                                  _path_calib=_path_calib, _path_out=_path_out,
                                  _obs=_obs, _mode=_mode, _filt=_filt, _win=win, cy0=y_lock, cx0=x_lock,
                                  _pivot=pivot,
                                  _nthreads=_config['faint']['n_threads'],
                                  _remove_tmp=True, _v=True, _interactive_plot=False)

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        # return False
        return {'job_type': 'faint', 'status': 'failed', 'obs': _obs}

    # return True
    return {'job_type': 'faint', 'status': 'success', 'obs': _obs}


@huey.task()
# @numba.jit
def job_pca(_config, _path_in, _fits_name, _obs, _path_out,
            _plate_scale, _method='sextractor', _x=None, _y=None, _drizzled=True):
    """
        The task that runs the PCA pipeline
    :param _config:
    :param _path_in:
    :param _fits_name:
    :param _obs:
    :param _path_out:
    :param _plate_scale:
    :param _method:
    :param _x:
    :param _y:
    :param _drizzled:
    :return:
    """
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
        # return False
        return {'job_type': 'pca', 'status': 'failed', 'obs': _obs}

    # return True
    return {'job_type': 'pca', 'status': 'success', 'obs': _obs}


@huey.task()
def job_strehl(_path_in, _fits_name, _obs, _path_out, _plate_scale, _Strehl_factor,
               _method='pipeline_settings.txt', _win=100, _x=None, _y=None, _drizzled=True,
               _core_min=0.14, _halo_max=1.0):
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
        # return False
        return {'job_type': 'strehl', 'status': 'failed', 'obs': _obs}

    if core >= _core_min and halo <= _halo_max:
        flag = 'OK'
    else:
        flag = 'BAD?'

    # print(core, halo, SR*100, FWHM)

    try:
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

        # return True
        return {'job_type': 'strehl', 'status': 'success', 'obs': _obs}

    except Exception as _e:
        print(_obs, _e)
        return {'job_type': 'strehl', 'status': 'failed', 'obs': _obs}


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


def naptime(_config):
    """
        Return time to sleep in seconds for the archiving engine
        before waking up to rerun itself.
         In the daytime, it's 1 hour
         In the nap time, it's nap_time_start_utc - utc_now()
    :return:
    """
    try:
        # local or UTC?
        tz = pytz.utc if _config['nap_time']['frame'] == 'UTC' else None
        now = datetime.datetime.now(tz)

        if _config['nap_time']['sleep_at_night']:

            last_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz)
            next_midnight = datetime.datetime(now.year, now.month, now.day, tzinfo=tz) \
                            + datetime.timedelta(days=1)

            hm_start = map(int, _config['nap_time']['start'].split(':'))
            hm_stop = map(int, _config['nap_time']['stop'].split(':'))

            if hm_stop[0] < hm_start[0]:
                h_before_midnight = 24 - (hm_start[0] + hm_start[1] / 60.0)
                h_after_midnight = hm_stop[0] + hm_stop[1] / 60.0

                # print((next_midnight - now).total_seconds() / 3600.0, h_before_midnight)
                # print((now - last_midnight).total_seconds() / 3600.0, h_after_midnight)

                if (next_midnight - now).total_seconds() / 3600.0 < h_before_midnight:
                    sleep_until = next_midnight + datetime.timedelta(hours=h_after_midnight)
                    print('sleep until:', sleep_until)
                elif (now - last_midnight).total_seconds() / 3600.0 < h_after_midnight:
                    sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight)
                    print('sleep until:', sleep_until)
                else:
                    sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                    print('sleep until:', sleep_until)

            else:
                h_after_midnight_start = hm_start[0] + hm_start[1] / 60.0
                h_after_midnight_stop = hm_stop[0] + hm_stop[1] / 60.0

                if (last_midnight + datetime.timedelta(hours=h_after_midnight_start) <
                        now < last_midnight + datetime.timedelta(hours=h_after_midnight_stop)):
                    sleep_until = last_midnight + datetime.timedelta(hours=h_after_midnight_stop)
                    print('sleep until:', sleep_until)
                else:
                    sleep_until = now + datetime.timedelta(minutes=_config['loop_interval'])
                print('sleep until:', sleep_until)

            return (sleep_until - now).total_seconds()

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        # return _config['loop_interval']*60
        # sys.exit()
        return False


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
    # skip empty lines (if accidentally present in the file)
    f_lines = [_l for _l in f_lines if len(_l) > 1]

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
    # skip empty lines (if accidentally present in the file)
    f_lines = [_l for _l in f_lines if len(_l) > 1]

    _tmp = f_lines[0].split()
    x_lock, y_lock = int(_tmp[-2]), int(_tmp[-1])

    cc = []
    for l in f_lines[1:]:
        cc.append(map(float, l.split()))

    return x_lock, y_lock, cc


def load_faint_shifts(fin):
    """
        Load shifts.txt produced by the faint pipeline
        Format: first line: lock position (x, y) on the original frame
                second line: format descriptor
                next lines: frame_number x_shift[pix] y_shift[pix] ex_shift[pix] ey_shift[pix]
    :param fin:
    :return:
    """
    with open(fin) as _f:
        f_lines = _f.readlines()
    # skip empty lines (if accidentally present in the file)
    f_lines = [_l for _l in f_lines if len(_l) > 1]

    _tmp = f_lines[0].split()
    x_lock, y_lock = int(_tmp[-2]), int(_tmp[-1])

    shifts = []
    for l in f_lines[2:]:
        _tmp = l.split()
        shifts.append([int(_tmp[0])] + map(float, _tmp[1:]))

    return x_lock, y_lock, shifts


def lbunzip2(_path_in, _files, _path_out, _keep=True, _v=False):

    """
        A wrapper around lbunzip2 - a parallel version of bunzip2
    :param _path_in: folder with the files to be unzipped
    :param _files: string or list of strings with file names to be uncompressed
    :param _path_out: folder to place the output
    :param _keep: keep the original?
    :return:
    """

    try:
        p0 = subprocess.Popen(['lbunzip2'])
        p0.wait()
    except Exception as _e:
        print(_e)
        print('lbzip2 not installed in the system. go ahead and install it!')
        return False

    if isinstance(_files, str):
        _files_list = [_files]
    else:
        _files_list = _files

    files_size = sum([os.stat(os.path.join(_path_in, fs)).st_size for fs in _files_list])
    # print(files_size)

    if _v:
        bar = pyprind.ProgBar(files_size, stream=1, title='Unzipping files', monitor=True)
    for _file in _files_list:
        file_in = os.path.join(_path_in, _file)
        file_out = os.path.join(_path_out, os.path.splitext(_file)[0])
        if os.path.exists(file_out):
            # print('uncompressed file {:s} already exists, skipping'.format(file_in))
            if _v:
                bar.update(iterations=os.stat(file_in).st_size)
            continue
        # else go ahead
        # print('lbunzip2 <{:s} >{:s}'.format(file_in, file_out))
        with open(file_in, 'r') as _f_in, open(file_out, 'w') as _f_out:
            _p = subprocess.Popen('lbunzip2'.split(), stdin=subprocess.PIPE,
                                 stdout=_f_out)
            _p.communicate(input=_f_in.read())
            # wait for it to finish
            _p.wait()
        # remove the original if requested:
        if not _keep:
            _p = subprocess.Popen(['rm', '-f', '{:s}'.format(os.path.join(_path_in, _file))])
            # wait for it to finish
            _p.wait()
            # pass
        if _v:
            bar.update(iterations=os.stat(file_in).st_size)

    return True


def get_mode(_fits):
    header = get_fits_header(_fits)
    return str(header['MODE_NUM'][0])


def load_darks_and_flats(_path_calib, _mode, _filt, image_size_x=1024):
    """
        Load darks and flats
    :param _path_calib:
    :param _mode:
    :param _filt:
    :param image_size:
    :return:
    """
    if image_size_x == 256:
        dark_image = os.path.join(_path_calib, 'dark_{:s}4.fits'.format(str(_mode)))
    else:
        dark_image = os.path.join(_path_calib, 'dark_{:s}.fits'.format(str(_mode)))
    flat_image = os.path.join(_path_calib, 'flat_{:s}.fits'.format(_filt))

    if not os.path.exists(dark_image) or not os.path.exists(flat_image):
        return None, None
    else:
        with fits.open(dark_image) as dark, fits.open(flat_image) as flat:
            # replace NaNs if necessary
            if image_size_x == 256:
                return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data[384:640, 384:640])
            else:
                return np.nan_to_num(dark[0].data), np.nan_to_num(flat[0].data)


# @jit
def calibrate_frame(im, _dark, _flat, _iter=3):
    im_BKGD = deepcopy(im)
    for j in range(int(_iter)):  # do 3 iterations of sigma-clipping
        try:
            temp = sigmaclip(im_BKGD, 3.0, 3.0)
            im_BKGD = temp[0]  # return arr is 1st element
        except Exception as _e:
            print(_e)
            pass
    sum_BKGD = np.mean(im_BKGD)  # average CCD BKGD
    im -= sum_BKGD
    im -= _dark
    im /= _flat

    return im


@jit
def shift2d(fftn, ifftn, data, deltax, deltay, xfreq_0, yfreq_0,
            return_abs=False, return_real=True):
    """
    2D version: obsolete - use ND version instead
    (though it's probably easier to parse the source of this one)

    FFT-based sub-pixel image shift.
    Will turn NaNs into zeros

    Shift Theorem:

    .. math::
        FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)


    Parameters
    ----------
    data : np.ndarray
        2D image
    """

    xfreq = deltax * xfreq_0
    yfreq = deltay * yfreq_0
    freq_grid = xfreq + yfreq

    kernel = np.exp(-1j*2*np.pi*freq_grid)

    result = ifftn( fftn(data) * kernel )

    if return_real:
        return np.real(result)
    elif return_abs:
        return np.abs(result)
    else:
        return result


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
    hdulist.writeto(path, clobber=True)


def image_center(_path, _fits_name, _x0=None, _y0=None, _win=None):

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
    # search everywhere in the image?
    if _x0 is None and _y0 is None and _win is None:
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
    # search around (_x0, _y0) in a window of width _win
    else:
        for sou in out['table'][0:10]:
            _r = rho(sou['X_IMAGE'], sou['Y_IMAGE'], x_0=_x0, y_0=_y0)
            if sou['FWHM_IMAGE'] > 1 and _r < _win:
                score = gauss_score(_r)
            else:
                score = 0  # it could so happen that reported FWHM is 0
            scores.append(score)

    # there was something to score? get the best score then
    if len(scores) > 0:
        best_score = np.argmax(scores)
        x_center = out['table']['YPEAK_IMAGE'][best_score]
        y_center = out['table']['XPEAK_IMAGE'][best_score]
    # somehow no sources detected? but _x0 and _y0 set? return the latter then
    elif _x0 is not None and _y0 is not None:
        x_center, y_center = _x0, _y0
    # no sources detected and _x0 and _y0 not set? return the simple maximum:
    else:
        scidata = fits.open(os.path.join(_path, _fits_name))[0].data
        x_center, y_center = np.unravel_index(scidata.argmax(), scidata.shape)

    return x_center, y_center


def reduce_faint_object_noram(_path_in, _files, _path_calib, _path_out, _obs,
                              _mode, _filt, _win, cy0, cx0, _pivot=(-1, -1),
                              _nthreads=1, _remove_tmp=True, _v=False, _interactive_plot=False):
    """

    :param _path_in: path to directory where to look for _files
    :param _files: names of unzipped fits-files
    :param _path_calib: path to lucky-pipe calibration data
    :param _path_out: where to place the pipeline output
    :param _obs: obs base name
    :param _mode: detector mode used in observation
    :param _filt: filter used
    :param _win: window size in pixels
    :param cy0: cut a window [+-_win] around this position to
    :param cx0: do image registration
    :param _pivot: raw file number (starting from zero) and frame number to align to
                  default: (-1, -1) means align to a seeing limited image
    :param _nthreads: number of threads to use in image registration
    :param _remove_tmp: remove unzipped fits-files if successfully finished processing?
    :param _v: verbose? [display progress bars and print statements]
    :param _interactive_plot: show interactively updated plot?
    :return:
    """
    try:
        if isinstance(_files, str):
            _files_list = [_files]
        else:
            _files_list = _files

        files_sizes = [os.stat(os.path.join(_path_in, fs)).st_size for fs in _files_list]

        # get total number of frames to allocate
        # bar = pyprind.ProgBar(sum(files_sizes), stream=1, title='Getting total number of frames')
        # number of frames in each fits file
        n_frames_files = []
        for jj, _file in enumerate(_files_list):
            with fits.open(os.path.join(_path_in, _file)) as _hdulist:
                if jj == 0:
                    # get image size (this would be (1024, 1024) for the Andor camera)
                    image_size = _hdulist[0].shape
                n_frames_files.append(len(_hdulist))
                # bar.update(iterations=files_sizes[jj])
        # total number of frames
        numFrames = sum(n_frames_files)

        # Stack to seeing-limited image
        if _v:
            bar = pyprind.ProgBar(sum(files_sizes), stream=1, title='Stacking to seeing-limited image')
        summed_seeing_limited_frame = np.zeros((image_size[0], image_size[1]), dtype=np.float)
        for jj, _file in enumerate(_files_list):
            # print(jj)
            with fits.open(os.path.join(_path_in, _file)) as _hdulist:
                # frames_before = sum(n_frames_files[:jj])
                for ii, _ in enumerate(_hdulist):
                    summed_seeing_limited_frame += _hdulist[ii].data
                    # print(ii + frames_before, '\n', _data[ii, :, :])
            if _v:
                bar.update(iterations=files_sizes[jj])

        # load darks and flats
        if _v:
            print('Loading darks and flats')
        dark, flat = load_darks_and_flats(_path_calib, _mode, _filt, image_size[0])
        if dark is None or flat is None:
            raise Exception('Could not open darks and flats')

        if _v:
            print('Total number of frames to be registered: {:d}'.format(numFrames))

        # Sum of all (properly shifted) frames (with not too large a shift and chi**2)
        summed_frame = np.zeros_like(summed_seeing_limited_frame, dtype=np.float)

        # Pick a frame to align to
        # seeing-limited sum of all frames:
        print(_pivot)
        if _pivot == (-1, -1):
            im1 = deepcopy(summed_seeing_limited_frame)
            print('using seeing-limited image as pivot frame')
        else:
            try:
                with fits.open(os.path.join(_path_in, _files_list[_pivot[0]])) as _hdulist:
                    im1 = np.array(_hdulist[_pivot[1]].data, dtype=np.float)
                print('using frame {:d} from raw fits-file #{:d} as pivot frame'.format(*_pivot[::-1]))
            except Exception as _e:
                print(_e)
                im1 = deepcopy(summed_seeing_limited_frame)
                print('using seeing-limited image as pivot frame')

        if _interactive_plot:
            plt.axes([0., 0., 1., 1.])
            plt.ion()
            plt.grid('off')
            plt.axis('off')
            plt.show()

        # print(im1.shape, dark.shape, flat.shape)
        im1 = calibrate_frame(im1, dark, flat, _iter=3)
        im1 = gaussian_filter(im1, sigma=5)  # 5, 10
        im1 = im1[cy0 - _win: cy0 + _win, cx0 - _win: cx0 + _win]

        # frame_num x y ex ey:
        shifts = np.zeros((numFrames, 5))

        # set up frequency grid for shift2d
        ny, nx = image_size
        xfreq_0 = np.fft.fftfreq(nx)[np.newaxis, :]
        yfreq_0 = np.fft.fftfreq(ny)[:, np.newaxis]

        fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(nthreads=_nthreads,
                                                                      use_numpy_fft=False)

        if _v:
            bar = pyprind.ProgBar(numFrames, stream=1, title='Registering frames')

        fn = 0
        for jj, _file in enumerate(_files_list):
            with fits.open(os.path.join(_path_in, _file)) as _hdulist:
                # frames_before = sum(n_frames_files[:jj])
                for ii, _ in enumerate(_hdulist):
                    img = np.array(_hdulist[ii].data, dtype=np.float)  # do proper casting

                    # tic = _time()
                    img = calibrate_frame(img, dark, flat, _iter=3)
                    # print(_time()-tic)

                    # tic = _time()
                    img_comp = gaussian_filter(img, sigma=5)
                    img_comp = img_comp[cy0 - _win: cy0 + _win, cx0 - _win: cx0 + _win]
                    # print(_time() - tic)

                    # tic = _time()
                    # chi2_shift -> chi2_shift_iterzoom
                    dy2, dx2, edy2, edx2 = image_registration.chi2_shift(im1, img_comp, nthreads=_nthreads,
                                                                         upsample_factor='auto', zeromean=True)
                    # print(dx2, dy2, edx2, edy2)
                    # print(_time() - tic)
                    # tic = _time()
                    # note the order of dx and dy in shift2d vs shiftnd!!!
                    # img = image_registration.fft_tools.shiftnd(img, (-dx2, -dy2),
                    #                                            nthreads=_nthreads, use_numpy_fft=False)
                    img = shift2d(fftn, ifftn, img, -dy2, -dx2, xfreq_0, yfreq_0)
                    # print(_time() - tic, '\n')

                    # if np.sqrt(dx2 ** 2 + dy2 ** 2) > 0.8 * _win \
                    #     or np.sqrt(edx2 ** 2 + edy2 ** 2) > 0.5:
                    if np.sqrt(dx2 ** 2 + dy2 ** 2) > 0.8 * _win:
                        # skip frames with too large a shift
                        pass
                        # print(' # {:d} shift was too big: '.format(i),
                        #       np.sqrt(shifts[i, 1] ** 2 + shifts[i, 2] ** 2), shifts[i, 1], shifts[i, 2])
                    else:
                        # otherwise store the shift values and add to the 'integrated' image
                        shifts[fn, :] = [fn, -dx2, -dy2, edx2, edy2]
                        summed_frame += img

                    if _interactive_plot:
                        plt.imshow(summed_frame, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
                        plt.draw()
                        plt.pause(0.001)

                    if _v:
                        bar.update()

                    # increment frame number
                    fn += 1

        if _interactive_plot:
            raw_input('press any key to close plot')

        if _v:
            print('Largest move was {:.2f} pixels for frame {:d}'.
                  format(np.max(np.sqrt(shifts[:, 1] ** 2 + shifts[:, 2] ** 2)),
                    np.argmax(np.sqrt(shifts[:, 1] ** 2 + shifts[:, 2] ** 2))))

        # output
        if not os.path.exists(os.path.join(_path_out)):
            os.makedirs(os.path.join(_path_out))

        # get original fits header for output
        with fits.open(os.path.join(_path_in, _files[0])) as _hdulist:
            # header:
            header = _hdulist[0].header

        export_fits(os.path.join(_path_out, _obs + '_simple_sum.fits'),
                    summed_seeing_limited_frame, header)

        export_fits(os.path.join(_path_out, _obs + '_summed.fits'),
                    summed_frame, header)

        cyf, cxf = image_center(_path=_path_out, _fits_name=_obs + '_summed.fits',
                                _x0=cx0, _y0=cy0, _win=_win)
        print('Output lock position:', cxf, cyf)
        with open(os.path.join(_path_out, 'shifts.txt'), 'w') as _f:
            _f.write('# lock position: {:d} {:d}\n'.format(cxf, cyf))
            _f.write('# frame_number x_shift[pix] y_shift[pix] ex_shift[pix] ey_shift[pix]\n')
            for _i, _x, _y, _ex, _ey in shifts:
                _f.write('{:.0f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(_i, _x, _y, _ex, _ey))

        # # Set Vars for Estimated-background and Resolution Masks
        # R = 85  # inner cutout radius
        # w = 50  # annular BKGD width
        #
        # # Find Annular-cutout BKGD around Star and Subtract for Final Image
        # sky_BKGD = findAverageSkyBKGD(summed_frame, cyf, cxf, R, w)
        # sky_corrected_summed_frame = summed_frame - sky_BKGD
        # # sky_corrected_summed_frame = summed_frame  # - sky_BKGD
        #
        # export_fits(end_path + name + '_summed.fits', sky_corrected_summed_frame)
        # os.remove(start_path + name + '_all.fits')

    finally:
        # clean up if successfully finished
        if _remove_tmp:
            if _v:
                print('Removing unbzipped fits-files')
            _files_list = _files if not isinstance(_files, str) else [_files]
            for _file in _files_list:
                os.remove(os.path.join(_path_in, _file))


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
        ax.imshow(preview_img, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
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
        ax.imshow(preview_img_cropped, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
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
        ax.imshow(_preview_img, cmap=plt.cm.magma, origin='lower', interpolation='nearest')
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


def get_xy_from_shifts_txt(_path):
    with open(os.path.join(_path, 'shifts.txt')) as _f:
        f_lines = _f.readlines()
    # skip empty lines (if accidentally present in the file)
    f_lines = [_l for _l in f_lines if len(_l) > 1]

    _tmp = f_lines[0].split()
    x_lock, y_lock = int(_tmp[-2]), int(_tmp[-1])

    return x_lock, y_lock


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


def get_best_pipe_frame(_path):
    """
        Get file and frame numbers with the highest Strehl for lucky-pipelined data
    :param _path:
    :return:
    """
    try:
        with open(os.path.join(_path, 'sorted.txt'), 'r') as _f:
            f_lines = _f.readlines()
        _tmp = f_lines[1].split()[-1]
        frame_num = int(re.search(r'.fits\[(\d+)\]', _tmp).group(1))
        file_num_pattern = re.search(r'_(\d+).fits', _tmp)
        # is it the first file? its number 0:
        if file_num_pattern is None:
            file_num = 0
        # no? then from blabla_n.fits its number is n+1
        else:
            file_num = int(file_num_pattern.group(1)) + 1
        return file_num, frame_num
    except Exception as _e:
        print(_e)
        return -1, -1


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
            p = fmin(residuals, p0, args=(pix_rad, pix_vals), maxiter=1000, maxfun=1000,
                     ftol=1e-3, xtol=1e-3, disp=False)

        p0 = [0.0, np.max(core_pix_vals), 5.0, 2.0]
        core_p = fmin(residuals, p0, args=(core_pix_rad, core_pix_vals), maxiter=1000, maxfun=1000,
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
                    from 'pipeline_settings.txt' (if this is the output of the standard lucky pipeline),
                    from 'shifts.txt' (if this is the output of the faint pipeline),
                    using 'sextractor', a simple 'max', or 'manual'
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
    elif _method == 'shifts.txt':
        if _win is None:
            _win = 100
        y, x = get_xy_from_shifts_txt(_path)
        if _drizzled:
            x *= 2.0
            y *= 2.0
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


# @jit
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

    max_inds = np.where(box_roboao == np.max(box_roboao))

    centroid = Star(max_inds[1][0], max_inds[0][0], box_roboao)
    FWHM, FWHM_x, FWHM_y = centroid.fwhm
    fwhm_arcsec = FWHM * _plate_scale
    # print(FWHM, FWHM_x, FWHM_y)
    # print('image FWHM: {:.5f}\"\n'.format(fwhm_arcsec))

    # model = centroid.model()(centroid._XGrid, centroid._YGrid)
    # export_fits('1.fits', model)

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
                'program_PI': None
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
                        'force_redo': False,
                        'done': False,
                        'retries': 0,
                        'last_modified': time_now_utc
                    },
                    'location': [],
                    'classified_as': None,
                    'fits_header': {},
                    'strehl': {
                        'status': {
                            'force_redo': False,
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
                            'force_redo': False,
                            'done': False,
                            'retries': 0
                        },
                        'preview': {
                            'force_redo': False,
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
                        'force_redo': False,
                        'done': False,
                        'retries': 0
                    },
                    'preview': {
                        'force_redo': False,
                        'done': False,
                        'retries': 0,
                        'last_modified': time_now_utc
                    },
                    'location': [],
                    'lock_position': None,
                    'fits_header': {},
                    'shifts': None,
                    'strehl': {
                        'status': {
                            'force_redo': False,
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
                            'force_redo': False,
                            'done': False,
                            'retries': 0
                        },
                        'preview': {
                            'force_redo': False,
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
            'distributed': {
                'status': False,
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

    return _logger, utc_now.strftime('%Y%m%d')


def get_config(_abs_path=None, _config_file='config.ini'):
    """ Get config data

    :return:
    """
    config = ConfigParser.RawConfigParser()

    if _config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(_abs_path, _config_file)):
            config.read(os.path.join(_abs_path, _config_file))
            if len(config.read(os.path.join(_abs_path, _config_file))) == 0:
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
    # path to temporary stuff:
    _config['path_tmp'] = config.get('Path', 'path_tmp')

    # telescope data (voor, o.a., Strehl computation)
    _tmp = ast.literal_eval(config.get('Strehl', 'Strehl_factor'))
    _config['telescope_data'] = dict()
    for telescope in 'Palomar', 'KittPeak':
        _config['telescope_data'][telescope] = ast.literal_eval(config.get('Strehl', telescope))
        _config['telescope_data'][telescope]['Strehl_factor'] = _tmp[telescope]

    # metrics [arcsec] to judge if an observation is good/bad
    _config['core_min'] = float(config.get('Strehl', 'core_min'))
    _config['halo_max'] = float(config.get('Strehl', 'halo_max'))

    # faint pipeline:
    _config['faint'] = dict()
    _config['faint']['win'] = int(config.get('Faint', 'win'))
    _config['faint']['n_threads'] = int(config.get('Faint', 'n_threads'))

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

    # seeing
    _config['seeing'] = dict()
    _config['seeing']['fit_model'] = config.get('Seeing', 'fit_model')
    _config['seeing']['win'] = int(config.get('Seeing', 'win'))

    # database access:
    _config['mongo_host'] = config.get('Database', 'host')
    _config['mongo_port'] = int(config.get('Database', 'port'))
    _config['mongo_db'] = config.get('Database', 'db')
    _config['mongo_collection_obs'] = config.get('Database', 'collection_obs')
    _config['mongo_collection_aux'] = config.get('Database', 'collection_aux')
    _config['mongo_collection_pwd'] = config.get('Database', 'collection_pwd')
    _config['mongo_collection_weather'] = config.get('Database', 'collection_weather')
    _config['mongo_user'] = config.get('Database', 'user')
    _config['mongo_pwd'] = config.get('Database', 'pwd')

    # server ip addresses
    _config['analysis_machine_external_host'] = config.get('Server', 'analysis_machine_external_host')
    _config['analysis_machine_external_port'] = config.get('Server', 'analysis_machine_external_port')

    # consider data from:
    _config['archiving_start_date'] = datetime.datetime.strptime(
                            config.get('Misc', 'archiving_start_date'), '%Y/%m/%d')
    # how many times to try to rerun pipelines:
    _config['max_pipelining_retries'] = config.get('Misc', 'max_pipelining_retries')

    # nap time -- do not interfere with the nightly operations On/Off
    _config['nap_time'] = dict()
    _config['nap_time']['sleep_at_night'] = eval(config.get('Misc', 'nap_time'))
    # local or UTC?
    _config['nap_time']['frame'] = config.get('Misc', 'nap_time_frame')
    _config['nap_time']['start'] = config.get('Misc', 'nap_time_start')
    _config['nap_time']['stop'] = config.get('Misc', 'nap_time_stop')

    _config['loop_interval'] = float(config.get('Misc', 'loop_interval'))

    return _config


def connect_to_db(_config, _logger=None):
    """ Connect to the mongodb database

    :return:
    """
    try:
        _client = MongoClient(host=_config['mongo_host'], port=_config['mongo_port'])
        _db = _client[_config['mongo_db']]
        if _logger is not None:
            _logger.debug('Connecting to the Robo-AO database at {:s}:{:d}'.
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
        _coll_aux = _db[_config['mongo_collection_aux']]
        if _logger is not None:
            _logger.debug('Using collection {:s} with aux data in the database'.
                          format(_config['mongo_collection_aux']))
    except Exception as _e:
        _coll_aux = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with aux data in the database'.
                          format(_config['mongo_collection_aux']))
    try:
        _coll_weather = _db[_config['mongo_collection_weather']]
        if _logger is not None:
            _logger.debug('Using collection {:s} with KP weather data in the database'.
                          format(_config['mongo_collection_weather']))
    except Exception as _e:
        _coll_weather = None
        if _logger is not None:
            _logger.error(_e)
            _logger.error('Failed to use a collection {:s} with KP weather data in the database'.
                          format(_config['mongo_collection_weather']))
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

    if _logger is not None:
        _logger.debug('Successfully connected to the Robo-AO database at {:s}:{:d}'.
                      format(_config['mongo_host'], _config['mongo_port']))

    return _client, _db, _coll, _coll_aux, _coll_weather, _program_pi


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


def set_key(_logger, _coll, _obs, _key, _value):
    try:
        _coll.update_one(
            {'_id': _obs},
            {
                '$set': {
                    _key: _value
                }
            }
        )
    except Exception as _e:
        print(_e)
        traceback.print_exc()
        _logger.error('Setting {:s} key with {:s} failed for: {:s}\n{:s}'.format(_key, str(_value), _obs, _e))
        return False

    return True


def check_pipe_automated(_config, _logger, _coll, _select, _date, _obs):
    """
        Check if observation has been automatically lucky-pipelined

        If force_redo switch is on, no checks are performed
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
                            'exposure': float(header['EXPOSURE'][0])
                                          if ('EXPOSURE' in header and tag != 'failed') else None,
                            'magnitude': float(header['MAGNITUD'][0])
                                          if ('MAGNITUD' in header and tag != 'failed') else None,
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

                # zero flux? try faint pipeline! update: don't do that! this never works, but takes a lot of time
                # if tag == 'zero_flux':
                #     _coll.update_one(
                #         {'_id': _obs},
                #         {
                #             '$set': {
                #                 'pipelined.faint.status.force_redo': True,
                #             }
                #         }
                #     )

                _logger.debug('Updated automated pipeline entry for {:s}'.format(_obs))

            # check on Strehl:
            check_strehl(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                         _date, _obs, _pipe='automated')
            # check on PCA
            # reload entry from db:
            _select = _coll.find_one({'_id': _obs})
            # Strehl done and ok? then proceed:
            # FIXME: _config['core_min']*0.1 this is to force PSF subtraction on "lucky-asteroids"
            # FIXME: _config['halo_max']*1.5 this is to force PSF subtraction on "lucky-asteroids"
            if _select['pipelined']['automated']['pca']['status']['force_redo'] or \
                    (_select['pipelined']['automated']['strehl']['status']['done'] and
                     _select['pipelined']['automated']['strehl']['core_arcsec'] > _config['core_min']*0.1 and
                     _select['pipelined']['automated']['strehl']['halo_arcsec'] < _config['halo_max']*1.5):
                check_pca(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                          _date, _obs, _pipe='automated')
            # make preview images
            check_preview(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                          _date, _obs, _pipe='automated')
            # once(/if) Strehl is ready, it'll rerun preview generation to show SR on the image
            _logger.info('Ran checks for {:s}'.format(_obs))

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
        traceback.print_exc()
        _logger.error('{:s}, automated pipeline: {:s}'.format(_obs, _e))
        # make sure all force_redo switches are off:
        for _key in ('pipelined.automated.pca.status.force_redo',
                     'pipelined.automated.pca.preview.force_redo',
                     'pipelined.automated.preview.force_redo',
                     'pipelined.automated.strehl.status.force_redo'):
            set_key(_logger, _coll, _obs, _key, False)
        return False

    # make sure all force_redo switches are off:
    for _key in ('pipelined.automated.pca.status.force_redo',
                 'pipelined.automated.pca.preview.force_redo',
                 'pipelined.automated.preview.force_redo',
                 'pipelined.automated.strehl.status.force_redo'):
        _status = set_key(_logger, _coll, _obs, _key, False)

    return _status


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
    try:
        # ran through lucky pipeline? computed Strehl? is it bad? all positive - then proceed
        if _select['pipelined']['faint']['status']['force_redo'] or \
                (_select['pipelined']['automated']['status']['done'] and
                 _select['pipelined']['automated']['strehl']['status']['done'] and
                     (_select['pipelined']['automated']['strehl']['core_arcsec'] < _config['core_min'] or
                      _select['pipelined']['automated']['strehl']['halo_arcsec'] > _config['halo_max']))\
                or (_select['pipelined']['faint']['status']['retries'] > 0):
            _logger.debug('{:s} suitable for faint pipeline'.format(_obs))

            # following structure.md:
            path_faint = os.path.join(_config['path_archive'], _date, _obs, 'faint')

            # path exists? (if yes - it must have been created by job_faint_pipeline)
            if os.path.exists(path_faint):
                # check folder modified date:
                time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_faint).st_mtime)
                # new/changed? (re)load data from disk + update database entry + (re)make preview
                if _select['pipelined']['faint']['last_modified'] != time_tag:
                    # load data from disk
                    # load shifts.txt:
                    f_shifts = [f for f in os.listdir(path_faint) if f == 'shifts.txt'][0]
                    # lock position + shifts (frame_number x y ex ey)
                    y, x, shifts = load_faint_shifts(os.path.join(path_faint, f_shifts))
                    # get fits header:
                    f_fits = os.path.join(path_faint, '{:s}_summed.fits'.format(_obs))
                    header = get_fits_header(f_fits)

                    # update database entry
                    _coll.update_one(
                        {'_id': _obs},
                        {
                            '$set': {
                                'pipelined.faint.status.done': True,
                                'pipelined.faint.last_modified': time_tag,
                                'pipelined.faint.fits_header': header,
                                'pipelined.automated.location': ['{:s}:{:s}'.format(
                                    _config['analysis_machine_external_host'],
                                    _config['analysis_machine_external_port']),
                                    _config['path_archive']],
                                'pipelined.faint.lock_position': [x, y],
                                'pipelined.faint.shifts': shifts,
                                'pipelined.faint.preview.done': False,
                                'pipelined.faint.strehl.status.done': False,
                                'pipelined.faint.pca.status.done': False
                            }
                        }
                    )
                    _logger.info('Updated faint pipeline entry for {:s}'.format(_obs))
                # check on Strehl:
                check_strehl(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                             _date, _obs, _pipe='faint')
                # check on PCA
                # reload entry from db:
                _select = _coll.find_one({'_id': _obs})
                # Strehl done? then proceed:
                if _select['pipelined']['faint']['pca']['status']['force_redo'] or \
                        _select['pipelined']['faint']['strehl']['status']['done']:
                    check_pca(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                              _date, _obs, _pipe='faint')
                # make preview images
                check_preview(_config, _logger, _coll, _coll.find_one({'_id': _obs}),
                              _date, _obs, _pipe='faint')
                # once(/if) Strehl is ready, it'll rerun preview generation to show SR on the image

            # path does not exist? make sure it's not marked 'done'
            elif _select['pipelined']['faint']['status']['done']:
                # update database entry if incorrectly marked 'done'
                # (could not find the respective directory)
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.faint.status.done': False,
                            'pipelined.faint.last_modified': utc_now(),
                            'pipelined.automated.location': [],
                            'pipelined.faint.lock_position': None,
                            'pipelined.faint.shifts': None,
                            'pipelined.faint.preview.done': False,
                            'pipelined.faint.strehl.status.done': False,
                            'pipelined.faint.pca.status.done': False
                        }
                    }
                )
                _logger.debug(
                    '{:s} not (yet) processed (at least I could not find it), marking undone'.
                        format(_obs))
                # reload entry from db:
                _select = _coll.find_one({'_id': _obs})
                _logger.info('Corrected faint pipeline entry for {:s}'.format(_obs))
                # a job will be placed into the queue at the next invocation of check_strehl

            # if 'done' is changed to False (ex- or internally), the if clause is triggered,
            # which in turn triggers a job placement into the queue to rerun PCA.
            # when invoked for the next time, the previous portion of the code will make sure
            # to update the database entry (since the last_modified value
            # will be different from the new folder modification date)

            if _select['pipelined']['faint']['status']['force_redo'] or \
                    (not _select['pipelined']['faint']['status']['done'] and
                     _select['pipelined']['faint']['status']['retries'] <
                     _config['max_pipelining_retries']):
                # prepare stuff for job execution
                _raws_zipped = sorted(_select['raw_data']['data'])
                # put a job into the queue
                if not inqueue('job_faint_pipeline', _obs, path_faint):
                    job_faint_pipeline(_config=_config, _raws_zipped=_raws_zipped, _date=_date,
                                       _obs=_obs, _path_out=path_faint)
                # update database entry
                _coll.update_one(
                    {'_id': _obs},
                    {
                        '$set': {
                            'pipelined.faint.last_modified': utc_now()
                        },
                        '$inc': {
                            'pipelined.faint.status.retries': 1
                        }
                    }
                )
                _logger.info('put a faint pipeline job into the queue for {:s}'.format(_obs))

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        _logger.error('{:s}, faint pipeline: {:s}'.format(_obs, _e))
        # make sure all force_redo switches are off:
        for _key in ('pipelined.faint.status.force_redo',
                     'pipelined.faint.pca.status.force_redo',
                     'pipelined.faint.pca.preview.force_redo',
                     'pipelined.faint.preview.force_redo',
                     'pipelined.faint.strehl.status.force_redo'):
            set_key(_logger, _coll, _obs, _key, False)
        return False

        # make sure all force_redo switches are off:
    for _key in ('pipelined.faint.status.force_redo',
                 'pipelined.faint.pca.status.force_redo',
                 'pipelined.faint.pca.preview.force_redo',
                 'pipelined.faint.preview.force_redo',
                 'pipelined.faint.strehl.status.force_redo'):
        _status = set_key(_logger, _coll, _obs, _key, False)

    return _status


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

        if _select['pipelined'][_pipe]['strehl']['status']['force_redo'] or \
                (not _select['pipelined'][_pipe]['strehl']['status']['done'] and
                 _select['pipelined'][_pipe]['strehl']['status']['retries'] <
                 _config['max_pipelining_retries']):
            if _pipe == 'automated':
                # check if actually processed through pipeline
                path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                                 tag in ('high_flux', 'faint', 'zero_flux') if
                                 os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
                _fits_name = '100p.fits'
                _drizzled = True
                _method = 'pipeline_settings.txt'
            elif _pipe == 'faint':
                # check if actually processed through pipeline
                _path_obs = os.path.join(_config['path_archive'], _date, _obs, 'faint')
                path_obs_list = [_path_obs] if os.path.exists(_path_obs) else []
                _fits_name = '{:s}_summed.fits'.format(_obs)
                _drizzled = False
                _method = 'shifts.txt'
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
                if not inqueue('job_strehl', _obs, path_out):
                    job_strehl(_path_in=path_obs, _fits_name=_fits_name,
                               _obs=_obs, _path_out=path_out,
                               _plate_scale=plate_scale, _Strehl_factor=Strehl_factor,
                               _method=_method, _drizzled=_drizzled,
                               _core_min=_config['core_min'], _halo_max=_config['halo_max'])
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
        _logger.error('{:s}, Strehl for {:s} pipeline: {:s}'.format(_obs, _pipe, _e))
        return False

    return True


def check_pca(_config, _logger, _coll, _select, _date, _obs, _pipe):
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

        # path exists? (if yes - it must have been created by job_pca)
        if os.path.exists(path_pca):
            # check folder modified date:
            time_tag = datetime.datetime.utcfromtimestamp(os.stat(path_pca).st_mtime)
            # new/changed? (re)load data from disk + update database entry + (re)make preview
            if _select['pipelined'][_pipe]['pca']['last_modified'] != time_tag:
                # load data from disk
                # load contrast curve
                f_cc = '{:s}_contrast_curve.txt'.format(_obs)
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

        if _select['pipelined'][_pipe]['pca']['status']['force_redo'] or \
                (not _select['pipelined'][_pipe]['pca']['status']['done'] and
                 _select['pipelined'][_pipe]['pca']['status']['retries'] <
                 _config['max_pipelining_retries']):
            if _pipe == 'automated':
                # check if actually processed through pipeline
                path_obs_list = [os.path.join(_config['path_pipe'], _date, tag, _obs) for
                                 tag in ('high_flux', 'faint', 'zero_flux') if
                                 os.path.exists(os.path.join(_config['path_pipe'], _date, tag, _obs))]
                _fits_name = '100p.fits'
                _drizzled = True
                _method = 'pipeline_settings.txt'
            elif _pipe == 'faint':
                # check if actually processed through pipeline
                _path_obs = os.path.join(_config['path_archive'], _date, _obs, 'faint')
                path_obs_list = [_path_obs] if os.path.exists(_path_obs) else []
                _fits_name = '{:s}_summed.fits'.format(_obs)
                _drizzled = False
                _method = 'shifts.txt'
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
                if not inqueue('job_pca', _obs, path_out):
                    job_pca(_config=_config, _path_in=path_obs, _fits_name=_fits_name, _obs=_obs,
                            _path_out=path_out, _plate_scale=plate_scale,
                            _method=_method, _x=None, _y=None, _drizzled=_drizzled)
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
        _logger.error('{:s}, PCA for {:s} pipeline: {:s}'.format(_obs, _pipe, _e))
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
        if _select['pipelined'][_pipe]['preview']['force_redo'] or \
                (not _select['pipelined'][_pipe]['preview']['done'] and
                _select['pipelined'][_pipe]['status']['done'] and
                _select['pipelined'][_pipe]['preview']['retries']
                    < _config['max_pipelining_retries'])\
                or (_select['pipelined'][_pipe]['status']['done'] and
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
                _drizzled = True
            elif _pipe == 'faint':
                _path_obs = os.path.join(_config['path_archive'], _date, _obs, 'faint')
                path_obs_list = [_path_obs] if os.path.exists(_path_obs) else []
                _drizzled = False
            else:
                raise NotImplemented()
                # path_obs_list = []

            # processed?
            if len(path_obs_list) == 1:
                # this also considers the pathological case when an obs ended up in several classes
                path_obs = path_obs_list[0]

                # what's in a fits name?
                if _pipe == 'automated':
                    if 'failed' not in path_obs:
                        f_fits = os.path.join(path_obs, '100p.fits')
                        _method = 'pipeline_settings.txt'
                    else:
                        f_fits = os.path.join(path_obs, 'sum.fits')
                        _method = 'sextractor'
                elif _pipe == 'faint':
                    f_fits = os.path.join(path_obs, '{:s}_summed.fits'.format(_obs))
                    _method = 'shifts.txt'
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
                                                                 _win=None, _method=_method,
                                                                 _x=None, _y=None, _drizzled=_drizzled)
                    else:
                        # don't crop planets
                        preview_img_cropped = preview_img
                        _x, _y = None, None
                    # Strehl ratio (if available, otherwise will be None)
                    SR = _select['pipelined'][_pipe]['strehl']['ratio_percent']

                    fits_header = get_fits_header(f_fits)
                    try:
                        # _pix_x = int(re.search(r'(:)(\d+)',
                        #                        _select['pipelined'][_pipe]['fits_header']['DETSIZE'][0]).group(2))
                        _pix_x = int(re.search(r'(:)(\d+)', fits_header['DETSIZE'][0]).group(2))
                    except KeyError:
                        # this should be there, even if it's sum.fits
                        _pix_x = int(fits_header['NAXIS1'][0])

                    _status = generate_pipe_preview(path_out, _obs, preview_img, preview_img_cropped,
                                                    SR, _fow_x=36, _pix_x=_pix_x, _drizzled=_drizzled,
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
        _logger.error('{:s}, preview for {:s} pipeline: {:s}'.format(_obs, _pipe, _e))
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
        if _select['pipelined'][_pipe]['pca']['preview']['force_redo'] or \
                (not _select['pipelined'][_pipe]['pca']['preview']['done'] and
                 _select['pipelined'][_pipe]['pca']['status']['done'] and
                 _select['pipelined'][_pipe]['pca']['preview']['retries']
                         < _config['max_pipelining_retries']):
            # following structure.md:
            path_pca = os.path.join(_config['path_archive'], _date, _obs, _pipe, 'pca')

            # processed?
            if os.path.exists(path_pca):

                # what's in a fits name?
                f_fits = os.path.join(path_pca, '{:s}_pca.fits'.format(_obs))

                # load first image frame from the fits file
                preview_img = load_fits(f_fits)
                # scale with local contrast optimization for preview:
                preview_img = scale_image(preview_img, correction='local')

                # contrast_curve:
                _cc = _select['pipelined'][_pipe]['pca']['contrast_curve']

                # number of pixels in X on the detector
                _pix_x = int(re.search(r'(:)(\d+)',
                                   _select['pipelined'][_pipe]['fits_header']['DETSIZE'][0]).group(2))

                if _pipe == 'automated':
                    _drizzled = True
                elif _pipe == 'faint':
                    _drizzled = False
                else:
                    raise NotImplementedError

                _status = generate_pca_images(_path_out=path_pca,
                                              _obs=_obs, _preview_img=preview_img,
                                              _cc=_cc, _fow_x=36, _pix_x=_pix_x, _drizzled=_drizzled)

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
        _logger.error('{:s}, PCA preview for {:s} pipeline failed: {:s}'.format(_obs, _pipe, _e))
        return False

    return True


def check_raws(_config, _logger, _coll, _select, _date, _obs, _date_files):
    """
        Check if raw data were updated/changed, reflect that in the database
    :param _config:
    :param _logger:
    :param _coll:
    :param _select:
    :param _date:
    :param _obs:
    :param _date_files:
    :return:
    """
    try:
        # raw file names
        _raws = [_s for _s in _date_files if re.match(re.escape(_obs), _s) is not None]
        # time tags. use the 'freshest' time tag for 'last_modified'
        time_tags = [datetime.datetime.utcfromtimestamp(
                        os.stat(os.path.join(_config['path_raw'], _date, _s)).st_mtime)
                     for _s in _raws]
        time_tag = max(time_tags)

        # print(_obs, _select['raw_data']['last_modified'], time_tag)

        # changed? update database entry then:
        if _select['raw_data']['last_modified'] != time_tag:
            _coll.update_one(
                {'_id': _obs},
                {
                    '$set': {
                        'raw_data.data': sorted(_raws),
                        'raw_data.last_modified': time_tag
                    }
                }
            )
            _logger.info('Corrected raw_data entry for {:s}'.format(_obs))

    except Exception as _e:
        traceback.print_exc()
        _logger.error('{:s}, checking raw files failed: {:s}'.format(_obs, _e))
        return False

    return True


def check_distributed(_config, _logger, _coll, _select, _date, _obs, _n_days=1.0):
    """
        Check if observation has been processed and can be distributed.
        Generate a compressed file with everything.

    :param _config: config data
    :param _logger: logger instance
    :param _coll: collection in the database
    :param _select: database entry
    :param _date: date of obs
    :param _obs: obs name
    :param _n_days: pipelining done n days ago

    :return:

    """
    try:
        if not _select['distributed']['status']:
            # to keep things simple, just check if the pipelines are marked 'done'
            # more than _n_days ago. this should give enough time for all other tasks
            # to finish/fail
            # if lucky Strehl is OK, faint pipeline would never trigger
            done = _select['pipelined']['automated']['status']['done'] and \
                    (_select['pipelined']['faint']['status']['done'] or
                     _select['pipelined']['faint']['status']['retries']
                      > _config['max_pipelining_retries'] or
                     _select['pipelined']['faint']['status']['retries']
                      == _config['max_pipelining_retries'] or
                     _select['pipelined']['automated']['strehl']['flag'] == 'OK')

            # make sure nothing's enqueued for the _obs
            enqueued = inqueue('job_faint_pipeline', _obs) and inqueue('job_strehl', _obs) \
                        and inqueue('job_pca', _obs)

            if done and not enqueued:
                _path_obs = os.path.join(_config['path_archive'], _date, _obs)
                last_modified = datetime.datetime.utcfromtimestamp(
                                    os.stat(os.path.join(_path_obs)).st_mtime)
                last_modified = last_modified.replace(tzinfo=pytz.utc)

                if (utc_now() - last_modified).total_seconds()/86400.0 > _n_days:
                    # prepare everything for distribution
                    _path_pipe = os.path.join(_config['path_pipe'], _date,
                                              str(_select['pipelined']['automated']['classified_as']))
                    _path_archive = os.path.join(_config['path_archive'], _date)
                    _status_ok = prepare_for_distribution(_path_pipe, _path_archive, _obs)
                    # update database entry:
                    if _status_ok:
                        _coll.update_one(
                            {'_id': _obs},
                            {
                                '$set': {
                                    'distributed.status': True,
                                    'distributed.location': ['{:s}:{:s}'.format(
                                            _config['analysis_machine_external_host'],
                                            _config['analysis_machine_external_port']),
                                            _config['path_archive']],
                                    'distributed.last_modified': utc_now()
                                }
                            }
                        )
                        _logger.info('{:s} ready for distribution'.format(_obs))
                    else:
                        _coll.update_one(
                            {'_id': _obs},
                            {
                                '$set': {
                                    'distributed.status': False,
                                    'distributed.location': [],
                                    'distributed.last_modified': utc_now()
                                }
                            }
                        )
                        _logger.info('{:s} ready for distribution'.format(_obs))

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        _logger.error('{:s}, distributed or not check failed: {:s}'.format(_obs, _e))
        return False

    return True


def check_aux(_config, _logger, _coll, _coll_aux, _date, _seeing_frames, _n_days=1.5):
    """
        Check nightly auxiliary (summary) data
    :param _config:
    :param _logger:
    :param _coll:
    :param _coll_aux:
    :param _date:
    :param _seeing_frames:
    :param _n_days:
    :return:
    """
    try:
        _select = _coll_aux.find_one({'_id': _date})
        _path_out = os.path.join(_config['path_archive'], _date, 'summary')

        ''' check/do seeing '''
        last_modified = _select['seeing']['last_modified'].replace(tzinfo=pytz.utc)
        # path to store unzipped raw files
        _path_tmp = _config['path_tmp']
        # path to raw seeing data
        _path_seeing = os.path.join(_config['path_raw'], _date)
        # path to calibration data produced by lucky pipeline:
        _path_calib = os.path.join(_config['path_pipe'], _date, 'calib')

        # _seeing_frames = _select['seeing']['frames']
        _seeing_raws = ['{:s}.fits.bz2'.format(_s[0]) for _s in _seeing_frames]

        if len(_seeing_raws) > 0:
            try:
                time_tags = [datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_path_seeing, _s)).st_mtime)
                             for _s in _seeing_raws]
                time_tag = max(time_tags)
                time_tag = time_tag.replace(tzinfo=pytz.utc)
                # not done or new files appeared in the raw directory
                if (not _select['seeing']['done'] or last_modified != time_tag) and \
                        (_select['seeing']['retries'] <= _config['max_pipelining_retries']):

                        # unbzip source file(s):
                        lbunzip2(_path_in=_path_seeing, _files=_seeing_raws, _path_out=_path_tmp,
                                 _keep=True, _v=True)

                        # unzipped file names:
                        _obsz = [_s[0] for _s in _seeing_frames]
                        # print(raws)

                        seeing_plot = []
                        for ii, _obs in enumerate(_obsz):
                            print('processing {:s}'.format(_obs))
                            # this returns datetime, seeing in " and in pix, and used filter:
                            _date_utc, seeing, _, _filt, seeing_x, seeing_y = \
                                process_seeing(_path_in=_path_tmp, _seeing_frame=_obs,
                                               _path_calib=_path_calib, _path_out=_path_out,
                                               _frame_size_x_arcsec=36,
                                               _fit_model=_config['seeing']['fit_model'],
                                               _box_size=_config['seeing']['win'])
                            if seeing is not None:
                                seeing_plot.append([_date_utc, seeing, _filt])
                            _seeing_frames[ii][1] = _date_utc
                            _seeing_frames[ii][2] = _filt
                            _seeing_frames[ii][3] = seeing
                            _seeing_frames[ii][4] = seeing_x
                            _seeing_frames[ii][5] = seeing_y

                        # generate summary plot for the whole night:
                        if len(seeing_plot) > 0:
                            seeing_plot = np.array(seeing_plot)
                            # sort by time stamp:
                            seeing_plot = seeing_plot[seeing_plot[:, 0].argsort()]

                            # filter colors on the plot:
                            filter_colors = {'lp600': plt.cm.Blues(0.82),
                                             'Sg': plt.cm.Greens(0.7),
                                             'Sr': plt.cm.Reds(0.7),
                                             'Si': plt.cm.Oranges(0.7),
                                             'Sz': plt.cm.Oranges(0.5)}

                            fig = plt.figure('Seeing data for {:s}'.format(_date), figsize=(8, 3), dpi=200)
                            ax = fig.add_subplot(111)

                            # all filters used that night:
                            filters_used = set(seeing_plot[:, 2])

                            for filter_used in filters_used:
                                # plot different filters in different colors
                                mask = seeing_plot[:, 2] == filter_used
                                fc = filter_colors[filter_used] if filter_used in filter_colors else plt.cm.Greys(0.7)
                                ax.plot(seeing_plot[mask, 0], seeing_plot[mask, 1], '.',
                                        c=fc, markersize=8, label=filter_used)

                            ax.set_ylabel('Seeing, arcsec')  # , fontsize=18)
                            ax.grid(linewidth=0.5)

                            # evaluate estimators
                            try:
                                # make a robust fit to seeing data for visual reference
                                t_seeing_plot = np.array([(_t - seeing_plot[0, 0]).total_seconds()
                                                          for _t in seeing_plot[:, 0]])
                                t_seeing_plot = np.expand_dims(t_seeing_plot, axis=1)
                                estimators = [('RANSAC', linear_model.RANSACRegressor()), ]
                                for name, estimator in estimators:
                                    model = make_pipeline(PolynomialFeatures(degree=5), estimator)
                                    model.fit(t_seeing_plot, seeing_plot[:, 1])
                                    y_plot = model.predict(t_seeing_plot)
                                    # noinspection PyUnboundLocalVariable
                                    ax.plot(seeing_plot[:, 0], y_plot, '--', c=plt.cm.Blues(0.4),
                                            linewidth=1, label='Robust {:s} fit'.format(name), clip_on=True)
                            except Exception as _e:
                                print(_e)
                                traceback.print_exc()

                            myFmt = mdates.DateFormatter('%H:%M')
                            ax.xaxis.set_major_formatter(myFmt)
                            fig.autofmt_xdate()

                            # make sure our 'robust' fit didn't spoil the scale:
                            ax.set_ylim([np.min(seeing_plot[:, 1]) * 0.9, np.max(seeing_plot[:, 1]) * 1.1])
                            dt = datetime.timedelta(seconds=(seeing_plot[-1, 0] -
                                                             seeing_plot[0, 0]).total_seconds() * 0.05)
                            ax.set_xlim([seeing_plot[0, 0] - dt, seeing_plot[-1, 0] + dt])

                            # add legend:
                            ax.legend(loc='best', numpoints=1, fancybox=True, prop={'size': 6})

                            plt.tight_layout()

                            # plt.show()
                            f_seeing_plot = os.path.join(_path_out, 'seeing.{:s}.png'.format(_date))
                            fig.savefig(f_seeing_plot, dpi=300)

                        # update database record:
                        _coll_aux.update_one(
                            {'_id': _date},
                            {
                                '$set': {
                                    'seeing.done': True,
                                    'seeing.frames': _seeing_frames,
                                    'seeing.last_modified': time_tag
                                },
                                '$inc': {
                                    'seeing.retries': 1
                                }
                            }
                        )
                        _logger.error('Successfully generated seeing summary for {:s}'.format(_date))

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                try:
                    _coll_aux.update_one(
                        {'_id': _date},
                        {
                            '$set': {
                                'seeing.done': False,
                                'seeing.frames': [],
                                'seeing.last_modified': utc_now()
                            },
                            '$inc': {
                                'seeing.retries': 1
                            }
                        }
                    )
                    # clean up stuff
                    seeing_summary_plot = os.path.join(_path_out, 'seeing.{:s}.png'.format(_date))
                    if os.path.exists(seeing_summary_plot):
                        os.remove(seeing_summary_plot)
                    individual_frames_path = os.path.join(_path_out, 'seeing')
                    for individual_frame in os.listdir(individual_frames_path):
                        if os.path.exists(os.path.join(individual_frames_path, individual_frame)):
                            os.remove(os.path.join(individual_frames_path, individual_frame))
                finally:
                    _logger.error('Seeing summary generation failed for {:s}: {:s}'.format(_date, _e))

            finally:
                # remove unzipped files
                _seeing_raws_unzipped = [os.path.splitext(_f)[0] for _f in _seeing_raws]
                for _seeing_raw_unzipped in _seeing_raws_unzipped:
                    if os.path.exists(os.path.join(_path_tmp, _seeing_raw_unzipped)):
                        os.remove(os.path.join(_path_tmp, _seeing_raw_unzipped))

        ''' make summary Strehl plot '''
        last_modified = _select['strehl']['last_modified'].replace(tzinfo=pytz.utc)
        if (not _select['strehl']['done']
            or (utc_now() - last_modified).total_seconds() / 86400.0 > _n_days) and \
                (_select['strehl']['retries'] <= _config['max_pipelining_retries']):
            try:
                _logger.debug('Trying to generate summary Strehl plot for {:s}'.format(_date))
                print('Generating summary Strehl plot for {:s}'.format(_date))
                # query the database:
                day = datetime.datetime.strptime(_date, '%Y%m%d')

                _pipe = 'automated'
                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                SR_good_lucky = np.array([[_obs['date_utc'], _obs['pipelined'][_pipe]['strehl']['ratio_percent']]
                                          for _obs in cursor
                                          if _obs['pipelined'][_pipe]['strehl']['status']['done'] and
                                          _obs['pipelined'][_pipe]['strehl']['flag'] == 'OK'])
                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                SR_notgood_lucky = np.array([[_obs['date_utc'], _obs['pipelined'][_pipe]['strehl']['ratio_percent']]
                                             for _obs in cursor
                                             if _obs['pipelined'][_pipe]['strehl']['status']['done'] and
                                             _obs['pipelined'][_pipe]['strehl']['flag'] == 'BAD?'])

                _pipe = 'faint'
                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                SR_good_faint = np.array([[_obs['date_utc'], _obs['pipelined'][_pipe]['strehl']['ratio_percent']]
                                          for _obs in cursor
                                          if _obs['pipelined'][_pipe]['strehl']['status']['done'] and
                                          _obs['pipelined'][_pipe]['strehl']['flag'] == 'OK'])
                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                SR_notgood_faint = np.array([[_obs['date_utc'], _obs['pipelined'][_pipe]['strehl']['ratio_percent']]
                                             for _obs in cursor
                                             if _obs['pipelined'][_pipe]['strehl']['status']['done'] and
                                             _obs['pipelined'][_pipe]['strehl']['flag'] == 'BAD?'])

                if max(map(len, (SR_good_lucky, SR_notgood_lucky, SR_good_faint, SR_notgood_faint))) > 0:
                    _logger.info('Generating summary Strehl plot for {:s}'.format(_date))

                    fig = plt.figure('Strehls for {:s}'.format(_date), figsize=(7, 3.18), dpi=200)
                    ax = fig.add_subplot(111)

                    if len(SR_good_lucky) > 0:
                        # sort by time stamps:
                        SR_good = SR_good_lucky[SR_good_lucky[:, 0].argsort()]
                        good = True
                        SR_mean = np.mean(SR_good[:, 1])
                        SR_std = np.std(SR_good[:, 1])
                        SR_max = np.max(SR_good[:, 1])
                        SR_min = np.min(SR_good[:, 1])

                        ax.plot(SR_good[:, 0], SR_good[:, 1], 'o', color=plt.cm.Oranges(0.7), markersize=5)

                        ax.axhline(y=SR_mean, linestyle='-', color=plt.cm.Blues(0.7), linewidth=1,
                                   label='Lucky mean = ' + str(round(SR_mean, 2)) + '%')
                        ax.axhline(y=SR_mean + SR_std, linestyle='--', color=plt.cm.Blues(0.7), linewidth=1,
                                   label=r'Lucky $\sigma _{SR}$ = ' + str(round(SR_std, 2)) + '%')
                        ax.axhline(y=SR_mean - SR_std, linestyle='--', color=plt.cm.Blues(0.7), linewidth=1)

                    if len(SR_notgood_lucky) > 0:
                        SR_notgood = SR_notgood_lucky[SR_notgood_lucky[:, 0].argsort()]
                        ax.plot(SR_notgood[:, 0], SR_notgood[:, 1], 'o', color=plt.cm.Greys(0.35), markersize=5)

                    if len(SR_good_faint) > 0:
                        # sort by time stamps:
                        SR_good = SR_good_faint[SR_good_faint[:, 0].argsort()]
                        good = True
                        SR_mean = np.mean(SR_good[:, 1])
                        SR_std = np.std(SR_good[:, 1])
                        SR_max = np.max(SR_good[:, 1])
                        SR_min = np.min(SR_good[:, 1])

                        ax.plot(SR_good[:, 0], SR_good[:, 1], 'o', color=plt.cm.Oranges(0.45), markersize=5)

                        ax.axhline(y=SR_mean, linestyle='-', color=plt.cm.Blues(0.35), linewidth=1,
                                   label='Faint mean = ' + str(round(SR_mean, 2)) + '%')
                        ax.axhline(y=SR_mean + SR_std, linestyle='--', color=plt.cm.Blues(0.35), linewidth=1,
                                   label=r'Faint $\sigma _{SR}$ = ' + str(round(SR_std, 2)) + '%')
                        ax.axhline(y=SR_mean - SR_std, linestyle='--', color=plt.cm.Blues(0.35), linewidth=1)

                    if len(SR_notgood_faint) > 0:
                        SR_notgood = SR_notgood_faint[SR_notgood_faint[:, 0].argsort()]
                        ax.plot(SR_notgood[:, 0], SR_notgood[:, 1], 'o', color=plt.cm.Blues(0.25), markersize=5)

                    # ax.set_xlabel('Time, UTC')
                    # xstart = np.min([SR_notgood[0, 0], SR_good[0, 0]]) - datetime.timedelta(minutes=15)
                    # xstop = np.max([SR_notgood[-1, 0], SR_good[-1, 0]]) + datetime.timedelta(minutes=15)
                    # ax.set_xlim([xstart, xstop])
                    ax.set_ylabel('Strehl Ratio, %')
                    # ax.legend(bbox_to_anchor=(1.35, 1), ncol=1, numpoints=1, fancybox=True)
                    leg1 = ax.legend(loc=1, numpoints=1, fancybox=True, prop={'size': 6})
                    ax.grid(linewidth=0.5)
                    ax.margins(0.05, 0.2)

                    myFmt = mdates.DateFormatter('%H:%M')
                    ax.xaxis.set_major_formatter(myFmt)
                    fig.autofmt_xdate()

                    # custom legend:
                    lucky_bad = mlines.Line2D([], [], markerfacecolor=plt.cm.Greys(0.35), marker='o',
                                              markersize=3, linewidth=0, label='Bad lucky Strehls')
                    faint_bad = mlines.Line2D([], [], markerfacecolor=plt.cm.Blues(0.25), marker='o',
                                              markersize=3, linewidth=0, label='Bad faint Strehls')
                    lucky_ok = mlines.Line2D([], [], markerfacecolor=plt.cm.Oranges(0.7), marker='o',
                                             markersize=3, linewidth=0, label='OK lucky Strehls')
                    faint_ok = mlines.Line2D([], [], markerfacecolor=plt.cm.Oranges(0.45), marker='o',
                                             markersize=3, linewidth=0, label='OK faint Strehls')
                    plt.legend(loc=2, handles=[lucky_bad, faint_bad, lucky_ok, faint_ok], prop={'size': 6})

                    # add the first legend back to the plot
                    ax.add_artist(leg1)

                    plt.tight_layout()

                    # dump results to disk
                    if not (os.path.exists(_path_out)):
                        os.makedirs(_path_out)

                    fig.savefig(os.path.join(_path_out, 'strehl.{:s}.png'.format(_date)), dpi=200)

                    _coll_aux.update_one(
                        {'_id': _date},
                        {
                            '$set': {
                                'strehl.done': True,
                                'strehl.last_modified': utc_now()
                            },
                            '$inc': {
                                'strehl.retries': 1
                            }
                        }
                    )
                    _logger.info('Generated summary Strehl plot for {:s}'.format(_date))

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                try:
                    _coll_aux.update_one(
                        {'_id': _date},
                        {
                            '$set': {
                                'strehl.done': False,
                                'strehl.last_modified': utc_now()
                            },
                            '$inc': {
                                'strehl.retries': 1
                            }
                        }
                    )
                    # clean up stuff
                    strehl_summary_plot = os.path.join(_path_out, 'strehl.{:s}.png'.format(_date))
                    if os.path.exists(strehl_summary_plot):
                        os.remove(strehl_summary_plot)
                finally:
                    _logger.error('Summary Strehl plot generation failed for {:s}: {:s}'.format(_date, _e))

        ''' make summary contrast curve plot '''
        last_modified = _select['contrast_curve']['last_modified'].replace(tzinfo=pytz.utc)
        if (not _select['contrast_curve']['done']
            or (utc_now() - last_modified).total_seconds() / 86400.0 > _n_days) and \
                (_select['contrast_curve']['retries'] <= _config['max_pipelining_retries']):
            try:
                _logger.debug('Trying to generate contrast curve summary for {:s}'.format(_date))
                print('Generating contrast curve summary for {:s}'.format(_date))
                # query the database:
                day = datetime.datetime.strptime(_date, '%Y%m%d')

                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                _pipe = 'automated'
                contrast_curves = np.array([np.array(_obs['pipelined'][_pipe]['pca']['contrast_curve'])
                                            for _obs in cursor
                                            if _obs['pipelined'][_pipe]['pca']['status']['done']
                                            and _obs['science_program']['program_id'] !=
                                                _config['planets_prog_num']])

                cursor = _coll.find({'date_utc': {'$gte': day, '$lt': day + datetime.timedelta(days=1)}})
                _pipe = 'faint'
                contrast_curves_faint = np.array([np.array(_obs['pipelined'][_pipe]['pca']['contrast_curve'])
                                                  for _obs in cursor
                                                  if _obs['pipelined'][_pipe]['pca']['status']['done']
                                                  and _obs['science_program']['program_id'] !=
                                                      _config['planets_prog_num']])

                if len(contrast_curves) > 0 or len(contrast_curves_faint) > 0:
                    _logger.info('Generating contrast curve summary for {:s}'.format(_date))
                    fig = plt.figure('Contrast curve', figsize=(8, 3.5), dpi=200)
                    ax = fig.add_subplot(111)

                    # add to plot:
                    sep_mean = np.linspace(0.2, 1.45, num=100)

                    # lucky-pipelined
                    cc_mean = []
                    for contrast_curve in contrast_curves:
                        ax.plot(contrast_curve[:, 0], contrast_curve[:, 1], '-', c=plt.cm.Greys(0.27), linewidth=1.1)
                        cc_tmp = np.interp(sep_mean, contrast_curve[:, 0], contrast_curve[:, 1])
                        if not np.isnan(cc_tmp).any():
                            cc_mean.append(cc_tmp)
                    if len(cc_mean) > 0:
                        # add median to plot:
                        ax.plot(sep_mean, np.median(np.array(cc_mean).T, axis=1), '-',
                                c=plt.cm.Oranges(0.7), linewidth=2.1)

                    # faint-pipelined:
                    cc_mean = []
                    for contrast_curve in contrast_curves_faint:
                        ax.plot(contrast_curve[:, 0], contrast_curve[:, 1], '--', c=plt.cm.Blues(0.27), linewidth=1.1)
                        cc_tmp = np.interp(sep_mean, contrast_curve[:, 0], contrast_curve[:, 1])
                        if not np.isnan(cc_tmp).any():
                            cc_mean.append(cc_tmp)

                    if len(cc_mean) > 0:
                        # add median to plot:
                        ax.plot(sep_mean, np.median(np.array(cc_mean).T, axis=1), '--',
                                c=plt.cm.Oranges(0.5), linewidth=2.1)
                    # # add median to plot:
                    # ax.plot(sep_mean, np.median(np.array(cc_mean).T, axis=1), '-',
                    #         c=plt.cm.Oranges(0.7), linewidth=2.5)

                    # beautify and save:
                    ax.set_xlim([0.2, 1.45])
                    ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
                    ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
                    ax.set_ylim([0, 8])
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.grid(linewidth=0.5)

                    # custom legend:
                    lucky_line = mlines.Line2D([], [], color=plt.cm.Greys(0.27), linestyle='-',
                                               label='Individual lucky contrast curves')
                    faint_line = mlines.Line2D([], [], color=plt.cm.Blues(0.27), linestyle='--',
                                               label='Individual faint contrast curves')
                    lucky_line_median = mlines.Line2D([], [], color=plt.cm.Oranges(0.7), linestyle='-',
                                                      label='Median lucky contrast curve')
                    faint_line_median = mlines.Line2D([], [], color=plt.cm.Oranges(0.5), linestyle='--',
                                                      markersize=15, label='Median faint contrast curve')
                    plt.legend(loc='best', handles=[lucky_line, faint_line, lucky_line_median, faint_line_median],
                               prop={'size': 6})

                    plt.tight_layout()

                    # dump results to disk
                    if not (os.path.exists(_path_out)):
                        os.makedirs(_path_out)

                    fig.savefig(os.path.join(_path_out, 'contrast_curve.{:s}.png'.format(_date)), dpi=200)

                    _coll_aux.update_one(
                        {'_id': _date},
                        {
                            '$set': {
                                'contrast_curve.done': True,
                                'contrast_curve.last_modified': utc_now()
                            },
                            '$inc': {
                                'contrast_curve.retries': 1
                            }
                        }
                    )
                    _logger.info('Generated summary contrast curve for {:s}'.format(_date))

            except Exception as _e:
                print(_e)
                traceback.print_exc()
                try:
                    _coll_aux.update_one(
                        {'_id': _date},
                        {
                            '$set': {
                                'contrast_curve.done': False,
                                'contrast_curve.last_modified': utc_now()
                            },
                            '$inc': {
                                'contrast_curve.retries': 1
                            }
                        }
                    )
                    # clean up stuff
                    cc_summary_plot = os.path.join(_path_out, 'contrast_curve.{:s}.png'.format(_date))
                    if os.path.exists(cc_summary_plot):
                        os.remove(cc_summary_plot)
                finally:
                    _logger.error('Summary contrast curve generation failed for {:s}: {:s}'.format(_date, _e))

        # TODO: check/insert weather data into the weather_kp? database
        pass

    except Exception as _e:
        print(_e)
        traceback.print_exc()
        _logger.error('Summary check failed for {:s}: {:s}'.format(_date, _e))
        return False

    return True


def prepare_for_distribution(_path_pipe, _path_archive, _obs):
    """
        Create an archive with all the processed data
    :param _path_pipe:
    :param _path_archive:
    :param _obs:
    :return:
    """
    try:
        # try to copy automated pipeline output. it's ok if this fails
        try:
            copy_tree(os.path.join(_path_pipe, _obs), os.path.join(_path_archive, _obs, 'automated'))
        finally:
            pass

        # create a tarball
        pipelines = [_path for _path in os.listdir(os.path.join(_path_archive, _obs))
                     if os.path.isdir(os.path.join(_path_archive, _obs, _path)) and _path[0] != '.']
        _p = subprocess.Popen(['tar', '-cf', '{:s}'.format(os.path.join(_path_archive, _obs, '{:s}.tar'.format(_obs))),
                               '-C', os.path.join(_path_archive, _obs)] + pipelines)
        # wait for it to finish
        _p.wait()

        # compress it:
        _p = subprocess.Popen(['lbzip2', '-f', '{:s}'.format(os.path.join(_path_archive,
                                                                          _obs, '{:s}.tar'.format(_obs))), '--best'])
        # wait for it to finish
        _p.wait()

        # remove the copy of automatically-pipelined files
        for _path in glob.glob(os.path.join(_path_archive, _obs, 'automated', '*.*')):
            if os.path.isfile(_path):
                os.remove(_path)
    except Exception as _e:
        print(_e)
        traceback.print_exc()
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


def init_db_entry(_config, _path_obs, _date_files, _obs,
                  _sou_name, _prog_num, _prog_pi, _date_utc, _camera, _filt):
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
    _raws = [_s for _s in _date_files if re.match(re.escape(_obs), _s) is not None]
    _entry['raw_data']['location'].append(['{:s}:{:s}'.format(
        _config['analysis_machine_external_host'],
        _config['analysis_machine_external_port']),
        _config['path_raw']])
    _entry['raw_data']['data'] = sorted(_raws)
    # use the 'freshest' timetag for 'last_modified'
    time_tags = [datetime.datetime.utcfromtimestamp(os.stat(os.path.join(_path_obs, _s)).st_mtime)
                 for _s in _raws]
    # print(time_tags)
    _entry['raw_data']['last_modified'] = max(time_tags)

    # get fits header:
    # header = get_fits_header(os.path.join(_path_obs, _raws[0]))
    # _entry['exposure'] = float(header['EXPOSURE'][0])
    # _entry['magnitude'] = float(header['MAGNITUD'][0])

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
    logger, logger_utc_date = set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG, _mode='a')

    while True:
        # check if a new log file needs to be started:
        if datetime.datetime.utcnow().strftime('%Y%m%d') != logger_utc_date:
            logger, logger_utc_date = set_up_logging(_path='logs', _name='archive', _level=logging.DEBUG, _mode='a')

        # if you start me up... if you start me up I'll never stop (hopefully not)
        logger.info('Started archiving cycle.')

        try:
            ''' script absolute location '''
            abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

            ''' load config data '''
            try:
                config = get_config(_abs_path=abs_path, _config_file=args.config_file)
                logger.debug('Successfully read in the config file {:s}'.format(args.config_file))
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                logger.error('Failed to read in the config file {:s}'.format(args.config_file))
                sys.exit()

            ''' check connection to redis server that processes pipeline tasks '''
            try:
                pubsub = huey.storage.listener()
                logger.debug('Successfully connected to the redis server at 127.0.0.1:6379')
            except ConnectionError as e:
                traceback.print_exc()
                logger.error(e)
                logger.error('Redis server not responding')
                sys.exit()

            ''' Connect to the mongodb database '''
            try:
                client, db, coll, coll_aux, coll_weather, program_pi = connect_to_db(_config=config, _logger=logger)
                if None in (db, coll, program_pi):
                    raise Exception('Failed to connect to the database')
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                sys.exit()

            '''
             ###############################
             CHECK IF DATABASE IS UP TO DATE
             ###############################
            '''

            ''' check all raw data starting from config['archiving_start_date'] '''
            # get all dates with some raw data
            dates = sorted([p for p in os.listdir(config['path_raw'])
                            if os.path.isdir(os.path.join(config['path_raw'], p))
                            and datetime.datetime.strptime(p, '%Y%m%d') >= config['archiving_start_date']])
            print(dates)
            # for each date get all unique obs names (used as _id 's in the db)
            for date in dates[::-1]:
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
                # handle seeing files separately [lower priority]
                date_seeing = [re.split(pattern_fits, s)[0] for s in date_files
                               if re.search(pattern_end, s) is not None and
                               re.match('seeing_', s) is not None]
                # print(date_seeing)
                # for each source name see if there's an entry in the database
                for obs in date_obs:
                    print('processing {:s}'.format(obs))
                    logger.info('processing {:s}'.format(obs))

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
                        entry = init_db_entry(_config=config,
                                              _path_obs=os.path.join(config['path_raw'], date),
                                              _date_files=date_files,
                                              _obs=obs, _sou_name=sou_name,
                                              _prog_num=prog_num, _prog_pi=prog_pi,
                                              _date_utc=date_utc, _camera=camera,
                                              _filt=filt)
                        # insert it into database
                        result = coll.insert_one(entry)
                        # and select it:
                        select = coll.find_one({'_id': obs})

                    # entry found in database, check if pipelined, update entry if necessary
                    else:
                        print('{:s} in database, checking...'.format(obs))

                    ''' check raw data '''
                    status_ok = check_raws(_config=config, _logger=logger, _coll=coll,
                                           _select=select, _date=date, _obs=obs, _date_files=date_files)
                    if not status_ok:
                        logger.error('Checking failed for raw data: {:s}'.format(obs))

                    ''' check lucky-pipelined data '''
                    # Strehl and PCA are checked from within check_pipe_automated
                    status_ok = check_pipe_automated(_config=config, _logger=logger, _coll=coll,
                                                     _select=coll.find_one({'_id': obs}), _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for lucky pipeline: {:s}'.format(obs))

                    ''' check faint-pipelined data '''
                    status_ok = check_pipe_faint(_config=config, _logger=logger, _coll=coll,
                                                 _select=coll.find_one({'_id': obs}), _date=date, _obs=obs)
                    if not status_ok:
                        logger.error('Checking failed for faint pipeline: {:s}'.format(obs))

                    # TODO: if it is a planetary observation, run the planetary pipeline
                    ''' check planetary-pipelined data '''

                    # mark distributed when all pipelines done or n_retries>3
                    # compress everything with bzip2, store and TODO: transfer over to Caltech
                    status_ok = check_distributed(_config=config, _logger=logger, _coll=coll,
                                                  _select=coll.find_one({'_id': obs}),
                                                  _date=date, _obs=obs, _n_days=2.1)
                    if not status_ok:
                        logger.error('Checking failed if distributed: {:s}'.format(obs))

                ''' auxiliary (summary) data '''
                # look up entry in the database:
                select = coll_aux.find_one({'_id': date})
                # if entry not in database, create empty one and populate it
                if select is None:
                    # insert date into aux database:
                    result = coll_aux.insert_one({'_id': date,
                                                  'seeing': {'done': False,
                                                             'frames': [[k, None, None, None] for k in date_seeing],
                                                             'retries': 0,
                                                             'last_modified': utc_now()},
                                                  'contrast_curve': {'done': False,
                                                                     'retries': 0,
                                                                     'last_modified': utc_now()},
                                                  'strehl': {'done': False,
                                                             'retries': 0,
                                                             'last_modified': utc_now()}})
                # query database for all contrast curves and Strehls +seeing
                # make joint plots to display on the website
                # do that once a day*1.5 or so
                status_ok = check_aux(_config=config, _logger=logger, _coll=coll, _coll_aux=coll_aux, _date=date,
                                      _seeing_frames=[[k, None, None, None, None, None] for k in date_seeing],
                                      _n_days=1.5)
                if not status_ok:
                    logger.error('Checking summaries failed for {:s}'.format(date))

                # print(huey.get_storage().enqueued_items())
                # print(huey.pending())

            logger.info('Finished archiving cycle.')
            sleep_for = naptime(config)  # seconds
            if sleep_for:
                time.sleep(sleep_for)
            else:
                logger.error('Could not fall asleep, exiting.')
                break

        except KeyboardInterrupt:
            logger.error('User exited the archiver.')
            logger.info('Finished archiving cycle.')
            # try disconnecting from the database (if connected)
            try:
                if 'client' in locals():
                    client.close()
            finally:
                break

        except Exception as e:
            print(e)
            traceback.print_exc()
            logger.error(e)
            logger.error('Unknown error, exiting. Please check the logs')
            # TODO: send out an email with an alert
            logger.info('Finished archiving cycle.')
            # try disconnecting from the database (if connected)
            try:
                if 'client' in locals():
                    client.close()
            finally:
                break
