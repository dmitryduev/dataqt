"""
    PCA pipeline for Robo-AO
    RMJC @ Caltech
"""
from __future__ import print_function
import os
import datetime
import numpy as np
from astropy.io import fits
import vip
import photutils
from scipy import stats
import operator
import sewpy
import argparse
import ConfigParser
import inspect

from scipy.optimize import fmin
from math import sqrt, pow, exp
from skimage import exposure
from copy import deepcopy
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


# detect observatiosn which are bad because of being too faint
# 1. make a radial profile, snip out the central 3 pixels
#    (removing the ones which are affected by photon noise)
# 2. measure the width of the remaining flux
# 3. check for too small a width (=> no flux) or too large a width (=> crappy performance)
def gaussian(p, x):
    return p[0] + p[1] * (exp(-x * x / (2.0 * p[2] * p[2])))


def moffat(p, x):
    base = 0.0
    scale = p[1]
    fwhm = p[2]
    beta = p[3]

    if pow(2.0, (1.0 / beta)) > 1.0:
        alpha = fwhm / (2.0 * sqrt(pow(2.0, (1.0 / beta)) - 1.0))
        return base + scale * pow(1.0 + ((x / alpha) ** 2), -beta)
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
            r = sqrt((x - p.shape[1] / 2) ** 2 + (y - p.shape[0] / 2) ** 2)
            if r > 3:  # remove core
                pix_rad.append(r)
                pix_vals.append(p[y][x])
            else:
                core_pix_rad.append(r)
                core_pix_vals.append(p[y][x])

    try:
        if return_halo:
            p0 = [0.0, np.max(pix_vals), 20.0, 2.0]
            p = fmin(residuals, p0, args=(pix_rad, pix_vals), maxiter=1000000, maxfun=1000000, ftol=1e-3,
                     xtol=1e-3, disp=False)

        p0 = [0.0, np.max(core_pix_vals), 5.0, 2.0]
        core_p = fmin(residuals, p0, args=(core_pix_rad, core_pix_vals), maxiter=1000000, maxfun=1000000,
                      ftol=1e-3, xtol=1e-3, disp=False)
    except OverflowError:
        _core = 0
        _halo = 0

    # Palomar PS = 0.021, KP PS = 0.0175797
    _core = core_p[2] * ps

    if return_halo:
        _halo = p[2] * ps
        return _core, _halo
    else:
        return _core


def log_gauss_score(_x, _mu=1.27, _sigma=0.17):
    """
        _x: pixel for pixel in [1,2048] - source FWHM.
            has a max of 1 around 35 pix, drops fast to the left, drops slower to the right
    """
    return np.exp(-(np.log(np.log(_x)) - _mu)**2 / (2*_sigma**2)) / 2


def gauss_score(_r, _mu=0, _sigma=512):
    """
        _r - distance from centre to source in pix
    """
    return np.exp(-(_r - _mu)**2 / (2*_sigma**2)) / 2


def rho(x, y, x_0=1024, y_0=1024):
    return np.sqrt((x-x_0)**2 + (y-y_0)**2)


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


def make_img(_path, _win, _x=None, _y=None):
    """

    :param _path: path to 100p.fits
    :param _win: window width
    :param _x: source x position -- if known in advance
    :param _y: source y position -- if known in advance

    :return: cropped image
    """
    scidata = fits.open(os.path.join(_path, '100p.fits'))[0].data

    if _x is None and _y is None:
        # extract sources
        sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                                "A_IMAGE", "B_IMAGE", "FWHM_IMAGE", "FLAGS"],
                                config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
                                sexpath="sex")

        out = sew(os.path.join(_path, '100p.fits'))
        # sort by FWHM
        out['table'].sort('FWHM_IMAGE')
        # descending order
        out['table'].reverse()

        print(out['table'])  # This is an astropy table.

        # get first 5 and score them:
        scores = []
        for sou in out['table'][0:10]:
            if sou['FWHM_IMAGE'] > 1:
                score = log_gauss_score(sou['FWHM_IMAGE']) + gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE']))
            else:
                score = 0  # it could so happen that reported FWHM is 0
            scores.append(score)

        print('scores: ', scores)

        N_sou = len(out['table'])
        # do not crop large planets and crowded fields
        if N_sou != 0 and N_sou < 30:
            # sou_xy = [out['table']['X_IMAGE'][0], out['table']['Y_IMAGE'][0]]
            best_score = np.argmax(scores) if len(scores) > 0 else 0
            # sou_size = np.max((int(out['table']['FWHM_IMAGE'][best_score] * 3), 90))
            scidata_cropped = scidata[out['table']['YPEAK_IMAGE'][best_score] - _win:
                                      out['table']['YPEAK_IMAGE'][best_score] + _win + 1,
                                      out['table']['XPEAK_IMAGE'][best_score] - _win:
                                      out['table']['XPEAK_IMAGE'][best_score] + _win + 1]
        else:
            # use a simple max instead:
            x, y = np.unravel_index(scidata.argmax(), scidata.shape)
            scidata_cropped = scidata[x - _win: x + _win + 1,
                                      y - _win: y + _win + 1]
    else:
        x, y = _x, _y
        scidata_cropped = scidata[x - _win: x + _win + 1,
                                  y - _win: y + _win + 1]

    return scidata_cropped


def pca_helper(_args):
    """
    Helper function to run PCA in parallel for multiple sources
    TODO: implement parallel processing!

    :param _args:
    :return:
    """
    # unpack args
    _trimmed_frame, _win, _sou_name, _sou_dir, _out_path, \
    _library, _library_names_short, _fwhm, _plsc, _sigma, _nrefs, _klip = _args

    # run pca
    try:
        pca(_trimmed_frame=_trimmed_frame, _win=_win, _sou_name=_sou_name,
            _sou_dir=_sou_dir, _out_path=_out_path,
            _library=_library, _library_names_short=_library_names_short,
            _fwhm=_fwhm, _plsc=_plsc, _sigma=_sigma, _nrefs=_nrefs, _klip=_klip)
    finally:
        return


def pca(_trimmed_frame, _win, _sou_name, _sou_dir, _out_path,
        _library, _library_names_short,
        _fwhm, _plsc=0.0175797, _sigma=5, _nrefs=5, _klip=1):
    """

    :param _trimmed_frame: image cropped around the source
    :param _win: window half-size in pixels
    :param _sou_name: source name
    :param _sou_dir: full Robo-AO source name
    :param _out_path: output path
    :param _library: PSF library
    :param _library_names_short: source names from the library
    :param _fwhm: FWHM
    :param _plsc: contrast curve parameter - plate scale (check if input img is upsampled)
    :param _sigma: contrast curve parameter - sigma level
    :param _nrefs: number of reference sources to use
    :param _klip: number of components to keep
    :return:
    """

    # Filter the trimmed frame with IUWT filter, 2 coeffs
    filtered_frame = (vip.var.cube_filter_iuwt(
        np.reshape(_trimmed_frame, (1, np.shape(_trimmed_frame)[0], np.shape(_trimmed_frame)[1])),
        coeff=5, rel_coeff=1))

    cy1, cx1 = np.unravel_index(filtered_frame[0].argmax(), filtered_frame[0].shape)
    _fwhm = bad_obs_check(filtered_frame[0][cy1-30:cy1+30+1, cx1-30:cx1+30+1], return_halo=False)

    # Print the resolution element size
    print('Using resolution element size = ', _fwhm)

    # Center the filtered frame
    centered_cube, shy, shx = (vip.calib.cube_recenter_gauss2d_fit(array=filtered_frame, pos_y=_win,
                                                                   pos_x=_win, fwhm=_fwhm,
                                                                   subi_size=6, nproc=1,
                                                                   full_output=True))
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
    library_notmystar = _library[~np.in1d(_library_names_short, _sou_name)]
    cross_corr = np.zeros(len(library_notmystar))
    flattened_frame = np.ndarray.flatten(centered_frame)

    for c in range(len(library_notmystar)):
        cross_corr[c] = stats.pearsonr(flattened_frame, np.ndarray.flatten(library_notmystar[c, :, :]))[0]

    cross_corr_sorted, index_sorted = (np.array(zip(*sorted(zip(cross_corr, np.arange(len(cross_corr))),
                                                            key=operator.itemgetter(0), reverse=True))))
    index_sorted = np.int_(index_sorted)
    library = library_notmystar[index_sorted[0:_nrefs], :, :]
    print('Library correlations = ', cross_corr_sorted[0:_nrefs])

    # Do PCA
    reshaped_frame = np.reshape(centered_frame,
                                (1, np.shape(centered_frame)[0], np.shape(centered_frame)[1]))
    pca_frame = vip.pca.pca(reshaped_frame, np.zeros(1), library, ncomp=_klip)

    pca_file_name = os.path.join(_out_path, _sou_dir + '_pca.fits')

    # remove fits if already exists
    if os.path.isfile(pca_file_name):
        os.remove(pca_file_name)

    # save fits after PCA
    hdu = fits.PrimaryHDU(pca_frame)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(pca_file_name)

    # save png after PCA
    # scale for beautification:
    scidata = deepcopy(pca_frame)
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata / norm * 65535)
    # logarithmic_corrected = exposure.adjust_log(img_as_float(scidata/norm) + 1, 1)
    # print(np.min(np.min(scidata)), np.max(np.max(scidata)))

    # add more contrast to the image:
    # scidata_corrected = exposure.equalize_adapthist(scidata, clip_limit=0.03)
    p_1, p_2 = np.percentile(scidata, (5, 100))
    # scidata_corrected = exposure.rescale_intensity(scidata, in_range=(p_1, p_2))

    # perform local histogram equalization instead:
    scidata_corrected = exposure.equalize_adapthist(scidata, clip_limit=0.03)

    plt.close('all')
    fig = plt.figure(_sou_dir)
    fig.set_size_inches(3, 3, forward=False)
    # ax = fig.add_subplot(111)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scidata_corrected, cmap='gray', origin='lower', interpolation='nearest')
    # add scale bar:
    # draw a horizontal bar with length of 0.1*x_size
    # (ax.transData) with a label underneath.
    bar_len = pca_frame.shape[0] * 0.1
    bar_len_str = '{:.1f}'.format(bar_len * 36 / 1024 / 2)
    asb = AnchoredSizeBar(ax.transData,
                          bar_len,
                          bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                          loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
    ax.add_artist(asb)

    # save figure
    fig.savefig(os.path.join(_out_path, _sou_dir + '_pca.png'), dpi=300)

    # Make contrast curve
    [con, cont, sep] = (vip.phot.contrcurve.contrast_curve(cube=reshaped_frame, angle_list=np.zeros(1),
                                                           psf_template=psf_template,
                                                           cube_ref=library, fwhm=_fwhm, pxscale=_plsc,
                                                           starphot=center_flux, sigma=_sigma,
                                                           ncomp=_klip, algo='pca-rdi-fullfr',
                                                           debug='false',
                                                           plot='false', nbranch=3, scaling=None,
                                                           mask_center_px=_fwhm, fc_rad_sep=6))

    plt.close('all')
    fig = plt.figure('Contrast curve for {:s}'.format(_sou_dir), figsize=(8, 3.5), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_title(_sou_dir)  # , fontsize=14)
    ax.plot(sep, -2.5 * np.log10(cont), 'k-', linewidth=2.5)
    ax.set_xlim([0.2, 1.45])
    ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
    ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
    ax.set_ylim([0, 8])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.grid(linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(_out_path, _sou_dir + '_contrast_curve.png'), dpi=200)

    # save txt for nightly median calc/plot
    with open(os.path.join(_out_path, _sou_dir + '_contrast_curve.txt'), 'w') as f:
        for s, dm in zip(sep, -2.5 * np.log10(cont)):
            f.write('{:.3f} {:.3f}\n'.format(s, dm))


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Becky\'s PCA pipeline')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='run computation in parallel mode')

    args = parser.parse_args()

    # script absolute location
    abs_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    ''' Get config data '''
    # load config data
    config = ConfigParser.RawConfigParser()
    # config.read(os.path.join(abs_path, 'config.ini'))
    if args.config_file[0] not in ('/', '~'):
        if os.path.isfile(os.path.join(abs_path, args.config_file)):
            config.read(os.path.join(abs_path, args.config_file))
            if len(config.read(os.path.join(abs_path, args.config_file))) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')
    else:
        if os.path.isfile(args.config_file):
            config.read(args.config_file)
            if len(config.read(args.config_file)) == 0:
                raise Exception('Failed to load config file')
        else:
            raise IOError('Failed to find config file')

    # path to (standard) pipeline data:
    path_pipe = config.get('Path', 'path_pipe')
    # path to Becky-pipeline data (output):
    path_pca = config.get('Path', 'path_pca')
    # path to PSF library:
    path_psf_reference_library = config.get('Path', 'path_psf_reference_library')
    path_psf_reference_library_short_names = config.get('Path', 'path_psf_reference_library_short_names')
    psf_reference_library = fits.open(path_psf_reference_library)[0].data
    psf_reference_library_short_names = np.genfromtxt(path_psf_reference_library_short_names, dtype='|S')

    win = int(config.get('PCA', 'win'))
    plate_scale = float(config.get('PCA', 'plate_scale'))
    sigma = float(config.get('PCA', 'sigma'))
    nrefs = float(config.get('PCA', 'nrefs'))
    klip = float(config.get('PCA', 'klip'))

    planets_prog_num = int(config.get('Programs', 'planets'))

    # try processing today if no date provided
    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')

    ''' Scientific images '''
    path = os.path.join(path_pipe, datetime.datetime.strftime(date, '%Y%m%d'))

    # path to pipelined data exists?
    if os.path.exists(path):
        # keep args to run pca for all sources in a safe cold place:
        args_pca = []
        # path to output for date
        path_data = os.path.join(path_pca, datetime.datetime.strftime(date, '%Y%m%d'))
        if not os.path.exists(path_pca):
            os.mkdir(path_pca)
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        for pot in ('high_flux', 'faint'):
            if os.path.exists(os.path.join(path, pot)):
                print(pot.replace('_', ' ').title())
                if not os.path.exists(os.path.join(path_data, pot)):
                    os.mkdir(os.path.join(path_data, pot))
                for sou_dir in sorted(os.listdir(os.path.join(path, pot))):
                    # frame_name = os.path.splitext(sou_dir)[0]
                    path_sou = os.path.join(path, pot, sou_dir)
                    tmp = sou_dir.split('_')
                    try:
                        # prog num set?
                        prog_num = int(tmp[0])
                        # stack name back together:
                        sou_name = '_'.join(tmp[1:-5])
                    except ValueError:
                        prog_num = 9999
                        # was it a pointing observation?
                        if 'pointing' in tmp:
                            sou_name = 'pointing'
                        else:
                            sou_name = '_'.join(tmp[0:-5])
                    # filter used:
                    filt = tmp[-4:-3][0]
                    # date and time of obs:
                    time = datetime.datetime.strptime(tmp[-2] + tmp[-1].split('.')[0], '%Y%m%d%H%M%S')

                    # do not try to process planets:
                    if prog_num == planets_prog_num and sou_name.title() not in ('Pluto', ):
                        continue

                    ''' go off with processing: '''
                    # trimmed image:
                    trimmed_frame = (make_img(_path=path_sou, _win=win))

                    # Check of observation passes quality check:

                    try:
                        cy1, cx1 = np.unravel_index(trimmed_frame.argmax(), trimmed_frame.shape)
                        core, halo = bad_obs_check(trimmed_frame[cy1-30:cy1+30+1, cx1-30:cx1+30+1],
                                                   ps=plate_scale)
                    except:
                        core = 0.14
                        halo = 1.0
                        continue
                    if core > 0.14 and halo < 1.0:
                        # run PCA
                        args_pca.append([trimmed_frame, win, sou_name, sou_dir, os.path.join(path_data, pot),
                                         psf_reference_library, psf_reference_library_short_names,
                                         core/plate_scale, plate_scale, sigma, nrefs, klip])
                    else:
                        print('Bad Observation. Faint star pipeline coming soon . . . ')

        # run computation:
        if len(args_pca) > 0:
            if args.parallel:
                raise NotImplementedError
            else:
                for arg_pca in args_pca:
                    pca_helper(arg_pca)
