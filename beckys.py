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
from scipy.signal import savgol_filter
from scipy import stats
import operator
import sewpy
import argparse
import multiprocessing

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
            if sou['FWHM_IMAGE'] > 0.01:
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

    :param _args:
    :return:
    """
    # unpack args
    _trimmed_frame, _win, _sou_name, _sou_dir, _library_path, _out_path, _filt, \
    plsc, sigma, _nrefs, _klip = _args
    # run pca
    try:
        pca(_trimmed_frame=_trimmed_frame, _win=_win, _sou_name=_sou_name,
            _sou_dir=_sou_dir, _library_path=_library_path, _out_path=_out_path, _filt=_filt,
            plsc=plsc, sigma=sigma, _nrefs=_nrefs, _klip=_klip)
    finally:
        return


def pca(_trimmed_frame, _win, _sou_name, _sou_dir, _library_path, _out_path, _filt,
        plsc=0.0168876, sigma=5, _nrefs=5, _klip=1):
    """

    :param _trimmed_frame: image cropped around the source
    :param _win: window half-size in pixels
    :param _sou_name: source name
    :param _sou_dir: full Robo-AO source name
    :param _library_path: path to PSF library
    :param _out_path: output path
    :param _filt: filter
    :param plsc: contrast curve parameter - plate scale (check if input img is upsampled)
    :param sigma: contrast curve parameter - sigma level
    :param _nrefs: number of reference sources to use
    :param _klip: number of components to keep
    :return:
    """
    # Filter the trimmed frame with IUWT filter, 2 coeffs
    filtered_frame = (vip.var.cube_filter_iuwt(
        np.reshape(_trimmed_frame, (1, np.shape(_trimmed_frame)[0], np.shape(_trimmed_frame)[1])),
        coeff=5, rel_coeff=1))

    # Choose the resolution element size -- to replace with fitting two gaussians
    mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta = (
        vip.var.fit_2dgaussian(filtered_frame[0], crop=True,
                               cropsize=15, debug=False, full_output=True))
    fwhm = float(np.mean([fwhm_y, fwhm_x]))
    print('Using resolution element size = ', fwhm)

    # Center the filtered frame
    centered_cube, shy, shx = (vip.calib.cube_recenter_gauss2d_fit(array=filtered_frame, pos_y=_win,
                                                                   pos_x=_win, fwhm=fwhm,
                                                                   subi_size=6, nproc=1,
                                                                   full_output=True))
    centered_frame = centered_cube[0]
    if shy > 5 or shx > 5:
        raise TypeError('Centering failed: pixel shifts too big')

    # Do aperture photometry on the central star
    center_aperture = photutils.CircularAperture(
        (int(len(centered_frame) / 2), int(len(centered_frame) / 2)), fwhm / 2.0)
    center_flux = photutils.aperture_photometry(centered_frame, center_aperture)['aperture_sum'][0]

    # Make PSF template for calculating PCA throughput
    psf_template = (
        centered_frame[len(centered_frame) / 2 - 3 * fwhm:len(centered_frame) / 2 + 3 * fwhm,
        len(centered_frame) / 2 - 3 * fwhm:len(centered_frame) / 2 + 3 * fwhm])

    # Import PSF reference library
    frame_filter = _filt
    if frame_filter == 'Sz':
        library = fits.open(os.path.join(_library_path,
                                         'centered_iuwtfiltered_2Coeffs_zfilt_library.fits'))[0].data
        # library_names = np.genfromtxt(os.path.join(_library_path,
        #                                            'zfilt_library_2Coeffs_names.txt'), dtype="|S")
        library_names_short = np.genfromtxt(os.path.join(_library_path,
                                                         'zfilt_library_2Coeffs_names_short.txt'),
                                            dtype="|S")
    elif frame_filter == 'Si':
        library = fits.open(os.path.join(_library_path,
                                         'centered_iuwtfiltered_2Coeffs_ifilt_library.fits'))[0].data
        # library_names = np.genfromtxt(os.path.join(_library_path,
        #                                            'ifilt_library_2Coeffs_names.txt'), dtype="|S")
        library_names_short = np.genfromtxt(os.path.join(_library_path,
                                                         'ifilt_library_2Coeffs_names_short.txt'),
                                            dtype="|S")
    else:
        print("Becky hasn't made a library for this filter yet, so we aren't doing PCA")
        noise_samp, rad_samp = vip.phot.noise_per_annulus(centered_frame, 1, fwhm, False)
        noise_samp_sm = savgol_filter(noise_samp, polyorder=1, mode='nearest',
                                      window_length=int(noise_samp.shape[0] * 0.1))
        n_res_els = np.floor(rad_samp / fwhm * 2 * np.pi)
        ss_corr = np.sqrt(1 + 1.0 / (n_res_els - 1))
        sigma_student = stats.t.ppf(stats.norm.cdf(sigma), n_res_els) / ss_corr
        cont = (sigma_student * noise_samp_sm) / center_flux

        fig = plt.figure('Contrast curve for {:s}'.format(_sou_name), figsize=(8, 3.5), dpi=200)
        ax = fig.add_subplot(111)
        ax.set_title(_sou_dir + '\n Without PCA')  # , fontsize=14)
        ax.plot(rad_samp * plsc, -2.5 * np.log10(cont), 'k-', linewidth=2.5)
        ax.set_xlim([0.2, 1.45])
        ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
        ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
        ax.set_ylim([0, 8])
        ax.set_ylim(ax.get_ylim()[::-1])  # reverse y
        ax.grid(linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(_out_path, _sou_dir + '_NOPCA_contrast_curve.png'), dpi=200)

        # save txt for nightly median calc/plot
        with open(os.path.join(_out_path, _sou_dir + '_NOPCA_contrast_curve.txt'), 'w') as f:
            for s, dm in zip(rad_samp * plsc, -2.5 * np.log10(cont)):
                f.write('{:.3f} {:.3f}\n'.format(s, dm))

        return

    # Choose reference frames via cross correlation
    library_notmystar = library[~np.in1d(library_names_short, _sou_name)]
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

    # scidata_corrected = exposure.equalize_adapthist(scidata, clip_limit=0.03)
    p_1, p_2 = np.percentile(scidata, (5, 100))
    scidata_corrected = exposure.rescale_intensity(scidata, in_range=(p_1, p_2))

    fig = plt.figure(_sou_name)
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
    bar_len_str = '{:.1f}'.format(bar_len * 34.5858 / 1024 / 2)
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
                                                           cube_ref=library, fwhm=fwhm, pxscale=plsc,
                                                           starphot=center_flux, sigma=sigma,
                                                           ncomp=_klip, algo='pca-rdi-fullfr',
                                                           debug='false',
                                                           plot='false', nbranch=3, scaling=None,
                                                           mask_center_px=fwhm, fc_rad_sep=6))

    fig = plt.figure('Contrast curve for {:s}'.format(_sou_name), figsize=(8, 3.5), dpi=200)
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

    parser.add_argument('path_pipe', metavar='path_pipe',
                        action='store', help='path to pipelined data.', type=str)
    parser.add_argument('library_path', metavar='library_path',
                        action='store', help='path to library.', type=str)
    parser.add_argument('output_path', metavar='output_path',
                        action='store', help='output path.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)
    parser.add_argument('--win', metavar='win', action='store', dest='win',
                        help='window size', type=int, default=100)
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='run computation in parallel mode')

    args = parser.parse_args()

    output_path = args.output_path
    library_path = args.library_path

    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')

    win = args.win
    # print(win)

    ''' Scientific images '''
    path = os.path.join(args.path_pipe, datetime.datetime.strftime(date, '%Y%m%d'))

    # path to pipelined data exists?
    if os.path.exists(path):
        # keep args to run pca for all sources in a safe cold place:
        args_pca = []
        # path to output for date
        path_data = os.path.join(output_path, datetime.datetime.strftime(date, '%Y%m%d'))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
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
                    time = datetime.datetime.strptime(tmp[-2] + tmp[-1].split('.')[0],
                                                      '%Y%m%d%H%M%S')

                    ''' go off with processing: '''
                    # trimmed image:
                    trimmed_frame = (make_img(_path=path_sou, _win=win))
                    
                    # run PCA
                    # pca(_trimmed_frame=trimmed_frame, _win=win, _sou_name=sou_name,
                    #     _sou_dir=sou_dir, _library_path=library_path, _out_path=os.path.join(path_data, pot),
                    #     _filt=filt, plsc=0.0168876, sigma=5.0, _nrefs=5, _klip=1)
                    args_pca.append([trimmed_frame, win,
                                     sou_name, sou_dir, library_path, os.path.join(path_data, pot),
                                     filt, 0.0168876, 5.0, 5, 1])

        # run computation:
        if len(args_pca) > 0:
            # parallel?
            if args.parallel:
                raise NotImplemented('VIP forks stuff, so it is not straightforward to parallelize it.')
                # otherwise it would have been as simple as the following:
                # # number of threads available on the system
                # n_cpu = multiprocessing.cpu_count()
                # # create pool (do not create more than necessary)
                # pool = multiprocessing.Pool(min(n_cpu, len(args_pca)))
                # # asynchronously apply pca_helper
                # result = pool.map_async(pca_helper, args_pca)
                # # close bassejn
                # pool.close()  # we are not adding any more processes
                # pool.join()  # wait until all threads are done before going on
                # # get the ordered results
                # # output = result.get()
            # serial?
            else:
                for arg_pca in args_pca:
                    pca_helper(arg_pca)
