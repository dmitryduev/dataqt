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


def make_img(_path, _win):
    scidata = fits.open(os.path.join(_path, '100p.fits'))[0].data

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
    for sou in out['table'][0:5]:
        score = log_gauss_score(sou['FWHM_IMAGE']) + gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE']))
        scores.append(score)

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

    return scidata_cropped





if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='data quality monitoring')

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

                    # Filter the trimmed frame with IUWT filter, 2 coeffs
                    filtered_frame = (vip.var.cube_filter_iuwt(
                        np.reshape(trimmed_frame, (1, np.shape(trimmed_frame)[0], np.shape(trimmed_frame)[1])),
                        coeff=5, rel_coeff=1))

                    # Choose the resolution element size -- to replace with fitting two gaussians
                    mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta = (
                                    vip.var.fit_2dgaussian(filtered_frame[0], crop=True,
                                                           cropsize=15, debug=False, full_output=True))
                    fwhm = np.mean([fwhm_y, fwhm_x])
                    print('Using resolution element size = ', fwhm)

                    # Center the filtered frame
                    centered_cube, shy, shx = (vip.calib.cube_recenter_gauss2d_fit(array=filtered_frame, pos_y=win,
                                                                                   pos_x=win, fwhm=fwhm,
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

                    # Define contrast curve parameters
                    plsc = 0.0168876
                    my_sigma = 5.0

                    # Import PSF reference library
                    frame_filter = filt
                    if frame_filter == 'Sz':
                        library = fits.open(os.path.join(library_path,
                                                         'centered_iuwtfiltered_2Coeffs_zfilt_library.fits'))[0].data
                        library_names = np.genfromtxt(os.path.join(library_path,
                                                                   'zfilt_library_2Coeffs_names.txt'), dtype="|S")
                        library_names_short = np.genfromtxt(os.path.join(library_path,
                                                                         'zfilt_library_2Coeffs_names_short.txt'),
                                                            dtype="|S")
                    elif frame_filter == 'Si':
                        library = fits.open(os.path.join(library_path,
                                                         'centered_iuwtfiltered_2Coeffs_ifilt_library.fits'))[0].data
                        library_names = np.genfromtxt(os.path.join(library_path,
                                                                   'ifilt_library_2Coeffs_names.txt'), dtype="|S")
                        library_names_short = np.genfromtxt(os.path.join(library_path,
                                                                         'ifilt_library_2Coeffs_names_short.txt'),
                                                            dtype="|S")
                    else:
                        print("Becky hasn't made a library for this filter yet, so we aren't doing PCA")
                        noise_samp, rad_samp = vip.phot.noise_per_annulus(centered_frame, 1, fwhm, False)
                        noise_samp_sm = savgol_filter(noise_samp, polyorder=1, mode='nearest',
                                                      window_length=noise_samp.shape[0] * 0.1)
                        n_res_els = np.floor(rad_samp / fwhm * 2 * np.pi)
                        ss_corr = np.sqrt(1 + 1 / (n_res_els - 1))
                        sigma_student = stats.t.ppf(stats.norm.cdf(my_sigma), n_res_els) / ss_corr
                        cont = (sigma_student * noise_samp_sm) / center_flux

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.set_title(sou_dir + '\n Without PCA', fontsize=14)
                        ax.plot(rad_samp * plsc, -2.5 * np.log10(cont), 'k-', linewidth=3)
                        ax.set_xlim([0.2, 1.45])
                        ax.set_xlabel('Separation [arcseconds]', fontsize=18)
                        ax.set_ylabel('Contrast [$\Delta$mag]', fontsize=18)
                        # plt.gca().invert_yaxis()
                        ax.set_ylim(ax.get_ylim()[::-1])
                        fig.savefig(os.path.join(path_data, pot, sou_dir + '_NOPCA_contrast_curve.jpg'))
                        # raise Exception('No library for this filter yet :(')
                        continue

                    # Choose reference frames via cross correlation
                    nrefs = 5
                    library_notmystar = library[~np.in1d(library_names_short, sou_name)]
                    cross_corr = np.zeros(len(library_notmystar))
                    flattened_frame = np.ndarray.flatten(centered_frame)

                    for c in xrange(len(library_notmystar)):
                        cross_corr[c] = stats.pearsonr(flattened_frame, np.ndarray.flatten(library_notmystar[c, :, :]))[
                            0]

                    cross_corr_sorted, index_sorted = (np.array(zip(*sorted(zip(cross_corr, np.arange(len(cross_corr))),
                                                                            key=operator.itemgetter(0), reverse=True))))
                    index_sorted = np.int_(index_sorted)
                    my_library = library_notmystar[index_sorted[0:nrefs], :, :]
                    print('Library correlations = ', cross_corr_sorted[0:nrefs])

                    # Do PCA
                    reshaped_frame = np.reshape(centered_frame,
                                                (1, np.shape(centered_frame)[0], np.shape(centered_frame)[1]))
                    klip = 1
                    pca_frame = vip.pca.pca(reshaped_frame, np.zeros(1), my_library, ncomp=klip)

                    pca_file_name = os.path.join(path_data, pot, sou_dir + '_pca.fits')

                    # remove fits if already exists
                    if os.path.isfile(pca_file_name):
                        os.remove(pca_file_name)

                    hdu = fits.PrimaryHDU(pca_frame)
                    hdulist = fits.HDUList([hdu])
                    hdulist.writeto(pca_file_name)

                    # Make contrast curve
                    [con, cont, sep] = (vip.phot.contrcurve.contrast_curve(cube=reshaped_frame, angle_list=np.zeros(1),
                                                                           psf_template=psf_template,
                                                                           cube_ref=my_library, fwhm=fwhm, pxscale=plsc,
                                                                           starphot=center_flux, sigma=my_sigma,
                                                                           ncomp=klip, algo='pca-rdi-fullfr',
                                                                           debug='false',
                                                                           plot='false', nbranch=3, scaling=None,
                                                                           mask_center_px=fwhm, fc_rad_sep=6))

                    fig = plt.figure('Contrast curve')
                    ax = fig.add_subplot(111)
                    ax.set_title(sou_dir, fontsize=14)
                    ax.plot(sep, -2.5 * np.log10(cont), 'k-', linewidth=3)
                    ax.set_xlim([0.2, 1.45])
                    ax.set_xlabel('Separation [arcseconds]', fontsize=18)
                    ax.set_ylabel('Contrast [$\Delta$mag]', fontsize=18)
                    # plt.gca().invert_yaxis()
                    ax.set_ylim(ax.get_ylim()[::-1])
                    fig.savefig(os.path.join(path_data, pot, sou_dir + '_contrast_curve.jpg'))


