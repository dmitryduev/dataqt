from __future__ import print_function

import argparse
import os
from astropy.io import fits
import image_registration
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter
import pyprind
from numba import jit
import multiprocessing

# set up plotting
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


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


@jit
def shift2d(fftn, ifftn, data, deltax, deltay, xfreq_0, yfreq_0, return_abs=False, return_real=True):
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


def stack_images(_files_list, _path_out='./', cx0=None, cy0=None, _win=None,
                 _obs=None, _nthreads=4, _interactive_plot=True, _v=True):
    """

    :param _files_list:
    :param _path_out:
    :param _obs:
    :param _nthreads:
    :param _interactive_plot:
    :param _v:
    :return:
    """

    if _obs is None:
        _obs = os.path.split(_files_list[0])[1]

    if _interactive_plot:
        plt.axes([0., 0., 1., 1.])
        plt.ion()
        plt.grid('off')
        plt.axis('off')
        plt.show()

    numFrames = len(_files_list)

    # use first image as pivot:
    with fits.open(_files_list[0]) as _hdulist:
        im1 = np.array(_hdulist[0].data, dtype=np.float)  # do proper casting
        image_size = _hdulist[0].shape
        # get fits header for output:
        header = _hdulist[0].header
        if cx0 is None:
            cx0 = header.get('NAXIS1') // 2
        if cy0 is None:
            cy0 = header.get('NAXIS2') // 2
        if _win is None:
            _win = int(np.min([cx0, cy0]))
        im1 = im1[cy0 - _win: cy0 + _win, cx0 - _win: cx0 + _win]

    # Sum of all frames (with not too large a shift and chi**2)
    summed_frame = np.zeros(image_size)

    # frame_num x y ex ey:
    shifts = np.zeros((numFrames, 5))

    # set up frequency grid for shift2d
    ny, nx = image_size
    xfreq_0 = np.fft.fftfreq(nx)[np.newaxis, :]
    yfreq_0 = np.fft.fftfreq(ny)[:, np.newaxis]

    fftn, ifftn = image_registration.fft_tools.fast_ffts.get_ffts(nthreads=_nthreads, use_numpy_fft=False)

    if _v:
        bar = pyprind.ProgBar(numFrames-1, stream=1, title='Registering frames')

    fn = 0
    for jj, _file in enumerate(_files_list[1:]):
        with fits.open(_file) as _hdulist:
            for ii, _ in enumerate(_hdulist):
                img = np.array(_hdulist[ii].data, dtype=np.float)  # do proper casting

                # tic = _time()
                # img_comp = gaussian_filter(img, sigma=5)
                img_comp = img
                img_comp = img_comp[cy0 - _win: cy0 + _win, cx0 - _win: cx0 + _win]
                # print(_time() - tic)

                # tic = _time()
                # chi2_shift -> chi2_shift_iterzoom
                dy2, dx2, edy2, edx2 = image_registration.chi2_shift(im1, img_comp, nthreads=_nthreads,
                                                                     upsample_factor='auto', zeromean=True)
                img = shift2d(fftn, ifftn, img, -dy2, -dx2, xfreq_0, yfreq_0)
                # print(_time() - tic, '\n')

                if np.sqrt(dx2 ** 2 + dy2 ** 2) > 0.8 * _win:
                    # skip frames with too large a shift
                    pass
                else:
                    # otherwise store the shift values and add to the 'integrated' image
                    shifts[fn, :] = [fn, -dx2, -dy2, edx2, edy2]
                    summed_frame += img

                if _interactive_plot:
                    plt.imshow(summed_frame, cmap='gray', origin='lower', interpolation='nearest')
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

    export_fits(os.path.join(_path_out, _obs + '.stacked.fits'),
                summed_frame, header)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='stack multiple fits images')
    parser.add_argument('fits_files', nargs='*')  # This is it!!
    parser.add_argument('--cx0', metavar='cx0', action='store', dest='cx0',
                        help='x lock position [pix]', type=int)
    parser.add_argument('--cy0', metavar='cy0', action='store', dest='cy0',
                        help='y lock position [pix]', type=int)
    parser.add_argument('--win', metavar='win', action='store', dest='win',
                        help='window size [pix]', type=int)
    args = parser.parse_args()
    fits_files = args.fits_files
    # print(fits_files)

    # number of threads available:
    n_cpu = multiprocessing.cpu_count()

    stack_images(_files_list=fits_files, _path_out='./',
                 cx0=args.cx0, cy0=args.cy0, _win=args.win,
                 _obs='object', _nthreads=n_cpu, _interactive_plot=True, _v=True)
