from __future__ import print_function
from astropy.modeling import models, fitting
import astropy.io.fits as fits
try:
    import sewpy
except:
    pass

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import functools
import subprocess
import multiprocessing
import datetime
import inspect
import ConfigParser

# Force matplotlib to not use any Xwindows backend.
import matplotlib

import seaborn as sns
# sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')

matplotlib.use('Agg')


# some metrics to score SExtractor solutions.
# there are 2 of them, so them sum up to a max of 1
def log_gauss_score(_x, _mu=1.27, _sigma=0.17):
    """
        _x: pixel for pixel in [1,2048] - source FWHM.
            has a max of 1 around 35 pix, drops fast to the left, drops slower to the right
    """
    return np.exp(-(np.log(np.log(_x)) - _mu)**2 / (2*_sigma**2)) / 2


def gauss_score(_r, _mu=0, _sigma=256):
    """
        _r - distance from centre to source in pix
    """
    return np.exp(-(_r - _mu)**2 / (2*_sigma**2)) / 2


def rho(x, y, x_0=512, y_0=512):
    return np.sqrt((x-x_0)**2 + (y-y_0)**2)


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


def replace_extension(filename, new_extension):
    """ From the name of a file (possibly full path) change the extension
        by another one given by user"""
    if new_extension[0] != ".":
        new_extension = "." + new_extension
    path, name = os.path.split(filename)
    name_root, name_ext = os.path.splitext(name)
    return os.path.join(path, name_root + new_extension)


def getModefromMag(mag):
    """
        VICD mode depending on the object magnitude
    """
    m = float(mag)
    if m < 8:
        mode = '6'
    elif 8 <= m < 10:
        mode = '7'
    elif 10 <= m < 12:
        mode = '8'
    elif 12 <= m < 13:
        mode = '9'
    elif m >= 13:
        mode = '10'
    return mode


class Star(object):
    """ Define a star by its coordinates and modelled FWHM
        Given the coordinates of a star within a 2D array, fit a model to the star and determine its
        Full Width at Half Maximum (FWHM).The star will be modelled using astropy.modelling. Currently
        accepted models are: 'Gaussian2D', 'Moffat2D'
    """

    _GAUSSIAN2D = 'Gaussian2D'
    _MOFFAT2D = 'Moffat2D'
    _MODELS = set([_GAUSSIAN2D, _MOFFAT2D])

    def __init__(self, x0, y0, data, model_type=_GAUSSIAN2D, box=100, out_path='./'):
        """ Instantiation method for the class Star.
        The 2D array in which the star is located (data), together with the pixel coordinates (x0,y0) must be
        passed to the instantiation method. .
        """
        self.x = x0
        self.y = y0
        self._box = box
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
        elif self.model_type == self._GAUSSIAN2D:
            sigma_x, sigma_y = [model_dict[ii] for ii in ("x_stddev_0", "y_stddev_0")]
            FWHM = 2.3548 * np.mean([sigma_x, sigma_y])
        return FWHM

    @memoize
    def _fit_model(self):
        fit_p = fitting.LevMarLSQFitter()
        model = self._initialize_model()
        p = fit_p(model, self._XGrid, self._YGrid, self.data)
        return p

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

    def plot_resulting_model(self, date_str=None):
        """ Make a plot showing data, model and residuals. """
        data = self.data
        model = self.model()(self._XGrid, self._YGrid)
        residuals = data - model
        # fig = plt.figure(figsize=(10, 2.5))
        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        # print(sns.diverging_palette(10, 220, sep=80, n=7))
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.RdBu_r)
        # plt.colorbar()
        # plt.title('Data', fontsize=14)
        # plt.tick_params(labelbottom='off', labelleft='off') # labels along the bottom edge are off
        plt.grid('off')
        ax1.set_axis_off()
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.imshow(model, origin='lower', interpolation='nearest',
                   vmin=data.min(), vmax=data.max(), cmap=plt.cm.RdBu_r)
        # plt.colorbar()
        ax2.set_axis_off()
        # plt.title('Model', fontsize=14)
        # plt.tick_params(labelbottom='off', labelleft='off')  # labels along the bottom edge are off
        plt.grid('off')
        ax3 = fig.add_subplot(1, 3, 3, sharey=ax1)
        plt.imshow(residuals, origin='lower', interpolation='nearest', cmap=plt.cm.RdBu_r)
        # plt.colorbar()
        # plt.title('Residual', fontsize=14)
        # plt.tick_params(labelbottom='off', labelleft='off')  # labels along the bottom edge are off
        plt.grid('off')
        ax3.set_axis_off()
        # plt.tight_layout()
        # title
        if date_str is not None:
            # fig.subplots_adjust(top=0.8, wspace=0.0)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
            # print date string:
            # plt.suptitle(date_str, fontsize=16)

            if not os.path.exists(os.path.join(self.out_path, 'plots')):
                os.mkdir(os.path.join(self.out_path, 'plots'))
            path = os.path.join(self.out_path, 'plots', date_str[:date_str.index('_')])
            if not os.path.exists(path):
                os.mkdir(path)
            # plt.savefig(os.path.join(path, '{:s}.png'.format(date_str)),
            #             bbox_inches='tight', dpi=200)
            plt.savefig(os.path.join(path, '{:s}.png'.format(date_str)), dpi=200)
            # plt.show()

        else:
            plt.show()


class StarField(object):
    """ Fit a model to a list of stars from an image, estimate their FWHM, write it to the image.
    To initialize the object you need the name of an astronomical image (im_name) and the name of a text file
     with the coordinates of the stars to be used to estimate the FWHM. The file must contain two columns with the
     RA and DEC of the stars, one row per star. If image pixels are given instead of RA and DEC, the wcs=False flag
     must be passed.
    """
    def __init__(self, im_name, coords_file, model_type, out_path='./', pipe_path=None,
                 wcs=True):
        self.im_name = im_name
        # self.im_data = fits.open(im_name)[0].data
        data = fits.open(im_name, ignore_missing_end=True, memmap=True)
        # simply add up all data from the shorter exposures
        z = data[0].data
        self.nframes = len(data)
        for i in range(1, len(data)):
            z += data[i].data
        self.im_data = z

        try:
            mag = data[0].header['MAGNITUD']
            self.mode = getModefromMag(mag)
        except:
            self.mode = '0'

        # data[0].data = z / float(self.nframes)
        # fits_stacked = self.im_name + '.stacked'
        # if not os.path.isfile(fits_stacked):
        #     data.writeto(im_name+'.stacked')

        self.coords_file = coords_file
        self._wcs = wcs
        self.out_path = out_path
        self.pipe_path = pipe_path
        self.model_type = model_type
        try:
            # get time stamp
            underscores = [i for i, c in enumerate(im_name) if c == '_']
            dots = [i for i, c in enumerate(im_name) if c == '.']
            date_str = im_name[underscores[-2]+1:dots[-2]]
            self.t_stamp = datetime.datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            self.date_str = date_str
        except:
            self.date_str = None

        try:
            basename = os.path.basename(im_name)
            underscores = [i for i, c in enumerate(basename) if c == '_']
            self.filter = basename[underscores[1]+1:underscores[2]]
        except:
            self.filter = 'Si'

    def __iter__(self):
        """ Iterate over all the stars defined in the coords_file."""
        return iter(self._star_list)

    @property
    def star_coords(self):
        """ Read from coords_file the coordinates of the stars.
        The file must have two columns, with one star per row. If the coordinates are in (RA,DEC) we will transform
        them into image pixels.
        """
        # x, y = np.genfromtxt(self.coords_file, unpack=True)
        # if self._wcs:  # if x,y are not pixels but RA,DEC
        #     with fits.open(self.im_name, 'readonly') as im:
        #         w = wcs.WCS(im[0].header, im)
        #     y, x = w.all_world2pix(x, y, 1)
        # try:
        #     len(x)
        #     return zip(x, y)
        # except:
        #     return [[x, y]]

        try:
            # try sextractor:
            # extract sources:
            sew = sewpy.SEW(
                params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                        "FWHM_IMAGE", "FLAGS"],
                config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
                sexpath="sex")
            # By default, this assumes that SExtractor can be called as "sex"
            # If this is not the case, or if the executable is not in your path,
            # specify the path by adding the argument sexpath="/path/to/sextractor"
            # to the above instantiation.

            # create a tmp stacked seeing image, corrected for darks/flats:
            fits_stacked = self.im_name+'.stacked'
            path_calib = os.path.join(self.pipe_path,
                                      datetime.datetime.strftime(self.t_stamp, '%Y%m%d'),
                                      'calib')
            # flat field:
            flat_fits = os.path.join(path_calib, 'flat_{:s}.fits'.format(self.filter))
            # dark current:
            dark_fits = os.path.join(path_calib, 'dark_{:s}.fits'.format(self.mode))
            if not os.path.isfile(fits_stacked):
                hdulist_dark = fits.open(dark_fits)
                hdulist_flat = fits.open(flat_fits)

                dark = hdulist_dark[0].data
                flat = hdulist_flat[0].data

                hdu = fits.PrimaryHDU((self.im_data / self.nframes - dark) / flat)
                hdu.writeto(fits_stacked)

            # for the above approach to work, the image must first be corrected for flats/darks

            out = sew(fits_stacked)
            # out = sew(self.im_name)
            # sort according to FWHM
            out['table'].sort('FWHM_IMAGE')
            # descending order
            out['table'].reverse()

            # print(out['table'][0:5])
            # print(len(out['table']), self.nframes)
            # print(out['table']['YPEAK_IMAGE'][0], out['table']['XPEAK_IMAGE'][0])
            # print(np.unravel_index(self.im_data.argmax(), self.im_data.shape))

            # get first 5 and score them:
            scores = []
            for sou in out['table'][0:5]:
                if sou['FWHM_IMAGE'] > 1:
                    score = log_gauss_score(sou['FWHM_IMAGE']) + gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE']))
                else:
                    score = 0  # it could so happen that reported FWHM is 0
                scores.append(score)
            # print(scores)

            # remove the tmp fits:
            p = subprocess.Popen(['rm', '-f', fits_stacked])
            p.wait()

            # for ii in range(len(out['table'])):
            #     if out['table']['FLAGS'][ii] == 0:
            #         x, y = out['table']['YPEAK_IMAGE'][ii], out['table']['XPEAK_IMAGE'][ii]

            best_score = np.argmax(scores) if len(scores) > 0 else 0
            x, y = out['table']['YPEAK_IMAGE'][best_score], out['table']['XPEAK_IMAGE'][best_score]

        except Exception as err:
            print(str(err))
            # use a simple max instead:
            x, y = np.unravel_index(self.im_data.argmax(), self.im_data.shape)

        return [[x, y]]

    @property
    def _star_list(self):
        """ Return a list of Star objects from the image data and the coordinates of the stars."""
        return [Star(x0, y0, self.im_data, self.model_type, out_path=self.out_path) for (x0,y0) in self.star_coords]

    def FWHM(self):
        """ Determine the median and median absolute deviation of the FWHM (seeing) of the image. """
        fwhm_stars = np.array([star.fwhm for star in self._star_list])
        if np.median(fwhm_stars) * 36/self.im_data.shape[0] > 0.4:
            self._star_list[0].plot_resulting_model(self.date_str)
        return np.median(fwhm_stars), np.median( np.abs( np.median(fwhm_stars) - fwhm_stars ) )

    def _write_FWHM(self):
        seeing, std = self.FWHM()
        print('Estimated seeing = {:.3f} pixels'.format(seeing))
        print('Estimated seeing = {:.3f}\"'.format(seeing*36/self.im_data.shape[0]))
        # if there are more than one star in an image:
        # print(std, 'Median absolute deviation of Seeing (in pixels)')

        ''' Write the FWHM to the header of the fits image.'''
        # with fits.open(self.im_name, 'update') as im:
        #     im[0].header["seeing"] = (self.FWHM()[0], 'Seeing estimated in pixels')
        #     im[0].header["seeing_MAD"] = (self.FWHM()[1], 'Median absolute deviation of Seeing (in pixels)')

        if seeing * 36 / self.im_data.shape[0] > 0.4:
            date = self.date_str[:self.date_str.index('_')]
            path = os.path.join(self.out_path, 'plots', date)

            # first check if it's there already:
            f_seeing = os.path.join(path, 'seeing.{:s}.txt'.format(date))
            if os.path.isfile(f_seeing):
                with open(f_seeing, 'r') as f:
                    f_lines = f.readlines()

            entry = '{:19s} {:7.3f} {:7.3f}\n'.format(str(self.t_stamp), seeing,
                                                      seeing*36/self.im_data.shape[0])
            if (not os.path.isfile(f_seeing)) or (entry not in f_lines):
                with open(os.path.join(path, 'seeing.{:s}.txt'.format(date)), 'a+') as f:
                    f.write(entry)

        return None


def calculate_seeing(args):
    """ Program to estimate the seeing from an image and a list of estimates
    for the positions of stars.
    """
    # for im_name, star_cat in zip(args.input, args.cat):
    #     im = StarField(im_name, star_cat, args.model, wcs=args.wcs)
    #     im._write_FWHM()

    im = StarField(args.input[0], args.cat, args.model, out_path=args.out_path, pipe_path=args.pipe_path,
                   wcs=args.wcs)
    im._write_FWHM()


def calc_seeing(f_in, model='Gaussian2D', out_path='./', pipe_path=None):
    # to run from seeing.py
    im = StarField(f_in, '', model, out_path=out_path, pipe_path=pipe_path, wcs=False)
    im._write_FWHM()


def process_date(_args):
    _path, _seeing_imgs = _args
    for _seeing_img in _seeing_imgs:
        # print(_path, seeing_img)
        # call(['gtar', '-jxvf', '{:s}'.format(os.path.join(_path, seeing_img))])
        print('unbzipping {:s}'.format(_seeing_img))
        p1 = subprocess.Popen(['bzip2', '-dk', '{:s}'.format(os.path.join(_path, _seeing_img))])
        p1.wait()
        _unbzipped = os.path.splitext(_seeing_img)[0]
        _unbzipped_base = _unbzipped
        # print(_unbzipped)
        # get time stamp
        _underscores = [i for i, c in enumerate(_unbzipped) if c == '_']
        dots = [i for i, c in enumerate(_unbzipped) if c == '.']
        date_str = _unbzipped[_underscores[-2]+1:dots[-2]]
        _t_stamp = datetime.datetime.strptime(date_str, '%Y%m%d_%H%M%S')
        # print(_t_stamp)
        _unbzipped = '{:s}'.format(os.path.join(_path, _unbzipped))
        # print(_unbzipped)
        p2 = subprocess.Popen(['mv', _unbzipped, 'tmp/'])
        p2.wait()

        # calculate seeing:
        print('processing {:s}'.format(_seeing_img))
        calc_seeing(os.path.join('tmp', _unbzipped_base))


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Calculate the FWHM (seeing) of an image '
                                                 'by fitting its stars.')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)
    # parser.add_argument('-p', '--parallel', action='store_true',
    #                     help='run computation in parallel mode')

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

    # path to raw data:
    path_raw = config.get('Path', 'path_raw')
    # path to (standard) pipeline data:
    path_pipe = config.get('Path', 'path_pipe')
    # path to output seeing data:
    path_seeing = config.get('Path', 'path_seeing')

    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')

    path = os.path.join(path_raw, datetime.datetime.strftime(date, '%Y%m%d'))
    # print(path)
    path_tmp = os.path.join(path_seeing, 'tmp/')

    if not os.path.exists(path_seeing):
        os.mkdir(path_seeing)
    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)

    # path exists?
    if os.path.exists(path):
        try:
            # taken seeing measurements?
            seeing_imgs = sorted([f for f in os.listdir(path) if 'seeing' in f and 'bz2' in f and
                                  f[0] != '.'])
            for seeing_img in seeing_imgs:
                # print(path, seeing_img)
                # call(['gtar', '-jxvf', '{:s}'.format(os.path.join(path, seeing_img))])
                print('unbzipping {:s}'.format(seeing_img))
                p0 = subprocess.Popen(['cp', os.path.join(path, seeing_img), path_tmp])
                p0.wait()
                p1 = subprocess.Popen(['bzip2', '-dk',
                                       '{:s}'.format(os.path.join(path_tmp, seeing_img))])
                # p1 = subprocess.Popen(['tar', '-jxvf',
                #                       '{:s}'.format(os.path.join(path, seeing_img)),
                #                       '-C', path_tmp])
                p1.wait()
                unbzipped = os.path.splitext(seeing_img)[0]
                unbzipped_base = unbzipped
                # print(unbzipped)
                # get time stamp
                underscores = [i for i, c in enumerate(unbzipped) if c == '_']
                dots = [i for i, c in enumerate(unbzipped) if c == '.']
                date_str = unbzipped[underscores[-2]+1:dots[-2]]
                t_stamp = datetime.datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                # print(t_stamp)
                unbzipped = '{:s}'.format(os.path.join(path, unbzipped))
                # print(unbzipped)
                # p2 = subprocess.Popen(['mv', unbzipped, path_tmp])
                # p2.wait()

                # calculate seeing:
                print('processing {:s}'.format(seeing_img))
                calc_seeing(os.path.join(path_tmp, unbzipped_base),
                            out_path=path_seeing,
                            pipe_path=path_pipe)

                # remove unbzipped + copied files:
                p3 = subprocess.Popen(['rm', '-f', os.path.join(path_tmp, unbzipped_base)])
                p3.wait()
                p4 = subprocess.Popen(['rm', '-f', os.path.join(path_tmp, seeing_img)])
                p4.wait()
        except Exception as err:
            print(str(err))
