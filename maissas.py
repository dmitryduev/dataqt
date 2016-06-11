"""
    Strehl calculator for Robo-AO
    MS @ UofH
"""
from __future__ import print_function
import glob
import os
import warnings
import datetime
import sys
import astropy.io.fits as pyfits
from math import sqrt, pow, exp
import numpy as np
from scipy import fftpack, optimize
from astropy.modeling import models, fitting
from scipy.optimize import fmin
from scipy.interpolate import interp1d  # , RectBivariateSpline
import argparse
import ConfigParser
import ast
import inspect


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


def bad_obs_check(p, ps=0.0175797):
    # Palomar PS = 0.021, KP PS = 0.0175797
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
        p0 = [0.0, np.max(pix_vals), 20.0, 2.0]
        p = fmin(residuals, p0, args=(pix_rad, pix_vals), maxiter=1000000, maxfun=1000000, ftol=1e-3,
                 xtol=1e-3, disp=False)

        p0 = [0.0, np.max(core_pix_vals), 5.0, 2.0]
        core_p = fmin(residuals, p0, args=(core_pix_rad, core_pix_vals), maxiter=1000000, maxfun=1000000,
                      ftol=1e-3, xtol=1e-3, disp=False)
    except OverflowError:
        return 0, 0

    _core = core_p[2] * ps
    _halo = p[2] * ps
    return _core, _halo


def make_circle(radius, central_obstruction, posx=512 / 2, posy=512 / 2, size=512, image=False, loc=None):
    """Make an image of a circle and save to a fits file.
    Inputs:
        - radius of circle in units of pixels
        - pos = 'center' if position of center of circle is center of detector,
            can put anything else if center position is not desired,
            define desired center position with xc and yc
        - size of one side of 2D array in units of pixels (default is 512)
        """
    detector = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - posx + 0.5) ** 2 + (j - posy + 0.5) ** 2)
            if central_obstruction <= dist <= radius:
                detector[j, i] = 1.0

    if image:
        hdu = pyfits.PrimaryHDU(detector)
        hdu.writeto(os.path.join(loc, 'circle.fits'), clobber=True)

    return detector


def make_gaussian(width, posx=512 / 2, posy=512 / 2, size=512, image=False, loc=None):
    """Make an image of a 2D gaussian and save to a fits file.
    Inputs:
        - standard deviation width of gaussian in units of pixels
        - posx and posy: position of center of circle in pixels
        - size of one side of 2D array in units of pixels (default is 512)
        - image: if True will save image in fits file (default False)
        """
    detector = np.zeros((size, size))

    def _gaussian(x):
        __gauss = np.e ** (-x ** 2 / (2. * width ** 2))
        return __gauss

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - posx) ** 2 + (j - posy) ** 2)
            _gauss = _gaussian(dist)
            detector[j, i] = _gauss

    if image:
        hdu = pyfits.PrimaryHDU(detector)
        hdu.writeto(os.path.join(loc, 'gaussian.fits'), clobber=True)

    return detector


def make_PSF(filled_circle, image=False, loc=None):
    FT_circle = fftpack.fft2(filled_circle)  # np.fft.fft2(filled_circle)
    F2 = fftpack.fftshift(FT_circle)
    oversampled_PSF = np.abs(F2) ** 2  # np.abs(np.real(F2))**2

    if image:
        hdu = pyfits.PrimaryHDU(oversampled_PSF)
        hdu.writeto(os.path.join(loc, 'FT_circle.fits'), clobber=True)
        np.savez(os.path.join(loc, 'FT_circle.npz'), PSF=oversampled_PSF)

    return oversampled_PSF


def calc_fwhm(array):
    def _gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x, y: height * np.exp(-(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

    def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        _params = moments(data)
        errorfunction = lambda p: np.ravel(_gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, _params)
        return p

    params = fitgaussian(array)
    sigx = params[3]
    sigy = params[4]
    fwhmx = 2 * np.sqrt(2 * np.log(2)) * sigx
    fwhmy = 2 * np.sqrt(2 * np.log(2)) * sigy
    fwhm = np.max([fwhmx, fwhmy])
    print("- fwhm =", fwhm, "pixels")

    return fwhm  # , params


def calc_fwhm1(array):
    # find peak  of model PSF
    fm_ind = np.where(array == np.max(array))
    fullmax = np.max(array)
    fm_ind0, fm_ind1 = fm_ind[0][0], fm_ind[1][0]
    print("- full-max =", fullmax, "located at:", fm_ind0, ",", fm_ind1, "pixels")

    # Half-max value:
    halfmax = fullmax / 2.
    print("- half-max =", halfmax)

    # indices full array:
    ind1_array = np.arange(0, fm_ind0 + 1)
    ind2_array = np.arange(0, fm_ind1 + 1)

    # interpolation function:
    f1 = interp1d(array[fm_ind0, 0:fm_ind1 + 1], ind1_array)
    hm_ind0 = f1(halfmax)

    f2 = interp1d(array[0:fm_ind0 + 1, fm_ind1], ind2_array)
    hm_ind1 = f2(halfmax)

    print("- located at:", hm_ind0, ",", hm_ind1, "pixels")

    fwhm0 = 2 * np.abs(hm_ind0 - fm_ind1)
    fwhm1 = 2 * np.abs(hm_ind1 - fm_ind0)
    fwhm = np.max([fwhm0, fwhm1])
    print("- fwhm =", fwhm, "pixels")

    return fwhm


def radius_fwhm_variation(fwhm_needed, obstructed=False):
    if obstructed:
        # radius_fwhm_file = open(loc+'radius_fwhm_obstructed_Pal_256.dat')
        radius_fwhm_file = open('radius_fwhm_obstructed_Pal_1024.dat')
    else:
        # radius_fwhm_file = open(loc+'radius_fwhm_Pal_256.dat')
        # radius_fwhm_file = open('radius_fwhm_2048.dat')
        radius_fwhm_file = np.loadtxt('radius_fwhm_2048.dat', delimiter=',')

    radius_ratio = radius_fwhm_file[:, 0]
    fwhm_ratio = radius_fwhm_file[:, 1]
    """
    l = 0
    for line in radius_fwhm_file:
        l += 1
        # skip the first row, which explains the columns
        if l == 1:
            continue
        else:
            row = line.split(",")
            radius_ratio = np.append(radius_ratio,float(row[0]))
            fwhm_ratio = np.append(fwhm_ratio,float(row[1]))
    """
    f_findfwhm = interp1d(fwhm_ratio, radius_ratio)
    radius_ratio = f_findfwhm(fwhm_needed)

    return radius_ratio


def makebox(array, halfwidth, peak1, peak2):
    boxside1a = peak1 - halfwidth
    boxside1b = peak1 + halfwidth
    boxside2a = peak2 - halfwidth
    boxside2b = peak2 + halfwidth

    box = array[boxside1a:boxside1b, boxside2a:boxside2b]
    box_fraction = np.sum(box) / np.sum(array)
    print('box has: {:.2f}% of light'.format(box_fraction * 100))

    return box, box_fraction


def ReplaceText(filename, SearchLine, ReplaceLine):
    """
    PURPOSE:
        Finds a line beginning with a given string in a text file and
        replaces the whole line with a new string
    NOTE:
        - The search string does not have to be a full line in the text file,
          but must start from the beginning.
        - Also, the replacing line must be a full line (including newline character)
    OUTPUTS:
        either 'Replaced!' or 'Not found!'

    example: ReplaceText('test.txt',"Find line with this string", "Replace whole line with this string\n")
    """

    with open(filename, 'r') as f:

        lines = []
        heres = []
        i = -1
        for _l in f:
            i += 1
            lines.append(_l)
            ind = len(SearchLine)
            find = np.where(_l[:ind] == SearchLine)  # [:23]
            if find[0] == [0]:
                here = i
            else:
                here = 'n/a'
            heres.append(here)

    textreturn = 'Not found!'

    for h in heres:
        if h != 'n/a':

            lines[h] = ReplaceLine

            with open(filename, 'w') as ff:
                for ll in lines:
                    ff.write(ll)

            textreturn = 'Replaced!'

        else:
            continue

    return textreturn


def cutout(_d, _output_dir):
    pix = 256
    scale = 2

    _img_name = os.path.join(_d, '100p.fits')

    # load the pipeline settings to get the guide star posn
    gs_region_next = False

    targ_x = 0
    targ_y = 0

    with open(os.path.join(os.path.split(_img_name)[0], 'pipeline_settings.txt'), 'r') as _f:
        f_lines = _f.readlines()
    f_lines = [_l for _l in f_lines if len(_l.strip()) > 0 and _l[0] != '#']
    for _l in f_lines:
        if _l.find("[Strehls]") == 0:
            gs_region_next = True
        else:
            if gs_region_next:
                x1, y1, x2, y2 = _l.split()[2].replace("(", "").replace(")", "").split(",")
                targ_x = (int(x1) + int(x2)) / 2
                targ_y = (int(y1) + int(y2)) / 2
                gs_region_next = False

    print(targ_x, targ_y)

    targ_x *= scale
    targ_y *= scale

    inf = pyfits.open(_img_name)[0].data
    # gain = pyfits.open(img_name)[0].header['AO_GAIN']
    header_check = 'MAGNITUD' in pyfits.open(_img_name)[0].header.keys()
    if header_check:
        _mag = pyfits.open(_img_name)[0].header['MAGNITUD']
    else:
        _mag = '?'

    # center in aperture
    maxv = 0.0

    max_x = 0.0
    max_y = 0.0
    max_v = -1e9
    for x in range(targ_x - 50, targ_x + 50):
        for y in range(targ_y - 50, targ_y + 50):
            try:
                if inf[y, x] > max_v:
                    max_v = inf[y, x]
                    max_x = x
                    max_y = y
            except:
                pass
    print(max_x, max_y)

    if max_x == 0 or max_y == 0:
        out_img_fn = 'Problem'

    else:
        print(max_v)
        out_img = np.ones((pix, pix), dtype=np.float32) * -10.0

        for x in range(max_x - pix / 2, max_x + pix / 2):
            for y in range(max_y - pix / 2, max_y + pix / 2):
                if 0 <= x < inf.shape[1] and 0 <= y < inf.shape[0]:
                    out_img[y - max_y + pix / 2, x - max_x + pix / 2] = inf[y, x]

        out_img_fn = os.path.join(_output_dir, os.path.basename(_d) + '.fits')
        print(out_img_fn)
        img_file = pyfits.PrimaryHDU(out_img)
        # img_file.header['AO_GAIN'] = gain
        img_file.header['MAGNITUD'] = _mag
        img_file.writeto(out_img_fn, clobber=True)

    return out_img_fn


def Strehl_calculator(name, imagepath, Strehl_factor, plate_scale, boxsize, newloc='n/a',
                      saveimage=False, save=False):

    """ Calculates the Strehl ratio of an image
    Inputs:
        - name: Target name as a string (no spaces, will be used as file name to save images, text, etc)
        - imagepath: full path and full image file name
        - Strehl_factor: from model PSF
        - boxsize: from model PSF
        - plate_scale: plate scale of telescope in arcseconds/pixel
        - newloc = path to where images and outputs should be saved, (needed if save = True) [default:'n/a']
        - save = True, saves images and output text to newloc  [default: False]
    Output:
        Strehl ratio (as a decimal)

    Example:
        Strehl_calculator('HR8799','/path/to/data/file/target1_20160304_100p.fits',765,1.524,0.578,0.0215,
                          newloc='/other/path/where/tosave/',save=True)
        """

    if save:
        saveout = sys.stdout
        output = open(newloc + name + '_Strehl_info.txt', 'a')
        sys.stdout = output
    else:
        saveout = None
        output = None

    print('-------- Real Image --------')

    # load real data image:
    image_data = pyfits.getdata(imagepath)
    mag = pyfits.open(imagepath)[0].header['MAGNITUD']

    ########################
    # ####### STEP 5 ######## normalize real image PSF by the flux in some radius
    ########################

    ##################################################
    #  Choose radius with 95-99% light in model PSF ##
    ##################################################

    # find peak image flux to center box around
    peak_ind = np.where(image_data == np.max(image_data))
    peak1, peak2 = peak_ind[0][0], peak_ind[1][0]
    print("max intensity =", np.max(image_data), "located at:", peak1, ",", peak2, "pixels")

    # find array within desired radius
    box_roboao, box_roboao_fraction = makebox(image_data, round(boxsize / 2.), peak1, peak2)
    print("size of box", np.shape(box_roboao))
    # box_roboao_check, box_roboao_fraction_check = sf.makebox(image_data,round(boxsize_check/2.),peak1,peak2)
    # print "size of box", np.shape(box_roboao_check), "(check)"#, "edges at:", edge1a,edge1b,edge2a,edge2b

    if saveimage:
        hdu3 = pyfits.PrimaryHDU(box_roboao)
        hdu3.writeto(newloc + name + '_box.fits', clobber=True)

    # sum the fluxes within the desired radius
    total_box_flux = np.sum(box_roboao)
    print("total flux in box", total_box_flux)
    # total_box_flux_check = np.sum(box_roboao_check)
    # print "total flux in box" , total_box_flux_check, "(check)"

    # normalize real image PSF by the flux in some radius:
    image_norm = image_data / total_box_flux
    # image_norm_check = image_data/total_box_flux_check

    ########################
    # ####### STEP 6 ######## divide normalized peak image flux by strehl factor
    ########################

    image_norm_peak = np.max(image_norm)
    print("normalized peak", image_norm_peak)
    # image_norm_peak_check = np.max(image_norm_check)
    # print "normalized peak", image_norm_peak_check, "(check)"

    #####################################################
    # ############# CALCULATE STREHL RATIO ###############
    #####################################################
    Strehl_ratio = image_norm_peak / Strehl_factor
    print('\n----------------------------------')
    print("Strehl ratio", Strehl_ratio * 100, '%')
    print("----------------------------------")

    # Strehl_ratio = image_norm_peak / Strehl_factor1_check
    # print "Strehl ratio", Strehl_ratio*100, '%, (check - model)'
    # Strehl_ratio = image_norm_peak_check / Strehl_factor1_check
    # print "Strehl ratio", Strehl_ratio*100, '%, (check - both)'
    # Strehl_ratio = image_norm_peak_check / Strehl_factor1
    # print "Strehl ratio", Strehl_ratio*100, '%, (check - image)'
    # print " "

    y, x = np.mgrid[:len(box_roboao), :len(box_roboao)]
    max_inds = np.where(box_roboao == np.max(box_roboao))

    g_init = models.Gaussian2D(amplitude=1., x_mean=max_inds[1][0], y_mean=max_inds[0][0],
                               x_stddev=1., y_stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = fit_g(g_init, x, y, box_roboao)

    sig_x = g.x_stddev[0]
    sig_y = g.y_stddev[0]

    FWHM = 2.3548 * np.mean([sig_x, sig_y])
    fwhm_arcsec = FWHM * plate_scale

    # fwhm_image = sf.calc_fwhm(box_roboao- np.median(box_roboao))
    # fwhm_arcsec = fwhm_image*plate_scale
    print('image FWHM: {:.5f}\"\n'.format(fwhm_arcsec))
    # fwhm_image_interp = sf.calc_fwhm1(box_roboao- np.median(box_roboao))
    # fwhm_arcsec_interp = fwhm_image_interp*plate_scale
    # print "image FWHM:", fwhm_arcsec_interp, '" (interpolated)'

    if save:
        sys.stdout = saveout
        output.close()

    return Strehl_ratio, fwhm_arcsec, mag


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Becky\'s PCA pipeline')

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

    # path to (standard) pipeline data:
    path_pipe = config.get('Path', 'path_pipe')
    # path to website static data:
    path_to_website_data = config.get('Path', 'path_to_website_data')
    # path to model PSFs:
    path_model_psf = config.get('Path', 'path_model_psf')

    # try processing today if no date provided
    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')
    date_str = datetime.datetime.strftime(date, '%Y%m%d')

    ''' go-go-go '''
    dirs = []
    for flux in ('high_flux', 'faint'):
        dirs += glob.glob(os.path.join(path_pipe, date_str, flux, '[!pointing]*_VIC_[S,l][i,r,g,z,p]*'))

    output_dir = os.path.join(path_to_website_data, date_str, 'strehl')
    output_dir_cuts = os.path.join(output_dir, 'tmp')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir_cuts):
        os.mkdir(output_dir_cuts)

    # Select telescope by date of obs:
    if date < datetime.datetime(2015, 9, 1):
        telescope = 'Palomar'
    else:
        telescope = 'KittPeak'
    # convert string from config into dictionary
    telescope_data = ast.literal_eval(config.get('Strehl', telescope))

    # Create output file:
    output_file = open(os.path.join(output_dir, 'SR_{:s}_all.dat'.format(date_str)), 'w')
    output_file.write(
        '# Object Name, Strehl Ratio [%], Core [arcsec], Halo [arcsec],FWHM [arcsec], Magnitude, Flags, Date \n')

    # Targets to be skipped:
    skip_target = ['']

    # loading model PSF Strehl factors:
    Strehlf = dict()
    l = 0
    for line in open(os.path.join(path_model_psf, 'Strehl_factors.dat')):
        l += 1
        if l == 1:
            continue
        else:
            if line.split(',')[3] == 'Palomar':
                Strehlf[line.split(',')[0] + '_Pal'] = [float(line.split(',')[1]), float(line.split(',')[2])]
            else:
                Strehlf[line.split(',')[0]] = [float(line.split(',')[1]), float(line.split(',')[2])]

    # Targets already calculated:
    previous_target = []
    for l in open(os.path.join(output_dir, 'SR_{:s}_all.dat'.format(date_str)), 'r'):
        if l[0] != '#':
            previous_target.append(l.split('    ')[0])
        else:
            continue

    # for line in open(output_dir + 'Time_log.txt', 'r'):
    #     if line[:3] != 'Pal' or line[:5] != 'Start' or line[:7] != 'Skipped' or line[:2] != '--':
    #         previous_target.append(line.split(' ')[0])
    #     else:
    #         continue

    # Create time log:
    # time_log = open(os.path.join(output_dir, 'Time_log.txt'), 'a')
    # time_log.write('-----------------------------------------\n')
    # time_log.write(telescope + ' data (' + str(len(previous_target)) + ' done out of ' + str(len(dirs)) + ')\n')
    # time_log.write('Start: ' + str(datetime.datetime.now()) + ', Pacific Time\n')

    # Running through all the desired directories:
    count_seeing = 0
    for d in dirs:

        try:
            img_name = os.path.basename(d)

            if img_name in previous_target or img_name in skip_target:
                continue

            elif img_name[:6] == 'seeing':
                count_seeing += 1
                countseeingline = 'Skipped seeing: ' + str(count_seeing) + ' images\n'
                ReplaceText(output_dir + 'Time_log.txt', 'Skipped seeing: ', countseeingline)

            elif '100p.fits' in os.listdir(d):
                print("*****************************")

                print(img_name)

                # procedure to identify filter used (i-band, g-band, etc?)
                if '_Si_' in img_name:
                    band = 'iband'
                elif '_Sg_' in img_name:
                    band = 'gband'
                elif '_Sr_' in img_name:
                    band = 'rband'
                elif '_Sz_' in img_name:
                    band = 'zband'
                elif '_lp600_' in img_name:
                    band = 'lp600'
                else:
                    print('Unable to guess filter')
                    continue

                # selecting correct Strehl factor and plate scale
                if '_VIC_' in img_name:  # visual camera?
                    # drizzled?
                    if pyfits.open(os.path.join(d, '100p.fits'))[0].data.shape[0] == 2048:
                        plate_scale = telescope_data['scale_red']
                    else:
                        plate_scale = telescope_data['scale']
                else:  # IR camera?
                    plate_scale = telescope_data['scale_IR']

                if telescope == 'KittPeak':
                    Strehl_factor = Strehlf[band]
                elif telescope == 'Palomar':
                    Strehl_factor = Strehlf[band + '_Pal']

                img_path = cutout(d, output_dir_cuts)

                if img_path == 'Problem':
                    # time_log.write(img_name + ' max x and/or y = 0 in cutout\n')
                    continue

                else:
                    try:
                        img = pyfits.open(img_path)[0].data
                        core, halo = bad_obs_check(img, ps=plate_scale)
                    except:
                        core = 0.139
                        halo = 1.1
                    if core >= 0.14 and halo <= 1.0:
                        flag = 'OK'
                    else:
                        flag = 'BAD?'

                    boxsize = int(round(3. / plate_scale))
                    SR, FWHM, mag = Strehl_calculator(img_name, img_path, Strehl_factor[0], plate_scale, boxsize,
                                                      newloc=output_dir, saveimage=False, save=False)

                    output_entry = '{:70s} {:8.5f} {:16.13f} {:16.13f} {:16.13f}  {:6.3f} {:s} {:s}\n'.\
                        format(img_name, SR * 100, core, halo, FWHM, mag, flag, date_str)
                    output_file.write(output_entry)

                    # remove tmp file:
                    os.remove(img_path)

            else:
                continue
        except Exception as e:
            print(e)
            continue

    output_file.close()
    # time_log.write('Finish: ' + str(datetime.datetime.now()) + ', Pacific Time\n')
    # time_log.close()
