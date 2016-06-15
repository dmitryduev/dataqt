"""
    Data Quality Monitoring for Robo-AO

    Generates stuff to be displayed on the DQM web-site

    DAD (Caltech), RMJC (Caltech), MS (UH) 2016
"""
from __future__ import print_function
from dead_times import dead_times
import inspect
from astropy.io import fits
import json
import argparse
import ConfigParser
import os
import shutil
import datetime
from collections import OrderedDict
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sewpy
from skimage import exposure, img_as_float
from skimage.morphology import disk
from skimage.filters import rank
# from skimage.transform import rescale
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, TextArea
import seaborn as sns
sns.set_style('whitegrid')
# sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=7))
plt.close('all')
sns.set_context('talk')


# some metrics to score SExtractor solutions.
# there are 2 of them, so them sum up to a max of 1
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


def make_img(_sou_name, _time, _filter, _prog_num, _camera, _marker,
             _path_sou, _path_data, pipe_out_type, _program_num_planets=24,
             _sr='none'):
    """

    :param _sou_name:
    :param _time:
    :param _filter:
    :param _prog_num:
    :param _camera:
    :param _marker:
    :param _path_sou:
    :param _path_data:
    :param pipe_out_type: 'high_flux' or 'faint'
    :param _program_num_planets:
    :param _sr:
    :return:
    """
    # read, process and display fits:
    hdulist = fits.open(os.path.join(_path_sou, '100p.fits'))
    scidata = hdulist[0].data
    # header:
    header = OrderedDict()
    for entry in hdulist[0].header.cards:
        header[entry[0]] = entry[1:]
    # print(header)

    # extract sources:
    sew = sewpy.SEW(
            # params=["X_IMAGE", "Y_IMAGE", "X2_IMAGE", "Y2_IMAGE",
            #         "A_IMAGE", "B_IMAGE", "FLUX_APER(3)", "FLAGS",
            #         "FWHM_IMAGE", "VIGNET"],
            params=["X_IMAGE", "Y_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE",
                    "A_IMAGE", "B_IMAGE",
                    "FWHM_IMAGE", "FLAGS"],
            config={"DETECT_MINAREA": 10, "PHOT_APERTURES": "10", 'DETECT_THRESH': '5.0'},
            sexpath="sex")
    # By default, this assumes that SExtractor can be called as "sex"
    # If this is not the case, or if the executable is not in your path,
    # specify the path by adding the argument sexpath="/path/to/sextractor"
    # to the above instantiation.

    out = sew(os.path.join(_path_sou, '100p.fits'))
    # sort according to FWHM
    out['table'].sort('FWHM_IMAGE')
    # descending order
    out['table'].reverse()

    # print(out['table'])  # This is an astropy table.

    # get first 5 and score them:
    scores = []
    for sou in out['table'][0:10]:
        if sou['FWHM_IMAGE'] > 1:
            score = log_gauss_score(sou['FWHM_IMAGE']) + gauss_score(rho(sou['X_IMAGE'], sou['Y_IMAGE']))
        else:
            score = 0  # it could so happen that reported FWHM is 0
        scores.append(score)

    # print(scores)

    # create a plot

    # bokeh
    # s = figure(x_range=(0, 10), y_range=(0, 10), width=700, plot_height=700, title=None)
    # s.image(image=[scidata], x=0, y=0, dw=10, dh=10, palette="Spectral11")
    # show(s)
    # raise Exception('hola!')

    # seaborn
    norm = np.max(np.max(scidata))
    mask = scidata <= 0
    scidata[mask] = 0
    scidata = np.uint16(scidata/norm*65535)
    # logarithmic_corrected = exposure.adjust_log(scidata, 1)
    # scidata_corrected = logarithmic_corrected
    # print(np.min(np.min(scidata)), np.max(np.max(scidata)))

    # don't do histogram equalization for planets:
    if _sou_name.lower() in ('mars', 'venus', 'jupiter', 'saturn'):
        p_1, p_2 = np.percentile(scidata, (8, 100))
        scidata_corrected = exposure.rescale_intensity(scidata, in_range=(p_1, p_2))
    #
    else:
        # Equalization
        # selem = disk(30)
        # scidata_corrected = rank.equalize(scidata, selem=selem)
        scidata_corrected = exposure.equalize_adapthist(scidata, clip_limit=0.03)

    ''' plot full image '''
    plt.close('all')
    fig = plt.figure(_sou_name+'__full')
    fig.set_size_inches(4, 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot detected objects:
    # ax.plot(out['table']['X_IMAGE']-1, out['table']['Y_IMAGE']-1, 'o',
    #         markeredgewidth=1, markerfacecolor='None', markeredgecolor=plt.cm.Oranges(0.8))
    # ax.imshow(scidata, cmap='gray', origin='lower', interpolation='nearest')
    ax.imshow(scidata_corrected, cmap='gray', origin='lower', interpolation='nearest')
    # ax.imshow(scidata, cmap='gist_heat', origin='lower', interpolation='nearest')
    # plt.axis('off')
    plt.grid('off')

    # save full figure
    fname_full = '{:d}_{:s}_{:s}_{:s}_{:s}_{:s}_full.png'.format(_prog_num, _sou_name, _camera, _filter, _marker,
                                                  datetime.datetime.strftime(_time, '%Y%m%d_%H%M%S.%f'))
    plt.savefig(os.path.join(_path_data, pipe_out_type, fname_full), dpi=300)

    ''' crop the brightest detected source: '''
    N_sou = len(out['table'])
    # do not crop large planets and crowded fields
    if (_prog_num != _program_num_planets) and (N_sou != 0 and N_sou < 30):
        # sou_xy = [out['table']['X_IMAGE'][0], out['table']['Y_IMAGE'][0]]
        best_score = np.argmax(scores) if len(scores) > 0 else 0
        sou_size = np.max((int(out['table']['FWHM_IMAGE'][best_score] * 3), 90))
        scidata_corrected_cropped = scidata_corrected[out['table']['YPEAK_IMAGE'][best_score] - sou_size/2:
                                                      out['table']['YPEAK_IMAGE'][best_score] + sou_size/2,
                                                      out['table']['XPEAK_IMAGE'][best_score] - sou_size/2:
                                                      out['table']['XPEAK_IMAGE'][best_score] + sou_size/2]
    else:
        scidata_corrected_cropped = scidata_corrected
    # save cropped image
    fig = plt.figure(_sou_name)
    fig.set_size_inches(3, 3, forward=False)
    # ax = fig.add_subplot(111)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(scidata_corrected_cropped, cmap='gray', origin='lower', interpolation='nearest')
    # add scale bar:
    # draw a horizontal bar with length of 0.1*x_size
    # (ax.transData) with a label underneath.
    bar_len = scidata_corrected_cropped.shape[0]*0.1
    bar_len_str = '{:.1f}'.format(bar_len*36/1024/2)
    asb = AnchoredSizeBar(ax.transData,
                          bar_len,
                          bar_len_str[0] + r"$^{\prime\prime}\!\!\!.$" + bar_len_str[-1],
                          loc=4, pad=0.3, borderpad=0.5, sep=10, frameon=False)
    ax.add_artist(asb)
    # add Strehl ratio
    if _sr != 'none':
        asb2 = AnchoredSizeBar(ax.transData,
                               0,
                               'Strehl: {:.2f}%'.format(float(_sr)),
                               loc=2, pad=0.3, borderpad=0.4, sep=5, frameon=False)
        ax.add_artist(asb2)
        # asb3 = AnchoredSizeBar(ax.transData,
        #                        0,
        #                        'SR: {:.2f}%'.format(float(_sr)),
        #                        loc=3, pad=0.3, borderpad=0.5, sep=10, frameon=False)
        # ax.add_artist(asb3)

    # save cropped figure
    fname_cropped = '{:d}_{:s}_{:s}_{:s}_{:s}_{:s}_cropped.png'.format(_prog_num, _sou_name, _camera, _filter,
                                                                            _marker, datetime.datetime.strftime(_time,
                                                                                             '%Y%m%d_%H%M%S.%f'))
    fig.savefig(os.path.join(_path_data, pipe_out_type, fname_cropped), dpi=300)

    return header


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Data quality monitoring for Robo-AO')

    parser.add_argument('config_file', metavar='config_file',
                        action='store', help='path to config file.', type=str)
    parser.add_argument('--date', metavar='date', action='store', dest='date',
                        help='obs date', type=str)

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

    # planetary program number (do no crop planetary images!)
    program_num_planets = int(config.get('Programs', 'planets'))
    # path to (standard) pipeline data:
    path_pipe = config.get('Path', 'path_pipe')
    # path to Becky-pipeline data:
    path_pca = config.get('Path', 'path_pca')
    # path to seeing plots:
    path_seeing = config.get('Path', 'path_seeing')

    # website data dwelling place:
    # path_to_website_data = os.path.join(abs_path, 'static', 'data')
    path_to_website_data = config.get('Path', 'path_to_website_data')

    if not args.date:
        now = datetime.datetime.now()
        date = datetime.datetime(now.year, now.month, now.day)
    else:
        date = datetime.datetime.strptime(args.date, '%Y%m%d')

    ''' Scientific images, PCA, contrast curves, Strehl, and aux data '''
    # date string
    date_str = datetime.datetime.strftime(date, '%Y%m%d')

    # path to pipelined data exists?
    if os.path.exists(os.path.join(path_pipe, date_str)):
        # puttin' all my eeeeeggs in onnne...baasket!
        # source_data = {'high_flux': [], 'faint': [], 'zero_flux': [], 'failed': []}
        source_data = OrderedDict((('high_flux', []), ('faint', []),
                                   ('zero_flux', []), ('failed', [])))
        # path to output
        path_data = os.path.join(path_to_website_data, date_str)
        if not os.path.exists(path_to_website_data):
            os.mkdir(path_to_website_data)
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        ''' make nightly joint Strehl ratio plot '''
        # store source data
        SR = OrderedDict()
        path_strehl = os.path.join(path_to_website_data, date_str, 'strehl')
        # check if exists and parse SR_{date_str}_all.dat
        if os.path.isfile(os.path.join(path_strehl, 'SR_{:s}_all.dat'.format(date_str))):
            print('Generating Strehl plot for {:s}'.format(date_str))
            with open(os.path.join(path_strehl, 'SR_{:s}_all.dat'.format(date_str)), 'r') as f:
                f_lines = f.readlines()

            f_lines = [l.split() for l in f_lines if l[0] != '#']

            if len(f_lines) > 0:

                for line in f_lines:
                    tmp = line[0].split('_')
                    time = datetime.datetime.strptime(tmp[-2] + tmp[-1], '%Y%m%d%H%M%S.%f')
                    SR[line[0]] = [time] + line[1:]

                SR_good = np.array([(v[0], float(v[1])) for k, v in SR.iteritems() if v[-2] == 'OK'])
                SR_notgood = np.array([(v[0], float(v[1])) for k, v in SR.iteritems() if v[-2] != 'OK'])

            if len(SR_good) > 0 or len(SR_notgood) > 0:
                fig = plt.figure('Strehls for {:s}'.format(date_str), figsize=(7, 3.18), dpi=200)
                ax = fig.add_subplot(111)

                if len(SR_good) > 0:
                    # sort by time stamps:
                    SR_good = SR_good[SR_good[:, 0].argsort()]
                    good = True
                    SR_mean = np.mean(SR_good[:, 1])
                    SR_std = np.std(SR_good[:, 1])
                    SR_max = np.max(SR_good[:, 1])
                    SR_min = np.min(SR_good[:, 1])

                    ax.plot(SR_good[:, 0], SR_good[:, 1], 'o', color=plt.cm.Oranges(0.6), markersize=6)

                    ax.axhline(y=SR_mean, linestyle='-', color='teal', linewidth=1,
                               label='mean = ' + str(round(SR_mean, 2)) + '%')
                    ax.axhline(y=SR_mean + SR_std, linestyle='--', color='teal', linewidth=1,
                               label=r'$\sigma _{SR}$ = ' + str(round(SR_std, 2)) + '%')
                    ax.axhline(y=SR_mean - SR_std, linestyle='--', color='teal', linewidth=1)

                if len(SR_notgood) > 0:
                    SR_notgood = SR_notgood[SR_notgood[:, 0].argsort()]
                    ax.plot(SR_notgood[:, 0], SR_notgood[:, 1], 'o', color=plt.cm.Greys(0.6), markersize=6)

                # ax.set_xlabel('Time, UTC')
                # xstart = np.min([SR_notgood[0, 0], SR_good[0, 0]]) - datetime.timedelta(minutes=15)
                # xstop = np.max([SR_notgood[-1, 0], SR_good[-1, 0]]) + datetime.timedelta(minutes=15)
                # ax.set_xlim([xstart, xstop])
                ax.set_ylabel('Strehl Ratio, %')
                # ax.legend(bbox_to_anchor=(1.35, 1), ncol=1, numpoints=1, fancybox=True)
                ax.legend(loc='best', numpoints=1, fancybox=True)
                ax.grid(linewidth=0.5)
                ax.margins(0.05, 0.2)

                myFmt = mdates.DateFormatter('%H:%M')
                ax.xaxis.set_major_formatter(myFmt)
                fig.autofmt_xdate()

                plt.tight_layout()
                # fig.subplots_adjust(right=0.75)

                fig.savefig(os.path.join(path_strehl, 'SR_{:s}_all.png'.format(date_str)), dpi=200)

                # save in json
                source_data['strehls'] = True
            else:
                source_data['strehls'] = False

        # contrast curves in txt format:
        ccs = []

        print('Generating images for {:s}'.format(date_str))
        for pot in source_data.keys():
            if os.path.exists(os.path.join(path_pipe, date_str, pot)):
                print(pot.replace('_', ' ').title())
                if not os.path.exists(os.path.join(path_data, pot)) \
                        and pot not in ('zero_flux', 'failed'):
                    os.mkdir(os.path.join(path_data, pot))
                # path to pca data exists?
                if os.path.exists(os.path.join(path_pca, date_str, pot)):
                    pca_ls = sorted(os.listdir(os.path.join(path_pca, date_str, pot)))
                    ccs += [os.path.join(path_pca, date_str, pot, pf) for pf in pca_ls
                            if '_contrast_curve.txt' in pf]
                else:
                    pca_ls = []
                for sou_dir in sorted(os.listdir(os.path.join(path_pipe, date_str, pot))):
                    path_sou = os.path.join(path_pipe, date_str, pot, sou_dir)
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
                    time = datetime.datetime.strptime(tmp[-2] + tmp[-1], '%Y%m%d%H%M%S.%f')
                    # camera:
                    camera = tmp[-5:-4][0]
                    # marker:
                    marker = tmp[-3:-2][0]

                    # contrast curve:
                    if (sou_dir + '_pca.png' in pca_ls) and (sou_dir + '_contrast_curve.png' in pca_ls) and \
                            (sou_dir + '_contrast_curve.txt' in pca_ls):
                        cc = 'pca'
                        f_contrast_curve = os.path.join(path_pca, date_str, pot,
                                                        sou_dir + '_contrast_curve.png')
                        shutil.copy(f_contrast_curve, os.path.join(path_data, pot))
                        f_pca = os.path.join(path_pca, date_str, pot, sou_dir + '_pca.png')
                        shutil.copy(f_pca, os.path.join(path_data, pot))
                    elif (sou_dir + '_NOPCA_contrast_curve.png' in pca_ls) and \
                            (sou_dir + '_NOPCA_contrast_curve.txt' in pca_ls):
                        cc = 'nopca'
                        f_contrast_curve = os.path.join(path_pca, date_str, pot,
                                                        sou_dir + '_NOPCA_contrast_curve.png')
                        shutil.copy(f_contrast_curve, os.path.join(path_data, pot))
                    else:
                        cc = 'none'

                    # we don't need contrast curves for pointing observations:
                    if 'pointing' in sou_name:
                        cc = 'none'

                    # Strehl ratio:
                    if sou_dir in SR:
                        sr = SR[sou_dir][1]
                        core = SR[sou_dir][2]
                        halo = SR[sou_dir][3]
                        fwhm = SR[sou_dir][4]
                        sr_flag = SR[sou_dir][5]
                    else:
                        sr = 'none'
                        core = 'none'
                        halo = 'none'
                        fwhm = 'none'
                        sr_flag = 'none'

                    # store
                    source = OrderedDict((('prog_num', prog_num), ('sou_name', sou_name),
                                          ('filter', filt),
                                          ('time', datetime.datetime.strftime(time, '%Y%m%d_%H%M%S.%f')),
                                          ('camera', camera), ('marker', marker),
                                          ('contrast_curve', cc),
                                          ('strehl_ratio', sr), ('core', core), ('halo', halo),
                                          ('fwhm', fwhm), ('sr_flag', sr_flag)))
                    print(dict(source))

                    if pot not in ('zero_flux', 'failed'):
                        try:
                            header = make_img(_sou_name=sou_name, _time=time, _filter=filt, _prog_num=prog_num,
                                              _camera=camera, _marker=marker,
                                              _path_sou=path_sou, _path_data=path_data, pipe_out_type=pot,
                                              _program_num_planets=program_num_planets, _sr=sr)
                            # dump header
                            source['header'] = header
                        except IOError:
                            continue

                    # put into the basket
                    source_data[pot].append(source)

        ''' make nightly joint contrast curve plot '''
        if len(ccs) > 0:
            print('Generating contrast curve summary for {:s}'.format(date_str))
            contrast_curves = []
            fig = plt.figure('Contrast curve', figsize=(8, 3.5), dpi=200)
            ax = fig.add_subplot(111)
            for f_cc in ccs:
                with open(f_cc, 'r') as f:
                    f_lines = f.readlines()
                cc_tmp = np.array([map(float, l.split()) for l in f_lines])
                if not np.isnan(cc_tmp[:, 1]).any():
                    contrast_curves.append(cc_tmp)
            contrast_curves = np.array(contrast_curves)

            # add to plot:
            sep_mean = np.linspace(0.2, 1.45, num=100)
            cc_mean = []
            for contrast_curve in contrast_curves:
                ax.plot(contrast_curve[:, 0], contrast_curve[:, 1], '-', c=plt.cm.Greys(0.3), linewidth=1.2)
                cc_mean.append(np.interp(sep_mean, contrast_curve[:, 0], contrast_curve[:, 1]))
            # add mean to plot:
            ax.plot(sep_mean, np.mean(np.array(cc_mean).T, axis=1), '-', c=plt.cm.Oranges(0.7), linewidth=2.5)
            # beautify and save:
            ax.set_xlim([0.2, 1.45])
            ax.set_xlabel('Separation [arcseconds]')  # , fontsize=18)
            ax.set_ylabel('Contrast [$\Delta$mag]')  # , fontsize=18)
            ax.set_ylim([0, 8])
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.grid(linewidth=0.5)
            plt.tight_layout()
            fig.savefig(os.path.join(path_to_website_data, date_str,
                                     'contrast_curve.{:s}.png'.format(date_str)), dpi=200)
            # save in json
            source_data['contrast_curve'] = True
        else:
            source_data['contrast_curve'] = False

        ''' compute dead times '''
        path_data_deadtimes = os.path.join(path_to_website_data, date_str, 'dead_times')
        if not os.path.exists(path_data_deadtimes):
            os.mkdir(path_data_deadtimes)
        print('Generating dead time plots for {:s}'.format(date_str))
        _, deadtimes_fig_names = dead_times(_path_pipe=path_pipe,
                                            _path_output=path_data_deadtimes, _date=date_str)
        if len(deadtimes_fig_names) > 0:
            source_data['dead_times'] = deadtimes_fig_names
        else:
            source_data['dead_times'] = []

        # dump sci json
        with open(os.path.join(path_data, '{:s}.json'.format(date_str)), 'w') as fp:
            # json.dump(source_data, fp, sort_keys=True, indent=4)
            json.dump(source_data, fp, indent=4)

    # plt.show()

    ''' Seeing '''
    path_seeing = os.path.join(path_seeing, 'plots', date_str)
    # print(path_seeing)

    # path exists?
    if os.path.exists(path_seeing):
        print('Generating seeing plots for {:s}'.format(date_str))
        # puttin' all my eeeeeggs in onnne...baasket!
        seeing_data = {'fimages': [], 'mean': 0, 'median': 0}
        # path to output
        path_data = os.path.join(path_to_website_data, date_str)
        if not os.path.exists(path_to_website_data):
            os.mkdir(path_to_website_data)
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        seeing_plot = []
        with open(os.path.join(path_seeing, 'seeing.{:s}.txt'.format(date_str))) as f:
            f_lines = f.readlines()
            f_lines = [l.split() for l in f_lines]
            for line in f_lines:
                estimate = float(line[3])
                if estimate > 0.4:
                    tstamp = datetime.datetime.strptime(' '.join(line[0:2]), '%Y-%m-%d %H:%M:%S')
                    seeing_plot.append((tstamp, estimate))

        # print(seeing_plot)
        seeing_plot = np.array(seeing_plot, dtype=[('t', 'S20'), ('seeing', float)])
        # sort
        seeing_plot = np.sort(seeing_plot, order='t')
        # convert back
        seeing_plot = np.array([[datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'), float(s)]
                                for t, s in seeing_plot])

        # make a robust fit to seeing data for visual reference
        t_seeing_plot = np.array([(t-seeing_plot[0, 0]).total_seconds() for t in seeing_plot[:, 0]])
        t_seeing_plot = np.expand_dims(t_seeing_plot, axis=1)

        # estimators = [('OLS', linear_model.LinearRegression()),
        #               ('Theil-Sen', linear_model.TheilSenRegressor(random_state=42)),
        #               ('RANSAC', linear_model.RANSACRegressor(random_state=42)), ]
        # estimators = [('OLS', linear_model.LinearRegression()),
        #               ('RANSAC', linear_model.RANSACRegressor()), ]
        estimators = [('RANSAC', linear_model.RANSACRegressor()), ]

        fig = plt.figure('Seeing data', figsize=(8, 3), dpi=200)
        ax = fig.add_subplot(111)
        # ax = plt.Axes(fig, [0.1, 0.2, 0.8, 0.8])
        # fig.add_axes(ax)
        # ax.set_title('2.1m vs 4m, WIYN, and 0.9m data from the nightly reports')
        # ax.plot(seeing_plot[:, 0], seeing_plot[:, 1], '-',
        #         c=plt.cm.Blues(0.82), linewidth=1.2, label='2.1m')
        # ax.plot(seeing_plot[:, 0], seeing_plot[:, 1], '--',
        #         c=plt.cm.Blues(0.82), linewidth=0.9, label='2.1m')
        ax.plot(seeing_plot[:, 0], seeing_plot[:, 1], '.',
                c=plt.cm.Oranges(0.82), markersize=10)  #, label='2.1m seeing measurements')
        # ax.set_ylim([0, 3])
        ax.set_ylabel('Seeing, arcsec')  # , fontsize=18)
        ax.grid(linewidth=0.5)

        # evaluate estimators
        try:
            for name, estimator in estimators:
                model = make_pipeline(PolynomialFeatures(degree=5), estimator)
                model.fit(t_seeing_plot, seeing_plot[:, 1])
                y_plot = model.predict(t_seeing_plot)
                ax.plot(seeing_plot[:, 0], y_plot, '--', label='Robust {:s} fit'.format(name))
            ax.legend(loc='best')
        except Exception as e:
            print(e)
            pass

        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()

        plt.tight_layout()

        # plt.show()
        f_seeing_plot = os.path.join(path_seeing, 'seeing.{:s}.png'.format(date_str))
        fig.savefig(f_seeing_plot, dpi=300)

        ''' make a local copy to display on the website '''
        path_data_seeing = os.path.join(path_to_website_data, date_str, 'seeing')
        if not os.path.exists(path_data_seeing):
            os.mkdir(path_data_seeing)

        shutil.copy(f_seeing_plot, path_data_seeing)

        f_seeing_txt = os.path.join(path_seeing, 'seeing.{:s}.txt'.format(date_str))
        shutil.copy(f_seeing_txt, path_data_seeing)

        seeing_images = sorted([fi for fi in os.listdir(path_seeing)
                                if ('seeing' not in fi) and ('.png' in fi)])
        for fsi in seeing_images:
            shutil.copy(os.path.join(path_seeing, fsi), path_data_seeing)

        # fill in the dict to be dumped to json:
        seeing_data['mean'] = '{:.3f}'.format(np.mean(seeing_plot[:, 1]))
        seeing_data['median'] = '{:.3f}'.format(np.median(seeing_plot[:, 1]))
        seeing_data['fimages'] = sorted([fi for fi in os.listdir(path_data_seeing)
                                         if ('seeing' not in fi) and ('.png' in fi)])

        # dump seeing json
        with open(os.path.join(path_data, '{:s}.seeing.json'.format(date_str)), 'w') as fp:
            json.dump(seeing_data, fp, sort_keys=True, indent=4)
